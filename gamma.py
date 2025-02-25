import cupy as cp
import numpy as np
from scipy.ndimage import map_coordinates
from cupyx.scipy.ndimage import map_coordinates as map_cp
from scipy import interpolate
import time

def interp3_resolution(sample, resolution, order=1):
    shape = np.array(sample.shape)
    new_shape = (shape - 1) * resolution + 1
    # 生成新坐标的索引
    grid = np.meshgrid(
        *(np.linspace(0, s - 1, ns) for s, ns in zip(shape, new_shape)),
        indexing='ij'
    )
    coords = np.array(grid)
    return map_coordinates(sample, coords, order=order)


def gamma_calc_resolution(reference, sample, dose_std='global', interpolation=True, resolution=(1, 1, 1), dis_tol=3,
                          dose_tol=0.02, threshold=0.1):
    # check for valid arguments
    if reference.shape != sample.shape:
        raise Exception("ValueError: shape mismatch: reference and sample must have the same shape")
    if dis_tol <= 0 or int(dis_tol) != dis_tol:
        raise Exception("ValueError: dis_tol is an integer greater than zero")
    if dose_tol <= 0 or dose_tol >= 1:
        raise Exception("ValueError: dose_tol is a float number between 0 (exclusive) and 1 (exclusive)")
    ndim = len(resolution)
    resolution = np.array(resolution, dtype=int)
    if interpolation:
        compare_sample = interp3_resolution(sample, resolution, order=3)
        sample_resolution = np.array([1 for _ in range(ndim)])
    else:
        compare_sample = sample
        sample_resolution = resolution
    pad_range = np.ceil(dis_tol / sample_resolution)
    pad_range = np.array(pad_range, dtype=int)

    sample = cp.array(sample)
    compare_sample = cp.array(compare_sample)
    reference = cp.array(reference)
    ex_sample = cp.pad(compare_sample, (tuple(zip(pad_range, pad_range))), 'reflect')

    # distance map
    dis_range = []
    for pad in pad_range:
        dis_range.append(np.arange(-pad, pad + 1, 1, dtype=float))
    _ = cp.array(np.meshgrid(*dis_range, indexing='ij'))
    for dim, res in enumerate(sample_resolution):
        _[dim] *= res
    dis_map = cp.sqrt(cp.sum(_ ** 2, axis=0))
    dis_mask = cp.sqrt(cp.sum(_ ** 2, axis=0)) >= dis_tol
    _sqDist = (dis_map / dis_tol) ** 2

    max_dose = cp.max(reference)
    dose_mask = np.bitwise_or(reference > max_dose * threshold, sample > max_dose * threshold)

    gamma = cp.zeros_like(reference)
    gamma[:] = cp.inf

    if ndim == 3:
        mz, mx, my = compare_sample.shape
        kz, kx, ky = cp.indices(dis_map.shape)
        for k, i, j in zip(kz.flatten(), kx.flatten(), ky.flatten()):
            idx = k, i, j
            move_sample = ex_sample[slice(k, k + mz, resolution[0]), slice(i, i + mx, resolution[1]), slice(j, j + my, resolution[2])]
            # skip masked voxels
            if dis_mask[idx]:
                continue
            # gamma index
            if dose_std == 'global':
                dose_standard = max_dose
            elif dose_std == 'local':
                dose_standard = reference
            else:
                raise Exception("ValueError: dose_std is not global or local!")
            _sqDose = ((move_sample - reference) / dose_standard / dose_tol) ** 2
            _gamma = cp.sqrt(_sqDist[idx] + _sqDose)
            _ = gamma > _gamma
            gamma[_] = _gamma[_]
        gamma_pass_rate = cp.sum(cp.bitwise_and(gamma <= 1, dose_mask)) / cp.sum(dose_mask)
        return gamma, gamma_pass_rate
    else:
        nx, ny = reference.shape
        kx, ky = cp.indices(dis_map.shape)
        for i, j in zip(kx.flatten(), ky.flatten()):
            idx = i, j
            move_sample = ex_sample[slice(i, i + nx), slice(j, j + ny)]
            # skip masked voxels
            if dis_mask[idx]:
                continue
            # gamma index
            if dose_std == 'global':
                dose_standard = max_dose
            elif dose_std == 'local':
                dose_standard = reference
            else:
                raise Exception("ValueError: dose_std is not global or local!")
            _sqDose = ((move_sample - reference) / dose_standard / dose_tol) ** 2
            _gamma = cp.sqrt(_sqDist[idx] + _sqDose)
            _ = gamma > _gamma
            gamma[_] = _gamma[_]
        gamma_pass_rate = cp.sum(cp.bitwise_and(gamma <= 1, dose_mask)) / cp.sum(dose_mask)
        return gamma, gamma_pass_rate


def gamma_calc_gradient(reference, sample, dose_std='global', resolution=(1, 1, 1), dis_tol=3, dose_tol=0.02, threshold=0.1):
    # 最高效，精度较高的算法，在sample中插值
    max_dose = np.max(reference)
    dose_mask = np.bitwise_or(reference > max_dose * threshold, sample > max_dose * threshold)
    reference = np.ma.array(reference, mask=~dose_mask)
    if dose_std == 'global':
        dose_tol = max_dose * dose_tol
    elif dose_std == 'local':
        dose_tol = reference * dose_tol
    else:
        raise ValueError('Not valid dose standard!')
    diff = sample - reference
    # grad = np.gradient(sample)
    n_dim = len(sample.shape)
    grad = []
    for i in range(n_dim):
        gi = np.diff(sample, axis=i)
        gi = np.concatenate(
            [np.expand_dims(np.take(gi, 0, i) * -np.take(np.sign(diff), 0, i), axis=i),
             np.max([-np.take(gi, range(0, gi.shape[i] - 1), i) * -np.take(np.sign(diff), range(1, gi.shape[i]), i),
                     np.take(gi, range(1, gi.shape[i]), i) * -np.take(np.sign(diff), range(1, gi.shape[i]), i)], axis=0),
             np.expand_dims(np.take(gi, -1, i) * -np.take(np.sign(diff), -1, i), axis=i)], axis=i)
        gi = np.max([gi, np.zeros_like(gi)], axis=0)
        grad.append(gi)
    sqGrad = sum([g ** 2 / res ** 2 for g, res in zip(grad, resolution)])
    D1 = (diff / dose_tol) ** 2
    D2 = sqGrad / dose_tol ** 2
    X = 1 / dis_tol ** 2
    gamma = np.sqrt(D1 * X / (X + D2))
    gamma_pass_rate = np.sum(gamma <= 1) / np.sum(dose_mask)
    return gamma, gamma_pass_rate

def gamma_calc_gradient2(reference, sample, dose_std='global', resolution=(1, 1, 1), dis_tol=2, dose_tol=0.03, threshold=0.1):
    max_dose = np.max(reference)
    dose_mask = np.bitwise_or(reference > max_dose * threshold, sample > max_dose * threshold)
    if dose_std == 'global':
        dose_tol = max_dose * dose_tol
    elif dose_std == 'local':
        dose_tol = reference * dose_tol
    else:
        raise ValueError('Not valid dose standard!')
    search_region = 8
    search_range = np.array(np.floor(search_region / np.array(resolution)), dtype=int)
    dis_range = []
    for dis in search_range:
        dis_range.append(list(range(-dis, dis + 1)))
    coords = np.array(np.meshgrid(*dis_range, indexing='ij'))
    dis_map = np.zeros_like(coords[0], dtype=float)
    for coord, res in zip(coords, resolution):
        dis_map += (coord * res) ** 2
    dis_map = np.sqrt(dis_map)
    dis_mask = dis_map <= search_region
    dis_map = dis_map[dis_mask]
    coords = [coord[dis_mask] for coord in coords]
    search_list = list(zip(dis_map, zip(*coords)))
    search_list.sort(key=lambda x: x[0])
    grad = np.gradient(sample)
    grad = np.stack([g / res for g, res in zip(grad, resolution)])
    # gamma_map = np.ones_like(reference) * np.inf
    # for dis, coord_bias in search_list:
    #     ref = reference
    #     samp = sample
    #     gd = grad
    #     pad_range = []
    #     for i, bias in enumerate(coord_bias):
    #         ref = np.take(ref, range(max(0, -bias), min(reference.shape[i] - bias, reference.shape[i])), i)
    #         samp = np.take(samp, range(max(0, bias), min(reference.shape[i] + bias, reference.shape[i])), i)
    #         gd = np.take(gd, range(max(0, bias), min(reference.shape[i] + bias, reference.shape[i])), i + 1)
    #         pad_range.append((max(0, -bias), max(0, bias)))
    #     diff = samp - ref
    #     X0 = np.array(coord_bias) * np.array(resolution)
    #     X0g = sum([x * g for x, g in zip(X0, gd)])
    #     d2g2 = np.sum(gd ** 2, axis=0) * (dis_tol / dose_tol) ** 2
    #     gamma = np.sqrt(((diff - X0g) ** 2 + 4 * X0g ** 2 * d2g2) / (1 + d2g2) / dose_tol ** 2)
    #     gamma = np.pad(gamma, pad_width=pad_range, mode='constant', constant_values=np.inf)
    #     gamma_map = np.min((gamma_map,gamma),axis=0)
    gamma_map = np.ones_like(reference) * np.inf
    for dis, coord_bias in search_list:
        ref = reference
        samp = sample
        pad_range = []
        gd = []
        for i, bias in enumerate(coord_bias):
            ref = np.take(ref, range(max(0, -bias), min(reference.shape[i] - bias, reference.shape[i])), i)
            samp = np.take(samp, range(max(0, bias), min(reference.shape[i] + bias, reference.shape[i])), i)
            pad_range.append((max(0, -bias), max(0, bias)))
        diff = samp - ref
        for i, bias in enumerate(coord_bias):
            gi = np.diff(sample, axis=i)
            if bias > 0:
                gi_side1 = np.concatenate([np.take(gi, range(bias, gi.shape[i]), i), np.expand_dims(np.take(gi, -1, i), axis=i)], axis=i)  # 正向梯度
                gi_side2 = np.take(gi, range(bias - 1, gi.shape[i]), i)  # 逆向梯度
            elif bias < 0:
                gi_side1 = np.take(gi, range(0, gi.shape[i] + bias + 1), i)
                gi_side2 = np.concatenate([np.expand_dims(np.take(gi, 0, i), axis=i), np.take(gi, range(0, gi.shape[i] + bias), i)], axis=i)
            else:
                gi_side1 = np.concatenate([gi, np.expand_dims(np.take(gi, -1, i), axis=i)], axis=i)
                gi_side2 = np.concatenate([np.expand_dims(np.take(gi, 0, i), axis=i), gi], axis=i)
            for j, other_bias in enumerate(coord_bias):
                if j != i:
                    gi_side1 = np.take(gi_side1, range(max(0, other_bias), min(reference.shape[j] + other_bias, reference.shape[j])), j)
                    gi_side2 = np.take(gi_side2, range(max(0, other_bias), min(reference.shape[j] + other_bias, reference.shape[j])), j)
            if bias == 0:
                gi = np.max([gi_side1 * -np.sign(diff), gi_side2 * np.sign(diff), np.zeros_like(gi_side1)], axis=0)
            elif bias > 0:
                gi = gi_side2
            else:
                gi = gi_side1
            gd.append(gi / resolution[i])
        X0 = np.array(coord_bias) * np.array(resolution)
        X0g = sum([x * g for x, g in zip(X0, gd)])
        d2g2 = np.sum(np.array(gd) ** 2, axis=0) * (dis_tol / dose_tol) ** 2
        X = np.array([(g - x) * bias for g, x, bias in
                      zip(-(diff + X0g) * (dis_tol / dose_tol) ** 2 * np.array(gd), X0, -np.sign(coord_bias) / np.array(resolution))])
        valid = np.bitwise_and(np.all(np.sign(X) >= 0, axis=0), np.all(X < 1, axis=0))
        gamma_point = np.sqrt(diff ** 2 / dose_tol ** 2 + np.sum(np.array(X0) ** 2, axis=0) / dis_tol ** 2)
        gamma = np.sqrt(((diff - X0g) ** 2 + 4 * X0g ** 2 * d2g2) / (1 + d2g2) / dose_tol ** 2)
        gamma = np.where(valid, gamma, gamma_point)
        gamma = np.pad(gamma, pad_width=pad_range, mode='constant', constant_values=np.inf)
        gamma_map = np.min((gamma_map, gamma), axis=0)
    gamma_pass_rate = np.sum(np.bitwise_and(gamma_map <= 1, dose_mask)) / np.sum(dose_mask)
    return gamma_map, gamma_pass_rate


def gamma_calc_3dvh(reference, sample, dose_std='global', resolution=(1, 1, 1), dis_tol=3, dose_tol=0.02, threshold=0.1):
    # 模仿SunNuclear 3DVH Version3.0的算法
    search_region = 8
    max_dose = np.max(reference)
    dose_mask = np.bitwise_or(reference > max_dose * threshold, sample > max_dose * threshold)
    # reference = np.ma.array(reference, mask=~dose_mask)
    if dose_std == 'global':
        dose_tol = max_dose * dose_tol
    elif dose_std == 'local':
        dose_tol = reference * dose_tol
    else:
        raise ValueError('Not valid dose standard!')
    DTA = np.zeros_like(reference)
    diff = reference - sample
    # 计算距离谱
    search_range = np.array(np.ceil(search_region / np.array(resolution)), dtype=int)
    dis_range = []
    for dis in search_range:
        dis_range.append(list(range(-dis, dis + 1)))
    coords = np.array(np.meshgrid(*dis_range, indexing='ij'))
    dis_map = np.zeros_like(coords[0], dtype=float)
    for coord, res in zip(coords, resolution):
        dis_map += (coord * res) ** 2
    dis_map = np.sqrt(dis_map)
    dis_mask = dis_map <= search_region
    dis_map = dis_map[dis_mask]
    z_coord = coords[0][dis_mask]
    y_coord = coords[1][dis_mask]
    x_coord = coords[2][dis_mask]
    search_list = list(zip(dis_map, zip(z_coord, y_coord, x_coord)))
    search_list.sort(key=lambda x: x[0])

    z_length = sample.shape[0]
    y_length = sample.shape[1]
    x_length = sample.shape[2]
    # distance map
    for i in range(z_length):
        for j in range(y_length):
            for k in range(x_length):
                if not dose_mask[i, j, k]:
                    continue
                current_dis = 0
                current_dta = search_region
                s = sample[i, j, k]
                origin_diff = reference[i, j, k] - s
                for dis, (z, y, x) in search_list:
                    if (dis > current_dis) and (current_dta < search_region):
                        DTA[i, j, k] = current_dta
                        break
                    if 0 <= (i + z) < z_length and 0 <= (j + y) < y_length and 0 <= (k + x) < x_length:
                        current_diff = reference[i + z, j + y, k + x] - s
                        if current_diff == 0:
                            if dis < current_dta:
                                current_dta = dis
                                current_dis = dis
                        elif current_diff * origin_diff < 0:
                            if np.sign(z) == 0:
                                z_coord = z
                            else:
                                z_coord = z - np.sign(z) * current_diff / (
                                        reference[i + z, j + y, k + x] - reference[i + z - np.sign(z), j + y, k + x])
                            if np.sign(y) == 0:
                                y_coord = y
                            else:
                                y_coord = y - np.sign(y) * current_diff / (
                                        reference[i + z, j + y, k + x] - reference[i + z, j + y - np.sign(y), k + x])
                            if np.sign(x) == 0:
                                x_coord = x
                            else:
                                x_coord = x - np.sign(x) * current_diff / (
                                        reference[i + z, j + y, k + x] - reference[i + z, j + y, k + x - np.sign(x)])
                            distance = np.sqrt((z_coord * resolution[0]) ** 2 + (y_coord * resolution[1]) ** 2 + (x_coord * resolution[2]) ** 2)
                            if distance < current_dta:
                                current_dta = distance
                                current_dis = dis
                else:
                    DTA[i, j, k] = current_dta
    D = (diff / dose_tol) ** 2
    X = (DTA / dis_tol) ** 2
    gamma = np.sqrt(np.min([D, X], axis=0))
    gamma_pass_rate = np.sum(np.bitwise_and(gamma <= 1, dose_mask)) / np.sum(dose_mask)
    return gamma, gamma_pass_rate


def gamma_calc_partition(reference, sample, dose_std='global', partition=1 / 10, resolution=(1, 1, 1),
                         dis_tol=3, dose_tol=0.02, threshold=0.1):
    # check for valid arguments
    if reference.shape != sample.shape:
        raise Exception("ValueError: shape mismatch: reference and sample must have the same shape")
    if dis_tol <= 0 or int(dis_tol) != dis_tol:
        raise Exception("ValueError: dis_tol is an integer greater than zero")
    if dose_tol <= 0 or dose_tol >= 1:
        raise Exception("ValueError: dose_tol is a float number between 0 (exclusive) and 1 (exclusive)")
    ndim = len(resolution)
    resolution = np.array(resolution, dtype=int)

    dis_unit = dis_tol * partition
    voxel_unit = dis_unit / resolution
    unit_num = int(np.ceil(1 / partition)) - 1

    sample = cp.array(sample)
    reference = cp.array(reference)

    max_dose = cp.max(reference)
    dose_mask = np.bitwise_or(reference > max_dose * threshold, sample > max_dose * threshold)

    # distance map
    dis_range = [np.arange(-unit_num, unit_num + 1, 1, dtype=float) for _ in range(ndim)]
    order_map = cp.array(np.meshgrid(*dis_range, indexing='ij'))
    dis_map = cp.sqrt(cp.sum((order_map * dis_unit) ** 2, axis=0))
    dis_mask = dis_map >= dis_tol
    _sqDist = (dis_map / dis_tol) ** 2

    gamma = cp.zeros_like(reference)
    gamma[:] = cp.inf

    if ndim == 3:
        kz, kx, ky = cp.indices(dis_map.shape)
        for k, i, j in zip(kz.flatten(), kx.flatten(), ky.flatten()):
            idx = k, i, j
            # 生成新坐标的索引
            grid = cp.indices(sample.shape)
            coords = cp.array(grid) + ((cp.array([k, i, j]) - unit_num) * cp.array(voxel_unit)).reshape(-1, 1, 1, 1)
            move_sample = map_cp(sample, coords, order=1)
            # skip masked voxels
            if dis_mask[idx]:
                continue
            # gamma index
            if dose_std == 'global':
                dose_standard = max_dose
            elif dose_std == 'local':
                dose_standard = reference
            else:
                raise Exception("ValueError: dose_std is not global or local!")
            _sqDose = ((move_sample - reference) / dose_standard / dose_tol) ** 2
            _gamma = cp.sqrt(_sqDist[idx] + _sqDose)
            _ = gamma > _gamma
            gamma[_] = _gamma[_]
        gamma_pass_rate = cp.sum(cp.bitwise_and(gamma <= 1, dose_mask)) / cp.sum(dose_mask)
        return gamma, gamma_pass_rate
    else:
        nx, ny = reference.shape
        kx, ky = cp.indices(dis_map.shape)
        for i, j in zip(kx.flatten(), ky.flatten()):
            idx = i, j
            move_sample = sample[slice(i, i + nx), slice(j, j + ny)]
            # skip masked voxels
            if dis_mask[idx]:
                continue
            # gamma index
            if dose_std == 'global':
                dose_standard = max_dose
            elif dose_std == 'local':
                dose_standard = reference
            else:
                raise Exception("ValueError: dose_std is not global or local!")
            _sqDose = ((move_sample - reference) / dose_standard / dose_tol) ** 2
            _gamma = cp.sqrt(_sqDist[idx] + _sqDose)
            _ = gamma > _gamma
            gamma[_] = _gamma[_]
        gamma_pass_rate = cp.sum(cp.bitwise_and(gamma <= 1, dose_mask)) / cp.sum(dose_mask)
        return gamma, gamma_pass_rate


if __name__ == '__main__':
    import pydicom as dcm

    # TPS_dose = dcm.dcmread(r"C:\Users\chenl\Desktop\RT dose\T_Plan\TomoOptimizationDoseInfo_rtdose~3~93467e3e-5d5a-417a-965a-8b53c9926093.dcm")
    # PDP_dose = dcm.dcmread(r"C:\Users\chenl\Desktop\RT dose\T_Plan_Overlap12\TomoFinalDoseInfo_rtdose~1~14edc9f2-be75-403d-a333-65611c744a9e.dcm")
    TPS_dose = dcm.dcmread(r"H:\QAapplication\total_data\1000120261\C1\patient\RTDOSE_1.3.46.670589.13.625812.20230421034116.832353.dcm")
    PDP_dose = dcm.dcmread(r"H:\QAapplication\total_data\1000120261\C1\ACPDP\ACPDP_HighSensitivity.dcm")
    ref_img = TPS_dose.pixel_array * TPS_dose.DoseGridScaling
    samp_img = PDP_dose.pixel_array * PDP_dose.DoseGridScaling
    # samp_img = samp_img[:, ::2, ::2]
    resolution = (TPS_dose.GridFrameOffsetVector[1] - TPS_dose.GridFrameOffsetVector[0], TPS_dose.PixelSpacing[0], TPS_dose.PixelSpacing[1])
    # gamma32, gpr32 = gamma_calc_partition(ref_img, samp_img, dose_std='global', partition=1 / 20, resolution=resolution,
    #                                       dis_tol=2, dose_tol=0.03, threshold=0.1)
    start_time = time.time()
    gamma32, gpr32 = gamma_calc_gradient2(ref_img, samp_img, dose_std='global', resolution=resolution, dis_tol=2, dose_tol=0.03,
                                         threshold=0.1)
    end_time = time.time()
    run_time = end_time - start_time
    print(f"Gamma pass rate: {gpr32 * 100:.3f}%")
    print(f"Run time: {run_time:.3f}s")
