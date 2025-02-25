import numpy as np
import time


def gamma_calc_gradient(reference, sample, dose_std='global', resolution=(1, 1, 1), dis_tol=2, dose_tol=0.03, threshold=0.1):
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
    gamma_map = np.ones_like(reference) * np.inf
    for dis, coord_bias in search_list:
        ref = reference
        samp = sample
        gd = grad
        pad_range = []
        for i, bias in enumerate(coord_bias):
            ref = np.take(ref, range(max(0, -bias), min(reference.shape[i] - bias, reference.shape[i])), i)
            samp = np.take(samp, range(max(0, bias), min(reference.shape[i] + bias, reference.shape[i])), i)
            gd = np.take(gd, range(max(0, bias), min(reference.shape[i] + bias, reference.shape[i])), i + 1)
            pad_range.append((max(0, -bias), max(0, bias)))
        diff = samp - ref
        X0 = np.array(coord_bias) * np.array(resolution)
        X0g = sum([x0 * g for x0, g in zip(X0, gd)])
        d2g2 = np.sum(gd ** 2, axis=0) * (dis_tol / dose_tol) ** 2
        X = [(vec_g - x0) for vec_g, x0 in zip((X0g - diff) * (dis_tol / dose_tol) ** 2 / (1 + d2g2) * np.array(gd), X0)]
        X_norm = np.array([x / res for x, res in zip(X, resolution)])
        X_norm = np.where(np.abs(X_norm) <= 1 / 2, X_norm, np.sign(X_norm) * 1 / 2)
        X = np.array([x * res + x0 for x, res, x0 in zip(X_norm, resolution, X0)])
        X = np.array([x + x0 for x, x0 in zip(X, X0)])
        Xg = sum([x * g for x, g in zip(X, gd)])
        gamma = np.sqrt((diff + Xg - X0g) ** 2 / dose_tol ** 2 + np.sum(np.array(X) ** 2, axis=0) / dis_tol ** 2)
        gamma = np.pad(gamma, pad_width=pad_range, mode='constant', constant_values=np.inf)
        gamma_map = np.min((gamma_map, gamma), axis=0)
    gamma_pass_rate = np.sum(np.bitwise_and(gamma_map <= 1, dose_mask)) / np.sum(dose_mask)
    return gamma_map, gamma_pass_rate


def gamma_calc_3dvh(reference, sample, dose_std='global', resolution=(1, 1, 1), dis_tol=3, dose_tol=0.02, threshold=0.1):
    # 模仿SunNuclear 3DVH Version3.0的算法
    search_region = 8
    max_dose = np.max(reference)
    dose_mask = np.bitwise_or(reference > max_dose * threshold, sample > max_dose * threshold)
    if dose_std == 'global':
        dose_tol = max_dose * dose_tol
    elif dose_std == 'local':
        dose_tol = reference * dose_tol
    else:
        raise ValueError('Not valid dose standard!')
    DTA = np.zeros_like(reference)
    diff = reference - sample
    # 距离谱
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


if __name__ == '__main__':
    import pydicom as dcm

    ref_dose = dcm.dcmread(r"参考剂量dicom文件路径")
    samp_dose = dcm.dcmread(r"采样剂量dicom文件路径")
    ref_img = ref_dose.pixel_array * ref_dose.DoseGridScaling
    samp_img = samp_dose.pixel_array * samp_dose.DoseGridScaling
    resolution = (ref_dose.GridFrameOffsetVector[1] - ref_dose.GridFrameOffsetVector[0], ref_dose.PixelSpacing[0], ref_dose.PixelSpacing[1])
    start_time = time.time()
    gamma32, gpr32 = gamma_calc_gradient(ref_img, samp_img, dose_std='global', resolution=resolution, dis_tol=2, dose_tol=0.03,
                                         threshold=0.1)
    end_time = time.time()
    run_time = end_time - start_time
    print(f"Gamma pass rate: {gpr32 * 100:.3f}%")
    print(f"Run time: {run_time:.3f}s")
