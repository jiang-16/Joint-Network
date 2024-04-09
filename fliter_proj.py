import numpy as np
import scipy.fftpack as sp_fft


def filtered(proj, geo, angles):
    filt_len = max(64, 2 ** int(np.ceil(np.log2(2 * proj.shape[0]))))
    ramp_kernel = ramp_flat(filt_len)

    d = 1  # cut off (0~1)
    filt = Filter(ramp_kernel, filt_len, d)
    filt = filt.T

    for ii in range(angles.shape[0]):
        fproj = np.zeros((filt_len, 1), dtype=np.float32)
        fproj[int(filt_len / 2 - geo['nDetector'][0] / 2):int(filt_len / 2 + geo['nDetector'][0] / 2), :] = proj[:, ii][
                                                                                                            :,
                                                                                                            np.newaxis]

        fproj = sp_fft.fft(fproj, axis=0)
        fproj = fproj * filt[:, np.newaxis]

        fproj = np.real(sp_fft.ifft(fproj, axis=0))

        proj[:, ii] = fproj[int(fproj.shape[0] / 2 - geo['nDetector'][0] / 2):int(
            fproj.shape[0] / 2 + geo['nDetector'][0] / 2), :].squeeze()

    return proj


def ramp_flat(n):
    nn = np.arange(-n / 2, n / 2).astype(np.float32)
    h = np.zeros_like(nn)
    h[int(n / 2)] = 1 / 4
    odd = nn % 2 == 1
    h[odd] = -1 / (np.pi * nn[odd]) ** 2
    return h


def Filter(kernel, order, d):
    f_kernel = np.abs(sp_fft.fft(kernel)) * 2
    filt = f_kernel[:int(order / 2 + 1)]
    w = 2 * np.pi * np.arange(filt.shape[0]) / order
    filt[w > np.pi * d] = 0
    filt = np.concatenate([filt, filt[-2:0:-1]])
    return filt