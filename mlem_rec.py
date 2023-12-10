import numpy as np
import matplotlib.pyplot as plt


def mlem_reconstruction (raw_image, sinogram, system_matrix, num_iter):
    recon_img = np.ones_like(raw_image)
    init_sino = np.ones_like(sinogram)
    sense_image = np.dot(init_sino.flatten(), system_matrix.T)
    for iter in range(num_iter):
        fp = np.dot(recon_img.flatten(), system_matrix)
        error_term = (sinogram.flatten() + 0.00001) / (fp + 0.00001)
        corr_fact = np.dot(error_term, system_matrix.T) / sense_image.flatten()
        recon_img = recon_img.flatten() * corr_fact
    return recon_img.reshape(raw_image.shape)


def osem_reconstruction(raw_image, sinogram, system_matrix, num_iter, num_subset):
    rbin, nview = sinogram.shape
    rec_img = np.ones_like(raw_image)
    init_sino = np.ones_like(sinogram)
    wgts = []
    for sub in range(num_subset):
        views = range(sub, nview, num_subset)
        partial_sino = init_sino[:, views]
        wgt = np.dot(partial_sino.flatten(), system_matrix[:, ::num_subset].T)
        wgts.append(wgt)
    for iter in range(num_iter):
        order = np.random.permutation(range(num_subset))
        for sub in order:
            views = range(sub, nview, num_subset)
            fp = np.dot(rec_img.flatten(), system_matrix[:, ::num_subset])
            ratio = sinogram[:, views].flatten() / (fp + 0.0000001)
            bp = np.dot(ratio, system_matrix[:, ::num_subset].T)
            bp = bp.reshape(raw_image.shape)
            rec_img *= (bp / (wgts[sub].reshape(raw_image.shape) + 0.0000001))
    return rec_img

