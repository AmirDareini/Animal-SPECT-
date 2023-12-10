import numpy as np
import matplotlib.pyplot as plt
from mlem_rec import mlem_reconstruction, osem_reconstruction
from forward_projection_distance_modeling_included import h_generator
from pydicom import dcmread
from utils import fwhm_finder, uniformity_calc
from scipy.ndimage import gaussian_filter

"""
A)  System matrix generation section
 
1) in 5 distance, H is generated 
    a) rbin - phi bin are extracted from the output data
    b) input ror must be in pixel unit -> ror(mm) / PIXEL_SIZE(mm)

"""

file_path_list = ['tomo_res-tomo-1_[256x256]_0.35 mm_dual_30.dcm',
                  'tomo_res-tomo-2_[256x256]_0.35 mm_dual_45.dcm',
                  'tomo_res-tomo-3_[256x256]_0.35 mm_dual_60.dcm',
                  'tomo_res-tomo-4_[256x256]_0.35 mm_dual_75.dcm',
                  'tomo_res-tomo-5_[256x256]_0.35 mm_dual_90.dcm']
dt_idx = 4
file_path = file_path_list[dt_idx]
dcm = dcmread(file_path)
dt = dcm.pixel_array
print('machine output data shape is: \n', dt.shape)
nslice = 25
prj = dt[32:, nslice, :]
phi_bin = prj.shape[0]
rbin = prj.shape[1]
print('Number of projections: %i, Projection size %i \n' % (phi_bin, rbin))
print('Slice %i is selected\n' % nslice)
PIXEL_SIZE = 1.4  # mm


init_ang = 0
end_ang = 360
num_prj = phi_bin
prj_angles = np.linspace(init_ang, end_ang, num_prj)

input_img = np.zeros((rbin, rbin))

# Be careful to choose the correct ror
ror = 90   # mm
pixel_ror = ror / PIXEL_SIZE

print(f'selected ror: {ror} in mm , in pixel: {pixel_ror} \n')
# H = h_generator(input_img, prj_angles, pixel_ror)
# np.save('modified_blurred_H_ror%i_mm' % ror, H)

"""
B) Reconstruction section

1) in 5 distance, data are reconstructed with both osem-mlem 
    a) each data(%ror) is reconstructed with its specific H%ror 
    b) un recovered image -> mlem(16), osem(4,4)
    c) recovered image -> mlem(40), osem(10,4)
    d) FWHM for both osem, mlem and each for both RR & No RR (4 totally)

"""

#
# H = np.load("modified_blurred_H_ror75_mm.npy")
# raw_H = np.load('unblurred_H_rbin64_Phi_bin32.npy')
# sino = prj.T
# rec_img_em = mlem_reconstruction(input_img, sino, H, num_iter)
# rec_img_os = osem_reconstruction(input_img, sino, H, num_iter // num_subset, num_subset)
# plt.subplot(121)
# plt.imshow(rec_img_em, cmap='jet')
# plt.subplot(122)
# plt.imshow(rec_img_os, cmap='jet')
# fwhm_em = fwhm_finder(np.sum(rec_img_em, axis=1))
# fwhm_os = fwhm_finder(np.sum(rec_img_os, axis=1))
# print(f'mlem fwhm in mm: {fwhm_em * 1.4}')
# print(f'osem fwhm in mm: {fwhm_os * 1.4}')
# plt.show()
#

"""
Uniformity test

"""
H = np.load("modified_blurred_H_ror30_mm.npy")
raw_H = np.load('unblurred_H_rbin64_Phi_bin32.npy')
file_path = "uniformity_uniformity-3_[256x256]_0.35 mm_dual_30.dcm"
dcm = dcmread(file_path)
dt = dcm.pixel_array
print('machine output data shape is: \n', dt.shape)
nslice = 25
prj = dt[32:, nslice, :]
phi_bin = prj.shape[0]
rbin = prj.shape[1]
print('Number of projections: %i, Projection size %i \n' % (phi_bin, rbin))
print('Slice %i is selected\n' % nslice)
PIXEL_SIZE = 1.4  # mm
sino = prj.T
num_iter = 16
num_subset = 4
raw_img_em = mlem_reconstruction(input_img, sino, raw_H, num_iter)
raw_img_os = osem_reconstruction(input_img, sino, raw_H, num_iter // num_subset, num_subset)
rec_img_em = mlem_reconstruction(input_img, sino, H, num_iter)
rec_img_os = osem_reconstruction(input_img, sino, H, num_iter // num_subset, num_subset)

plt.subplot(221)
plt.imshow(raw_img_em, cmap='jet')
plt.title(f'mlem NO RR, niter: {16}')

plt.subplot(222)
plt.imshow(rec_img_em, cmap='jet')
plt.title(f'mlem RR  niter: {num_iter}')

plt.subplot(223)
plt.imshow(raw_img_os, cmap='jet')
plt.title(f'osem NO RR niter:{16 // num_subset}, nsubset: {num_subset}')


plt.subplot(224)
plt.imshow(rec_img_os, cmap='jet')
plt.title(f'osem RR: niter: {num_iter // num_subset}, nsubset: {num_subset}')


plt.show()
"""
Visualization of results 
 
"""

# # Sample data loading
#
# raw_H = np.load('unblurred_H_rbin64_Phi_bin32.npy')
# img = np.ones((rbin, rbin))
# dists = [30, 45, 60, 75, 90]
# num_iter = 40
# num_subset = 4
#
# # Assuming you have some file_path_list defined
# file_path_list = ['tomo_res-tomo-1_[256x256]_0.35 mm_dual_30.dcm',
#                   'tomo_res-tomo-2_[256x256]_0.35 mm_dual_45.dcm',
#                   'tomo_res-tomo-3_[256x256]_0.35 mm_dual_60.dcm',
#                   'tomo_res-tomo-4_[256x256]_0.35 mm_dual_75.dcm',
#                   'tomo_res-tomo-5_[256x256]_0.35 mm_dual_90.dcm']
#
# # Create a figure and an array of axes
# fig, axs = plt.subplots(len(dists), 3, figsize=(15, 5 * len(dists)))
#
# for i, ax_row in enumerate(axs):
#     for j, ax in enumerate(ax_row):
#         H = np.load("modified_blurred_H_ror%i_mm.npy" % dists[i])
#         dcm = dcmread(file_path_list[i])
#         dt = dcm.pixel_array
#         nslice = 25
#         prj = dt[32:, nslice, :]
#         phi_bin = prj.shape[0]
#         rbin = prj.shape[1]
#         sino = prj.T
#         raw_img = mlem_reconstruction(img, sino, raw_H, 40)
#         em_img_rr = mlem_reconstruction(img, sino, H, num_iter)
#         os_img_rr = osem_reconstruction(img, sino, H, num_iter // num_subset, num_subset)
#         em_fwhm = fwhm_finder(np.sum(em_img_rr, axis=1))
#         os_fwhm = fwhm_finder(np.sum(os_img_rr, axis=1))
#
#
#         if j == 0:
#             ax.imshow(raw_img, cmap='jet')
#             ax.set_title(' No RR image ')
#         elif j == 1:
#             ax.imshow(em_img_rr, cmap='jet')
#             ax.set_title(f' mlem RR {num_iter} ')
#
#         elif j == 2:
#             ax.imshow(os_img_rr, cmap='jet')
#             ax.set_title(f'osem RR niter: {num_iter // num_subset}, nsub: {num_subset}')
#
# # Adjust layout
# plt.tight_layout()
# plt.show()
