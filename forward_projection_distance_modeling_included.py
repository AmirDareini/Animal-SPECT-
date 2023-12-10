"""
Blurring System matrix generation
Author - Developer : [Amir Dareyni]
Data of creation : 11 / 12 / 2023
Organization: [Parto Negar Persia]

"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import numpy as np
from scipy.stats import norm
from utils import get_line_coefficients, point_line_distance
from scipy.optimize import curve_fit

PIXEL_SIZE = 1.4  # mm

def h_generator(image, azi_angles, radius_rotation):
    """
    Generated system matrix for specific rbins and phi bins
    :param image: 2D array
    :param azi_angles: np. array of projection angles
    :return: sparse matrix called system matrix
    """

    rbin = image.shape[0]
    phbin = len(azi_angles)


    """ We define new coordinate system in the center point of the image"""
    coord_center = (image.shape[0] / 2, image.shape[1] / 2)
    source_to_detector_distance = radius_rotation

    def forward_projection(projection_angle, pixel_position):
        """
        Measures forward projection of a single pixel in a specific projection angle
        :param projection_angle: float ->  current angle
        :param pixel_position: tuple -> current pixel index
        :return:
        """
        rad_angle = np.deg2rad(projection_angle)

        """
        We need to swap the x_idx, y_idx here to have true positioning
        """
        x_idx, y_idx = pixel_position
        x_idx, y_idx = y_idx, x_idx

        #  Detector center point, unit vector and detector array in a specific projection angle is calculated.
        detector_center = (
            (source_to_detector_distance + coord_center[0]) * np.cos(rad_angle),
            (source_to_detector_distance + coord_center[1]) * np.sin(rad_angle)
        )
        detector_unit_vector = np.sin(rad_angle), -np.cos(rad_angle)

        off_set_central_to_start_point = (
            - detector_unit_vector[0] * (rbin / 2),
            - detector_unit_vector[1] * (rbin / 2)
        )
        line_detector_list = [
            (
                (detector_center[0] + off_set_central_to_start_point[0]) + (i + 0.5) * detector_unit_vector[0],
                (detector_center[1] + off_set_central_to_start_point[1]) + (i + 0.5) * detector_unit_vector[1]
            )
            for i in range(rbin)
        ]
        """
        1) Desired pixel position should convert to our convention coordinate [image.shape[0] / 2, image.shape[1] / 2 ]
        2) vec -> a vector from coordinate center toward pixel converted position
        3) There are three angles here: 
                a) alpha: angle of vector with coordinate system
                b) rad_angle: projection angle in radian 
                c) beta: difference between projection angle and alpha.
                    -> used for calculation the perpendicular distance from the detector center bin
        4) perpen_dist: perpendicular distance from current position to detector center bin
        5) cdf_values: Cumulative Density function for each detector bin -> should convert to pdfs.
        6) pdf_values: difference between two consecutive cdf value + the first element of cdf value 
        """
        current_position = x_idx - coord_center[0] + 0.5, coord_center[1] - y_idx - 0.5

        vec = np.array(current_position) - np.array((0, 0))
        vec_length = np.linalg.norm(np.array(vec))
        alpha = np.arctan2(current_position[1], current_position[0])
        beta = alpha - rad_angle
        perpen_dist = vec_length * np.sin(beta)
        detector_arr_idxs = np.linspace(-(rbin - 1) / 2, (rbin - 1) / 2, rbin)
        """ Calcualate the sigma using the experimental data from Animal SPECT """
        dist_dat = np.array([30, 45, 60, 75, 90]) / PIXEL_SIZE
        fwhm_data = np.array([3.23, 4.4, 5.37, 6.21, 6.59]) / PIXEL_SIZE
        line_coeff = get_line_coefficients(line_detector_list[2], line_detector_list[1])
        dist = point_line_distance(line_coeff, current_position)  # Distance from current position to line detector list
        # Fitting process to calculate sigma from dist
        def trend_func(dist, a, b):
            return (a * dist) + b
        params, _ = curve_fit(trend_func, dist_dat, fwhm_data)
        sigma = (trend_func(dist, *params)) / 2.355
        # calculate each projection values called 'pdf_values' using 'Cumulative density functions'
        pdf_values = [norm.pdf(x, loc=perpen_dist, scale=sigma) for x in detector_arr_idxs]
        # pdf_values = [cdf_values[i+1] - cdf_values[i] for i in range(rbin)]
        return pdf_values

    """
    Begin forward projection for each pixel
        1) pixel is selected
        2) forward projection is performed in all projection angles -> sinogram is generated for a single pixel
        3) the process is repeated for all the pixels in the image -> a set of sinogram is generated 
        4) All the sonograms are stacked to form the sparse matrices of H (System matrix)
    """
    # Batch operation sinogram generation
    # sino_list = []
    # for i in range(rbin):
    #     print('current: %i , remain: %i' % (i, rbin - i))
    #     for j in range(rbin):
    #         angles = np.array(azi_angles)
    #         sinogram = forward_projection(angles, (i, j))
    #         sino_list.append(sinogram)
    #
    # H = np.vstack(sino_list).reshape((rbin * rbin, phbin * rbin))
    # return H

    # Non batch operation sinogram generation
    sino_list = []
    for i in range(rbin):
        print('current: %i , remain: %i' % (i, rbin - i))
        for j in range(rbin):
            sinogram = np.zeros((phbin, rbin))
            for idx, angle in enumerate(azi_angles):
                sinogram[idx, :] = (forward_projection(angle, (i, j)))
            sino_list.append(sinogram.T)
    H = np.vstack(sino_list).reshape(rbin*rbin, rbin*phbin)
    return H


# if __name__ == '__main__':
#     _ = h_generator(np.ones((64, 64)), np.linspace(0, 360, 32), 50)