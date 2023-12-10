import math
import numpy as np
import pylab as p
from skimage.data import shepp_logan_phantom
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import cv2
# from lmfit import Model
import pydicom as dcm
import os
import statistics as stats


def get_line_coefficients(point1, point2):
    # extract the coordinate of the points
    x1, y1 = point1
    x2, y2 = point2

    # calculate the coefficients
    A = y1 - y2
    B = x2 - x1
    C = x1*y2 - x2*y1
    return A, B, C


def point_line_distance(line_coeff, point):
    a, b, c = line_coeff
    x1, y1 = point
    dist = abs(a * x1 + b*y1 + c) / np.sqrt(a ** 2 + b ** 2)
    return dist


FWHM_CONVERSION_FACTOR = np.sqrt(8 * np.log(2))


def scale(array, bit=8, int_out=True, maximum: int = None):
    """
    This function scales 2D variables and mainly designed for 8-bit image viewing.

    Args:
        array (numpy.ndarray): Array to scale.
        bit (int, optional): The bit depth of output integer data. Defaults to 8.
        int_out (bool, optional): rounds output to nearest integer and change data type to np.uint16. Defaults to True.
        maximum (int, optional): for using this function in another format, use for scale a variable, set maximum to a float or integer. this will DISCARD bit input. Defaults to None.

    Returns:
        _scaled_array (float/int): Scaled variable as main data type or np.uint16.
    """

    if maximum is None:
        _calibration_factor = 2 ** bit - 1
    else:  # maximum is an integer
        _calibration_factor = maximum
    _max = np.nanmax(array)
    _min = np.nanmin(array)
    if _max == _min:
        return array
    else:
        _scaled_array = _calibration_factor * ((array - _min) / (_max - _min))
        if int_out is True:
            return _scaled_array.round()
        else:  # int_out is False:
            return _scaled_array


def shift(im, neg_reflect=False, neg_cut=False, out_int=False):
    """
    This function shifts 2D variables to be non-zero and mainly designed for 8-bit image viewing.
    :param im: input 2D ndarray.
    :param neg_reflect: will change minos to positive values.
    :param neg_cut: set minos to zero.
    :param out_int: will round data and change data type format to np.uint16.
    :return: returns shifted image.
    """
    _min = np.min(im)
    if neg_reflect is False and neg_cut is False:  # just shift
        _shifted_im = im - _min
        if out_int is False:
            return _shifted_im
        else:
            return (_shifted_im.round()).astype(int)
    elif neg_reflect is True and neg_cut is False:  # neg_reflect is True
        _shifted_im = np.zeros_like(im)
        _shifted_im[im < 0] = im[im < 0] * (-1)
        _shifted_im[im > 0] = im[im > 0]
        if out_int is False:
            return _shifted_im
        else:
            return (_shifted_im.round()).astype(int)
    elif neg_cut is True and neg_reflect is False:
        _shifted_im = im
        _shifted_im[_shifted_im < 0] = 0
        if out_int is False:
            return _shifted_im
        else:
            return (_shifted_im.round()).astype(int)

    else:  # both are True:
        raise ValueError("both neg_cut and neg_reflect could not be True!")


def plotter(
        img=None,
        line_x_y_tup: tuple = None,
        make_new_fig: bool = True,
        window_title: str = None,
        plot_title: str = None,
        img_cmap: str = 'jet',
        img_show_colorbar: bool = True,
        line_color_and_style: str = None,
        x_lim=None,
        y_lim=None,
        blocking_process=True,
):
    """
    Plot an image or line. This is a wrapper around matplotlib's : func : ` pyplot ` that allows to specify title, colormap, color, and style of the plot.

    Args:
        img (_type_, optional): The image to plot. If it is None the line_x_y_tup must be None and is used to plot the image. Defaults to None.
        line_x_y_tup (tuple, optional): A tuple of x and y coordinates to plot. Defaults to None.
        make_new_fig (bool, optional): Whether to create a new Figure or not. Defaults to True.
        window_title (str, optional): Set window title. Defaults to None.
        plot_title (str, optional): Defines plot title. Defaults to None.
        img_cmap (str, optional): Colormap of grayscale Image. Defaults to 'jet'.
        img_show_colorbar (bool, optional): Colormap tag for showing Colormap. Defaults to True.
        line_color_and_style (str, optional): Color and Style of Line. Defaults to None.
        x_lim (tuple, optional)
        y_lim (tuple, optional)

    Raises:
        Exception: If both 'img' & 'line_x_y_tuple' where 'None'.
    """

    # Make a new figure if necessary.
    if make_new_fig:
        fig = plt.figure(window_title)

    # Specify the image colormap.
    if img is not None:
        plt.imshow(img, cmap=img_cmap)
        # Set the title of the plot.
        if plot_title is not None:
            plt.title(plot_title)
        # For images, if show_colorbar is true then the colorbar is shown.
        if img_show_colorbar:
            plt.colorbar()
    elif line_x_y_tup is not None:
        (x, y) = line_x_y_tup
        # For Lines, modifies the line color and style of the plot.
        if line_color_and_style is not None:
            plt.plot(x, y, line_color_and_style)
        else:
            plt.plot(x, y)
    else:
        raise Exception(
            f"The 'img' and 'line_x_y_tup' could not be None together!"
        )

    # Make the figure layout if necessary.
    if make_new_fig:
        fig.tight_layout()
    if x_lim is not None:
        plt.xlim(x_lim[0], x_lim[1])
    if y_lim is not None:
        plt.ylim(y_lim[0], y_lim[1])
    if blocking_process:
        plt.show()
    else:
        plt.show(block=False)


# def fwhm_finder_old(line, convert_to_sigma=False):
#     # line = np.array(array[row, :])
#     _len = len(line)
#     x_dt = np.linspace(0, _len - 1, _len)
#     g_model = Model(_gaussian)
#     result = g_model.fit(line, x=x_dt, amp=99, cen=_len // 2, wid=3)
#     # print(f'fwhm_finder>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n'
#     #       f'line:{line}\n')
#     sigma = result.best_values['wid']
#     if convert_to_sigma:
#         # plotter(line_x_y_tup=(x_dt, line), plot_title=f'sigma @ {row}: {sigma}')
#         print(f'sigma:{sigma}')
#         return sigma
#     else:  # FWHM
#         fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
#         plotter(line_x_y_tup=(x_dt, line), plot_title=f'fwhm: {fwhm}')
#         plotter(line_x_y_tup=(x_dt, _gaussian(x_dt, result.best_values['amp'], result.best_values['cen'], sigma)),
#                 line_color_and_style='.r')
#         # print(f"amp:{result.best_values['amp']}")
#         # print(f"cen:{result.best_values['cen']}")
#         print(f'fwhm @ fwhm_finder_old:{fwhm}')
#         return fwhm


def fwhm_finder(
        line: np.array,
        convert_to_sigma=False,
        pixel_size_mm: float = 1.,
        return_fwtm=False, ):
    """
    this function will return the FWHM (and FWTM) of an array According to  NEMA.
    Args:
        line (numpy.array): input numpy array.
        convert_to_sigma (bool, optional): calculates Sigma of Gaussian curve instead of FWHM. Only works when return_fwhm is False.
        pixel_size_mm (float, optional): its needed for calculation of sigma.
        return_fwtm (bool, optional): for returning FWTM after FWHM.

    Returns:
        FWHM: fwhm of line
        FWTM(Optional): full width of TENTH of Maximum
    """
    _len = len(line)
    x_dt = np.linspace(0, _len - 1, _len)
    nominated_indexes = _find_max_loc(line)
    coefficients = np.polyfit(nominated_indexes, line[nominated_indexes], deg=2)
    peak_location = np.float32(-coefficients[1] / (2 * coefficients[0]))  # -b/2a in ax2+bx+c

    interpolated_max_value = np.float32(np.polyval(coefficients, peak_location))
    half_interpolated_max_value = np.float32(interpolated_max_value / 2)

    diff = half_interpolated_max_value - np.float32(line)
    changing_sign_indexes = _find_sign_change_indices(diff)
    # print('changing_sign_indexes', changing_sign_indexes)
    index_before_after_peak = _find_sign_change_indices(changing_sign_indexes - peak_location)
    # print('index_before_after_peak', index_before_after_peak)
    # print(index_before_after_peak[0] - 1, index_before_after_peak[0])
    where_changing_before = changing_sign_indexes[index_before_after_peak[0] - 1]
    where_changing_after = changing_sign_indexes[index_before_after_peak[0]]
    x1_before, x2_before = where_changing_before - 1, where_changing_before
    x1_after, x2_after = where_changing_after - 1, where_changing_after

    y1_before, y2_before = line[x1_before], line[x2_before]
    y1_after, y2_after = line[x1_after], line[x2_after]
    x_before: np.float32 = linear_interpolation(half_interpolated_max_value, np.float32(x1_before),
                                                np.float32(x2_before), np.float32(y1_before), np.float32(y2_before))
    x_after: np.float32 = linear_interpolation(half_interpolated_max_value, np.float32(x1_after), np.float32(x2_after),
                                               np.float32(y1_after), np.float32(y2_after))
    _fwhm = (x_after - x_before) * np.float32(pixel_size_mm)

    if return_fwtm:
        tenth_interpolated_max_value = np.float32(interpolated_max_value / 10)
        diff_tenth = tenth_interpolated_max_value - np.float32(line)
        changing_sign_indexes_t = _find_sign_change_indices(diff_tenth)
        index_before_after_peak_t = _find_sign_change_indices(changing_sign_indexes_t - peak_location)
        where_changing_before = changing_sign_indexes_t[index_before_after_peak_t[0] - 1]
        where_changing_after = changing_sign_indexes_t[index_before_after_peak_t[0]]
        x1_before_t, x2_before_t = where_changing_before - 1, where_changing_before
        x1_after_t, x2_after_t = where_changing_after - 1, where_changing_after

        y1_before, y2_before = line[x1_before_t], line[x2_before_t]
        y1_after, y2_after = line[x1_after_t], line[x2_after_t]
        x_before_t = linear_interpolation(tenth_interpolated_max_value, np.float32(x1_before_t),
                                          np.float32(x2_before_t), np.float32(y1_before), np.float32(y2_before))
        x_after_t = linear_interpolation(tenth_interpolated_max_value, np.float32(x1_after_t), np.float32(x2_after_t),
                                         np.float32(y1_after), np.float32(y2_after))
        _fwtm = (x_after_t - x_before_t) * np.float32(pixel_size_mm)
        return _fwhm, _fwtm

    if convert_to_sigma:
        sigma = _fwhm / FWHM_CONVERSION_FACTOR
        # plotter(line_x_y_tup=(x_dt, line), plot_title=f'sigma @ {row}: {sigma}')
        print(f'sigma:{sigma}')
        return sigma
    else:  # FWHM
        print (_fwhm)
        return _fwhm


def _find_max_loc(x_array):
    """
    Finds maximum argument of a line then return before & after indices numbers.
    """
    # _max = np.max(x_array)
    # max_idx = np.where(x_array == _max)[0][0]

    max_idx = np.argmax(x_array)
    return np.array(((max_idx - 1), max_idx, (max_idx + 1)))


def _find_sign_change_indices(arr):
    indices = []
    for i in range(1, len(arr)):
        if arr[i] * arr[i - 1] < 0:
            indices.append(i)
    return indices


def linear_interpolation(y: np.float32, x1: np.float32, x2: np.float32, y1: np.float32, y2: np.float32):
    """
    solves this Equation for X (x2 > X > x1):
    [(X - x1) / (x2 - x1)] = [(y - y1) / (y2 - y1)]
    X = linear_interpolation(y, x1, x2, y1, y2)
    """
    # print(f"y={y}, x1={x1}, x2={x2}, y1={y1}, y2={y2}")
    # print(f"{round(x1, 2)} + ((({round(x2, 2)} - {round(x1, 2)}) / ({round(y2, 2)} - {round(y1, 2)}))"
    #       f" * ({round(y, 2)} - {round(y1, 2)}))")
    answer: np.float32 = x1 + (((x2 - x1) / (y2 - y1)) * (y - y1))
    # print(f"answer = {answer}")
    return answer


def _gaussian(x, amp, cen, wid):
    return (amp / (np.sqrt(2 * np.pi) * wid)) * np.exp(-(x - cen) ** 2 / (2 * wid ** 2))


def split_dcm(dcm_pth: str, index_ls):
    ds = dcm.dcmread(dcm_pth)
    dt = ds.pixel_array
    cp = []
    dicom_list = []

    try:
        total_copy = len(index_ls) + 1

    except Exception as e:
        print('Exception: ', e)
        print('Setting cp_len as 2.')
        if isinstance(index_ls, int):
            total_copy = 2
            index_ls = [index_ls]
        else:
            raise ValueError('INT or LIST for index_ls! \n( ´･･)ﾉ(._.`)')

    start = 0
    num = 1
    for c in index_ls:
        tmp = ds.copy()
        new_row = c - start
        print('new_row:', new_row)
        tmp.Rows = new_row

        tmp_dt = dt[:, start:c, :]
        print('tmp_dt.shape:', tmp_dt.shape)
        tmp.PixelData = tmp_dt.tobytes()

        prefix = str(num) + 'of' + str(total_copy) + '_'
        print('prefix:', prefix)
        pth = os.path.join(
            os.path.split(dcm_pth)[0], prefix + os.path.split(dcm_pth)[1]
        )
        tmp.save_as(os.path.abspath(pth))
        dicom_list.append(pth)
        cp.append(tmp)
        start = c
        num += 1

    lst_tmp = ds.copy()
    new_row = dt.shape[1] - start
    lst_tmp.Rows = new_row
    print('last start:', start)

    lst_tmp_dt = dt[:, start:, :]
    print('last tmp_dt.shape:', lst_tmp_dt.shape)

    lst_tmp.PixelData = lst_tmp_dt.tobytes()

    prefix = str(num) + 'of' + str(total_copy) + '_'
    print('prefix:', prefix)
    pth = os.path.join(
        os.path.split(dcm_pth)[0], prefix + os.path.split(dcm_pth)[1]
    )

    lst_tmp.save_as(pth)
    dicom_list.append(pth)

    cp.append(lst_tmp)

    return cp


def round_to_n(x, n=1, rounding_number_out=False):
    """
    round a number (an array) into its meaningful digits

    Args:
        x: float number.
        n: number of meaningful numbers. (default is 1.)
        rounding_number_out: if is True, rounding_number will be returned.
    Returns:
        rounded number (array)
        rounding_number (int): number for use in round() or np.round() module must be used for same result.
    """
    rounding_number = n - 1 - int(np.floor(np.log10(abs(x))))
    print(rounding_number)
    if rounding_number_out:
        if rounding_number > 0:
            return np.round(x, rounding_number), rounding_number
        else:
            return np.round(x, rounding_number).astype(int), rounding_number
    else:
        if rounding_number > 0:
            return np.round(x, rounding_number)
        else:
            return np.round(x, rounding_number).astype(int)


def show_lines_on_mpl(_2d_img):
    fig, ax = plt.subplots()

    def on_mouse_move(event):
        # Get the x and y coordinates of the mouse pointer
        x = event.xdata
        y = event.ydata

        if x is not None and y is not None:
            # Clear the previous lines
            ax.lines.clear()

            # Draw the vertical line
            ax.axvline(x=x, color='red')

            # Draw the horizontal line
            ax.axhline(y=y, color='red')

            # Update the plot
            plt.draw()

    # Connect the event handler function to the 'motion_notify_event' event
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

    ax.imshow(_2d_img, cmap='jet')
    # img_img = ax.imshow(_2d_img, cmap='jet')
    # fig.colorbar(img_img)
    plt.show()


# def shoe_lines_3d(_3d_img)


def find_max_row_col(img_2d):
    # find Arg Max in an numpy array
    (row_, col_) = np.unravel_index(
        indices=np.argmax(img_2d),
        shape=img_2d.shape,
    )
    # col_, row_ = np.where(img_2d.max() == img_2d)
    return row_, col_


def fwhm_3d_finder(
        array_3d,
        upper_limit: int,
        lower_limit: int,
        xcol_yrow_ul: tuple,
        xcol_yrow_ll: tuple,
        up_sample_is_needed: bool = False,
        up_sample_scale_factor=0,
        us_img_out: bool = True,
        pixel_size_mm: float = 1.0, ):
    """
    Finds fwhm in row and column of each slices of a 3-D Image. returns
    """
    if up_sample_is_needed and up_sample_scale_factor != 0:
        img = np.zeros(
            (
                array_3d.shape[0],
                up_sample_scale_factor * array_3d.shape[1],
                up_sample_scale_factor * array_3d.shape[2],
            )
        )
        for i in range(img.shape[0]):
            img[i] = upscale_bicubic(array_3d[i], scale_factor=up_sample_scale_factor)
        xcol_yrow_ul = (
            xcol_yrow_ul[0] * up_sample_scale_factor,
            xcol_yrow_ul[1] * up_sample_scale_factor)
        xcol_yrow_ll = (
            xcol_yrow_ll[0] * up_sample_scale_factor,
            xcol_yrow_ll[1] * up_sample_scale_factor)
    else:
        img = array_3d
        if up_sample_scale_factor == 0:
            raise Exception("please set the 'up_sample_scale_factor'!")

    # accepted_range = range(img.shape[0])
    accepted_range = range(upper_limit, lower_limit + 1)
    fwhm_row = []
    fwhm_col = []

    for slc in accepted_range:
        this_slice = img[slc]

        # ##### old linear finding max ###### #
        # col = round(linear_interpolation(slc, xcol_yrow_ul[0], xcol_yrow_ll[0], upper_limit, lower_limit))
        # row = round(linear_interpolation(slc, xcol_yrow_ul[1], xcol_yrow_ll[1], upper_limit, lower_limit))

        # # #### one row/col for fining fwhm #### #
        # row, col = find_max_row_col(img_2d=this_slice)

        fwhm_row.append(fwhm_finder(line=this_slice.sum(axis=0), pixel_size_mm=pixel_size_mm))  # row-wise
        fwhm_col.append(fwhm_finder(line=this_slice.sum(axis=1), pixel_size_mm=pixel_size_mm))  # column-wise

    fwhm_row_std, rn_row = round_to_n(
        np.std(fwhm_row),
        n=2,
        rounding_number_out=True)
    fwhm_row_mean = np.round(np.mean(fwhm_row), rn_row)

    fwhm_col_std, rn_col = round_to_n(
        np.std(fwhm_col),
        n=2,
        rounding_number_out=True)
    fwhm_col_mean = np.round(np.mean(fwhm_col), rn_col)

    if up_sample_is_needed and us_img_out:
        return fwhm_row_mean, fwhm_row_std, fwhm_col_mean, fwhm_col_std, img
    else:
        return fwhm_row_mean, fwhm_row_std, fwhm_col_mean, fwhm_col_std


def upscale_bicubic(array, scale_factor):
    # Determine the new dimensions
    new_shape = (np.array(array.shape) * scale_factor).astype(int)

    # Upscale the array using bicubic interpolation
    upscaled_array = cv2.resize(array, tuple(new_shape[::-1]), interpolation=cv2.INTER_CUBIC)

    return upscaled_array


class IndexTracker(object):
    def __init__(self, img, _cmap='jet'):
        """

        """
        fig, self.ax = plt.subplots()

        self.img = img
        vmin = np.min(self.img)
        vmax = np.max(self.img)
        self.slices, self.row_range, self.col_range = img.shape  # [slices, row, column]
        self.ind = self.slices // 2

        self.im1 = self.ax.imshow(self.img[self.ind], vmin=vmin, vmax=vmax, cmap=_cmap)
        fig.colorbar(self.im1)
        self.ax.set_title(f"Slice {self.ind}/{self.slices}")
        self.slc_row_col = []
        fig.canvas.mpl_connect('scroll_event', self.onscroll)
        fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        fig.canvas.mpl_connect("button_press_event", self.onclick)
        # plt.show()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices

        self.update()

    def on_mouse_move(self, event):
        # Clear the previous lines
        self.ax.lines.clear()
        # Get the x and y coordinates of the mouse pointer
        x = event.xdata
        y = event.ydata

        if x is not None and y is not None:
            x_y_are_in_image = 0 <= x <= (self.col_range - 1) and 0 <= y <= (self.row_range - 1)
            if x_y_are_in_image:
                # Draw the vertical line
                self.ax.axvline(x=x, color='red')
                # Draw the horizontal line
                self.ax.axhline(y=y, color='red')
        # Update the plot
        plt.draw()

    def update(self):
        # im1_data = self.im1.to_rgba(self.X[self.ind], alpha=self.im1.get_alpha())
        self.im1.set_data(self.img[self.ind])

        # Update the title with the current slice number
        self.ax.set_title(f"Slice {self.ind}/{self.slices}")

        # Redraw the figure
        self.ax.figure.canvas.draw()

    def start(self):
        plt.show()

    def onclick(self, event):
        if event.button == 1:  # Left mouse button
            print(f"Clicked at slice={self.ind} with x={event.xdata} & y={event.ydata}")
            self.slc_row_col.append([self.ind, event.xdata, event.ydata])

def uniformity_calc(image):
    # calculate the mean intensity and standard deviation
    x_dim, y_dim = image.shape
    image = image[int(0.3 * x_dim): int(0.8 * x_dim), int(0.3 * y_dim): int(0.8 * y_dim)]
    image = np.array(image)
    # Calculate the COV
    _mean = np.mean(image)
    _std = np.std(image)
    cov = _std / _mean
    return cov

