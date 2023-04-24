import pydicom
import numpy as np


def read_dicom_image(file_path: str):
    """Reads a dicom image and returns the image array.

    Parameters
    ----------
        file_path: the path to the dicom image

    Returns
    -------
        img_array(np.ndarray): the image array
    """
    ds = pydicom.dcmread(file_path)
    img_data = ds.pixel_array
    img_array = np.array(img_data)
    return img_array


def get_enhanced_gray_image(image_array: np.ndarray):
    """Coverts a grayscale image to a 3 channel image.
       Takes as input an image array of shape (H, W, 1)
       and returns an image array of shape (H, W, 3)

    Parameters
    ----------
        image_array: the image array

    Returns
    -------
        image_array(np.ndarray): the 3 channel image
    """
    image_array = np.repeat(image_array, 3, axis=2)
    image_array = np.clip(image_array, 0, 1)
    image_array = (image_array * 255).astype(np.uint8)
    return image_array