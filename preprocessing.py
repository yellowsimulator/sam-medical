import numpy as np


def get_enhanced_gray_image(image_array: np.ndarray):
    """Coverts a grayscale image to a 3 channel image

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