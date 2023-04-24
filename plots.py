
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Union




def plot_image(image: Union[str, np.ndarray],
               is_resize: bool=False,
               scale: float=1.5, title='image'):
    """Plots an image.

    Parameters
    ----------
        image_image : the path to the image file or the image array
        is_resize : whether to resize the image or not
        scale : the scale to resize the image by

    Returns
    -------
        None
    """
    if isinstance(image, str):
        image = cv2.imread(image)
    if is_resize:
        height, width, _ = image.shape
        new_height = int(height * scale)
        new_width = int(width * scale)
        image = cv2.resize(image, (new_width, new_height))
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_mask(mask: np.ndarray, ax: plt.axes, random_color=False):
    """Show a mask on the given axes.

    source: https://github.com/facebookresearch/segment-anything/tree/main/notebooks

    Parameters
    ----------
        mask : the segmented mask
        ax : the axes to show the mask on
        random_color : whether to use a random color or not

    Returns
    -------
        None
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def plot_masks_areas(maks_list: list):
    """Iterates over a list of masks and plots them

    Parameters
    ----------
        maks_list : the list of masks
    """
    if len(maks_list) == 0:
        return
    sorted_maksl_list = sorted(maks_list, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for mask_item_dict in sorted_maksl_list:
        m = mask_item_dict['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))
