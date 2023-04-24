import numpy as np
from glob import glob
from pathlib import Path
from api.v1.get_models import download_models
from segment_anything import SamAutomaticMaskGenerator
from segment_anything import sam_model_registry
from segment_anything import SamPredictor

from plots import show_mask, plot_masks_areas, plot_image
from preprocessing import get_enhanced_gray_image



def get_automatic_mask(image_array: np.ndarray,
                       model_type: str='vit_h',
                       device: str="cpu"):
    """Returns the segmented image without
       specifying target points.

    Parameters
    ----------
        image_array: the image array
        model_type: the model type. Options are:
            - vit_b
            - vit_l
            - vit_h
        device: the device to use. Options are:
            - cpu
            - cuda

    Returns
    -------
        segmented_image(np.ndarray): the segmented image
    """
    parent_path = Path(f"models/{model_type}")
    checkpoint = list(parent_path.glob('*.pth'))[0]
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image_array)
    return masks





if __name__ == '__main__':
    import cv2, os
    import matplotlib.pyplot as plt
    k = 0
    data = np.load('data/images.npz')
    image = data['X'][k]
    #plot_image(image, is_resize=True, scale=4.5)

    enhanced_image = get_enhanced_gray_image(image)
    print(enhanced_image.shape)
    masks = get_automatic_mask(enhanced_image)
    plt.figure(figsize=(15,10))
        # two subplots
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    # add title to subplot 1
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(enhanced_image)
    plot_masks_areas(masks)
    # add title to subplot 2
    model_name = 'VIT-Large'
    plt.title(f'Segmented Image vith {model_name}')
    plt.axis('off')
    plt.savefig('masks_test.png')
    #plt.show()
