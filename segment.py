import numpy as np
from glob import glob
from pathlib import Path
from api.v1.get_models import download_models
from segment_anything import SamAutomaticMaskGenerator
from segment_anything import sam_model_registry
from segment_anything import SamPredictor



def get_automatic_mask(image_array: np.ndarray,
                        model_type: str='vit_h',
                        device: str="cpu"):
    """Returns the segmented image without
       specifying target points.

    Parameters
    ----------
        image_array: the image array
        model_type: the model type
        device: the device to use

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
    ...
    #download_models()