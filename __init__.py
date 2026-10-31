from .crop_by_mask_center import CropByMask_SpecPixels
from .crop_by_aspect_cover_mask import CropByMask_AspectRatio
from .resize_from_preset import PresetResize
from .crop_resize_by_mask_preset import CropByMask_PresetResize
from .crop_resize_by_mask_preset import CropByMask_PresetResize2
from .crop_paste_by_mask import CropByMask_PresetResizeInfo, PasteByMaskCropInfo

WEB_DIRECTORY = "./web"



NODE_CLASS_MAPPINGS = {
    "CropByMask_PresetResizeInfo": CropByMask_PresetResizeInfo,
    "PasteByMaskCropInfo": PasteByMaskCropInfo,
    "CropByMask_SpecPixels": CropByMask_SpecPixels,
    "CropByMask_AspectRatio": CropByMask_AspectRatio,
    "PresetResize": PresetResize,
    "CropByMask_PresetResize": CropByMask_PresetResize,
    "CropByMask_PresetResize2": CropByMask_PresetResize2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CropByMask_PresetResizeInfo": "Crop by Mask and Resize + Info ZLA",
    "PasteByMaskCropInfo": "Paste Processed Crop ZLA",
    "CropByMask_SpecPixels": "Crop by Mask to Spec Pixels",
    "CropByMask_AspectRatio": "Crop by Mask to Aspect Ratio",
    "PresetResize": "Resize to Preset Size",
    "CropByMask_PresetResize": "Crop by Mask and Resize to Preset Size",
    "CropByMask_PresetResize2": "Crop by Mask and Resize to Preset Size 2",
}

print("[ZLA] NODE_CLASS_MAPPINGS keys =", list(NODE_CLASS_MAPPINGS.keys()))
print("[ZLA] has CropByMask_PresetResize =", "CropByMask_PresetResize" in NODE_CLASS_MAPPINGS)
