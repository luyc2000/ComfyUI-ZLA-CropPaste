import torch

def _to_batch_image(image):
    """
    Expect image as [B, H, W, C] or [H, W, C].
    Returns (img_bhwc, B, H, W, C)
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"image must be a torch.Tensor, got {type(image)}")

    if image.dim() == 3:
        # [H, W, C] -> [1, H, W, C]
        H, W, C = image.shape
        return image.unsqueeze(0), 1, H, W, C
    elif image.dim() == 4:
        # [B, H, W, C]
        B, H, W, C = image.shape
        return image, B, H, W, C
    else:
        raise ValueError(f"Unsupported image shape {tuple(image.shape)}, expected [H,W,C] or [B,H,W,C].")


def _to_batch_mask(mask, B, H, W):
    """
    Accept mask as [H, W], [B, H, W], or [B, 1, H, W].
    Return mask as [B, H, W]
    """
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"mask must be a torch.Tensor, got {type(mask)}")

    if mask.dim() == 2:
        # [H, W] -> [B, H, W]
        if mask.shape != (H, W):
            raise ValueError(f"mask [H,W] must match image H,W. Got {tuple(mask.shape)} vs {(H,W)}")
        mask_b = mask.unsqueeze(0).repeat(B, 1, 1)
        return mask_b

    if mask.dim() == 3:
        # Could be [B, H, W] or [1, H, W]
        if mask.shape == (H, W, 1):
            # Uncommon, convert to [1, H, W]
            mask = mask.permute(2,0,1)
        if mask.shape[1:] != (H, W):
            raise ValueError(f"mask H,W must match image. Got {tuple(mask.shape[1:])} vs {(H,W)}")

        if mask.shape[0] == 1 and B > 1:
            return mask.repeat(B, 1, 1)
        if mask.shape[0] == B:
            return mask
        if mask.shape[0] != B:
            raise ValueError(f"mask batch {mask.shape[0]} != image batch {B}")
        return mask

    if mask.dim() == 4:
        # [B, 1, H, W] or [B, H, W, 1]
        if mask.shape[0] != B:
            raise ValueError(f"mask batch {mask.shape[0]} != image batch {B}")

        # Try [B,1,H,W]
        if mask.shape[1:] == (1, H, W):
            return mask[:,0,:,:]
        # Try [B,H,W,1]
        if mask.shape[1:] == (H, W, 1):
            return mask[:,:,:,0]
        raise ValueError(f"Unsupported 4D mask shape {tuple(mask.shape)}; expected [B,1,H,W] or [B,H,W,1].")

    raise ValueError(f"Unsupported mask shape {tuple(mask.shape)}")


def _bbox_from_mask(m, thresh=0.5):
    """
    m: [H, W] float/bool tensor
    Returns (xmin, ymin, xmax, ymax) inclusive bounds; if empty -> None
    """
    if m.dtype != torch.bool:
        m = m > thresh
    idx = torch.where(m)
    if idx[0].numel() == 0:
        return None
    ymin = int(idx[0].min().item())
    ymax = int(idx[0].max().item())
    xmin = int(idx[1].min().item())
    xmax = int(idx[1].max().item())
    return xmin, ymin, xmax, ymax


def _center_from_bbox(bbox, H, W):
    """
    bbox: (xmin, ymin, xmax, ymax) or None
    If None, use image center.
    Returns (xc, yc) as float center coords in pixel space.
    """
    if bbox is None:
        xc = (W - 1) / 2.0
        yc = (H - 1) / 2.0
        return xc, yc
    xmin, ymin, xmax, ymax = bbox
    xc = (xmin + xmax) / 2.0
    yc = (ymin + ymax) / 2.0
    return xc, yc


def _clamped_crop_rect(xc, yc, crop_w, crop_h, W, H):
    """
    Given center (xc,yc) and crop size, compute top-left (x0,y0) clamped to image bounds.
    All inputs pixel units.
    Returns (x0, y0, cw, ch) ints where cw, ch match requested or are reduced if image smaller.
    """
    # Ensure int sizes
    crop_w = int(crop_w)
    crop_h = int(crop_h)

    # If requested crop is larger than image, shrink to image
    cw = min(crop_w, W)
    ch = min(crop_h, H)

    # Center to top-left
    x0 = int(round(xc - cw / 2.0))
    y0 = int(round(yc - ch / 2.0))

    # Clamp
    x0 = max(0, min(x0, W - cw))
    y0 = max(0, min(y0, H - ch))

    return x0, y0, cw, ch


# class CropByMask_SpecPixels:
#     """
#     Inputs:
#       - image: IMAGE  (torch.Tensor [B,H,W,C] or [H,W,C], float 0..1)
#       - mask:  MASK   (torch.Tensor [B,H,W] / [H,W] / [B,1,H,W]/[B,H,W,1])
#       - width:  INT   (desired crop width in pixels)
#       - height: INT   (desired crop height in pixels)
#
#     Outputs:
#       - cropped_image: IMAGE
#       - cropped_mask:  MASK
#       - x0: INT (left)
#       - y0: INT (top)
#     """
#
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "image": ("IMAGE",),
#                 "mask": ("MASK",),
#                 "width": ("INT", {"default": 512, "min": 1, "max": 16384, "step": 1}),
#                 "height": ("INT", {"default": 512, "min": 1, "max": 16384, "step": 1}),
#                 # 可选：mask阈值
#                 "mask_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
#             }
#         }
#
#     RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
#     RETURN_NAMES = ("cropped_image", "cropped_mask", "x0", "y0")
#     FUNCTION = "crop"
#     CATEGORY = "image/transform"
#
#     def crop(self, image, mask, width, height, mask_threshold=0.5):
#         img, B, H, W, C = _to_batch_image(image)
#         msk = _to_batch_mask(mask, B, H, W)
#
#         cropped_imgs = []
#         cropped_msks = []
#
#         # 为了返回代表性的左上角坐标（当batch>1时），这里返回第 1 张的坐标
#         out_x0, out_y0 = 0, 0
#
#         for b in range(B):
#             m = msk[b]  # [H, W]
#             bbox = _bbox_from_mask(m, thresh=mask_threshold)
#             xc, yc = _center_from_bbox(bbox, H, W)
#             x0, y0, cw, ch = _clamped_crop_rect(xc, yc, width, height, W, H)
#
#             if b == 0:
#                 out_x0, out_y0 = int(x0), int(y0)
#
#             # 裁剪
#             crop_img = img[b, y0:y0+ch, x0:x0+cw, :]  # [ch, cw, C]
#             crop_msk = m[y0:y0+ch, x0:x0+cw]          # [ch, cw]
#
#             # 追加 batch 维
#             cropped_imgs.append(crop_img.unsqueeze(0))
#             cropped_msks.append(crop_msk.unsqueeze(0))
#
#         cropped_img_b = torch.cat(cropped_imgs, dim=0)  # [B, ch, cw, C]
#         cropped_msk_b = torch.cat(cropped_msks, dim=0)  # [B, ch, cw]
#
#         return (cropped_img_b, cropped_msk_b, out_x0, out_y0)

class CropByMask_SpecPixels:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "width": ("INT", {"default": 512, "min": 1, "max": 16384, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "max": 16384, "step": 1}),
                "mask_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("cropped_image", "cropped_mask", "top", "left", "right", "bottom")
    FUNCTION = "crop"
    CATEGORY = "zla/Crop"

    def crop(self, image, mask, width, height, mask_threshold=0.5):
        img, B, H, W, C = _to_batch_image(image)
        msk = _to_batch_mask(mask, B, H, W)

        cropped_imgs = []
        cropped_msks = []

        # 默认输出第一张的矩形坐标
        out_top, out_left, out_right, out_bottom = 0, 0, 0, 0

        for b in range(B):
            m = msk[b]  # [H, W]
            bbox = _bbox_from_mask(m, thresh=mask_threshold)
            xc, yc = _center_from_bbox(bbox, H, W)
            x0, y0, cw, ch = _clamped_crop_rect(xc, yc, width, height, W, H)

            if b == 0:
                out_top = int(y0)
                out_left = int(x0)
                out_right = int(x0 + cw)
                out_bottom = int(y0 + ch)

            crop_img = img[b, y0:y0+ch, x0:x0+cw, :]
            crop_msk = m[y0:y0+ch, x0:x0+cw]

            cropped_imgs.append(crop_img.unsqueeze(0))
            cropped_msks.append(crop_msk.unsqueeze(0))

        cropped_img_b = torch.cat(cropped_imgs, dim=0)
        cropped_msk_b = torch.cat(cropped_msks, dim=0)

        return (cropped_img_b, cropped_msk_b, out_top, out_left, out_right, out_bottom)
