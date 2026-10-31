import torch
import math

# ========= helpers =========

def _to_batch_image(image):
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"image must be torch.Tensor, got {type(image)}")
    if image.dim() == 3:  # [H,W,C]
        H, W, C = image.shape
        return image.unsqueeze(0), 1, H, W, C
    if image.dim() == 4:  # [B,H,W,C]
        B, H, W, C = image.shape
        return image, B, H, W, C
    raise ValueError(f"Unsupported image shape {tuple(image.shape)}; expect [H,W,C] or [B,H,W,C].")

def _to_batch_mask(mask, B, H, W):
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"mask must be torch.Tensor, got {type(mask)}")
    if mask.dim() == 2:           # [H,W] -> [B,H,W]
        if mask.shape != (H, W):
            raise ValueError(f"mask [H,W] must match image H,W. Got {mask.shape} vs {(H,W)}")
        return mask.unsqueeze(0).repeat(B, 1, 1)
    if mask.dim() == 3:           # [B,H,W] or [1,H,W]
        if mask.shape[1:] != (H, W):
            raise ValueError(f"mask H,W must match image. Got {mask.shape[1:]} vs {(H,W)}")
        if mask.shape[0] == 1 and B > 1:
            return mask.repeat(B, 1, 1)
        if mask.shape[0] != B and mask.shape[0] != 1:
            raise ValueError(f"mask batch {mask.shape[0]} != image batch {B}")
        return mask
    if mask.dim() == 4:           # [B,1,H,W] / [B,H,W,1]
        if mask.shape[0] != B:
            raise ValueError(f"mask batch {mask.shape[0]} != image batch {B}")
        if mask.shape[1:] == (1, H, W):     # [B,1,H,W]
            return mask[:, 0, :, :]
        if mask.shape[1:] == (H, W, 1):     # [B,H,W,1]
            return mask[:, :, :, 0]
        raise ValueError(f"Unsupported 4D mask shape {tuple(mask.shape)}; expect [B,1,H,W] or [B,H,W,1]")
    raise ValueError(f"Unsupported mask shape {tuple(mask.shape)}")

def _bbox_from_mask(m, thresh=0.5):
    if m.dtype != torch.bool:
        m = m > thresh
    ys, xs = torch.where(m)
    if ys.numel() == 0:
        return None
    ymin, ymax = int(ys.min()), int(ys.max())
    xmin, xmax = int(xs.min()), int(xs.max())
    return xmin, ymin, xmax, ymax

def _center_from_bbox(bbox, H, W):
    if bbox is None:
        return (W - 1) / 2.0, (H - 1) / 2.0
    xmin, ymin, xmax, ymax = bbox
    return (xmin + xmax) / 2.0, (ymin + ymax) / 2.0

def _min_rect_with_aspect_cover(bw, bh, ar_w, ar_h):
    scale = max(bw / float(ar_w), bh / float(ar_h))
    rw = int(math.ceil(scale * ar_w))
    rh = int(math.ceil(scale * ar_h))
    return rw, rh

def _solve_1d(xmin, xmax, length, img_len, center_target):
    left_cover  = xmax - (length - 1)
    right_cover = xmin
    left  = max(0, left_cover)
    right = min(img_len - length, right_cover)
    if left > right:
        raise ValueError("Internal error: feasible interval empty while length <= image size.")
    x0_target = int(round(center_target - length / 2.0))
    if x0_target < left:
        return left, False
    if x0_target > right:
        return right, False
    return x0_target, True

def _is_all_zero_mask_generic(mask_tensor, thresh=0.5):
    if not isinstance(mask_tensor, torch.Tensor):
        return True
    if mask_tensor.dtype == torch.bool:
        return not mask_tensor.any().item()
    else:
        return not (mask_tensor > thresh).any().item()

# ========= node =========

class CropByMask_AspectRatio:
    """
    严格长宽比 + 中心优先 + mask并集覆盖（mask2 可选，尺寸不匹配且全空则忽略）：
      - 覆盖区域 = union(mask, mask2) 的外接框（mask2 未连或“全空”则忽略）
      - 必须保持 aspect_w:aspect_h；若覆盖并集所需的最小比例矩形放不进图内 => 报错
      - 能放下时，优先让裁剪矩形中心 == 并集外接框中心；靠边放不下则放弃中心一致但仍覆盖
      - 输出的裁剪后 mask 只来自输入的 mask（忽略 mask2）
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "aspect_w": ("INT", {"default": 2, "min": 1, "max": 4096, "step": 1}),
                "aspect_h": ("INT", {"default": 3, "min": 1, "max": 4096, "step": 1}),
                "mask_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "mask2": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("cropped_image", "cropped_mask", "top", "left", "right", "bottom")
    FUNCTION = "crop"
    CATEGORY = "zla/Crop"

    def crop(self, image, mask, aspect_w, aspect_h, mask_threshold=0.5, mask2=None):

        print(f"crop>> aspect_w: {aspect_w}, aspect_h: {aspect_h}  ")
        if aspect_w <= 0 or aspect_h <= 0:
            raise ValueError("aspect_w and aspect_h must be positive integers.")

        img, B, H, W, C = _to_batch_image(image)
        msk1 = _to_batch_mask(mask,  B, H, W)

        # mask2：可选；若尺寸不匹配但“全空”，则忽略
        if mask2 is None:
            msk2 = torch.zeros((B, H, W), dtype=msk1.dtype, device=msk1.device)
        else:
            try:
                msk2 = _to_batch_mask(mask2, B, H, W)
            except Exception:
                try:
                    all_zero = _is_all_zero_mask_generic(mask2, thresh=mask_threshold)
                except Exception:
                    all_zero = True
                if all_zero:
                    msk2 = torch.zeros((B, H, W), dtype=msk1.dtype, device=msk1.device)
                else:
                    raise ValueError(
                        "mask2 shape does not match image, and mask2 is not empty. "
                        "Please align mask2 size to image or leave it unconnected."
                    )

        out_imgs, out_msks = [], []
        out_top = out_left = out_right = out_bottom = 0

        for b in range(B):
            m1 = msk1[b]
            m2 = msk2[b]

            m1b = (m1 > mask_threshold) if m1.dtype != torch.bool else m1
            m2b = (m2 > mask_threshold) if m2.dtype != torch.bool else m2
            union_mask = torch.logical_or(m1b, m2b)
            bbox = _bbox_from_mask(union_mask, thresh=0.5)

            if bbox is None:
                xc, yc = (W - 1) / 2.0, (H - 1) / 2.0
                rw, rh = aspect_w, aspect_h

                print(f"tag1>> xc: {xc}, yc: {yc}, rw: {rw}, rh: {rh}")

                if rw > W or rh > H:
                    scale = min(W / float(aspect_w), H / float(aspect_h))
                    if scale <= 0:
                        raise ValueError("Aspect cannot fit inside the image at any scale.")
                    rw = max(1, int(math.floor(aspect_w * scale)))
                    rh = max(1, int(math.floor(aspect_h * scale)))
                    rw = min(rw, W); rh = min(rh, H)
                x0, _ = _solve_1d(0, W-1, rw, W, xc)
                y0, _ = _solve_1d(0, H-1, rh, H, yc)
            else:
                xmin, ymin, xmax, ymax = bbox
                print(f"xmin:{xmin} ymin:{ymin} xmax:{xmax} ymax:{ymax}")
                bw = xmax - xmin + 1
                bh = ymax - ymin + 1
                print(f"bw:{bw} bh:{bh}")
                xc, yc = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
                print(f"xc:{xc} yc:{yc}")


                rw, rh = _min_rect_with_aspect_cover(bw, bh, aspect_w, aspect_h)
                if rw > W or rh > H:
                    raise ValueError(
                        f"Aspect {aspect_w}:{aspect_h} cannot be satisfied: "
                        f"required {rw}x{rh} exceeds image {W}x{H} while covering union(mask, mask2)."
                    )
                x0, _ = _solve_1d(xmin, xmax, rw, W, xc)
                y0, _ = _solve_1d(ymin, ymax, rh, H, yc)

            crop_img = img[b, y0:y0+rh, x0:x0+rw, :]
            crop_m1 = m1[y0:y0+rh, x0:x0+rw]  # 输出仅来自 mask1

            if b == 0:
                out_top, out_left = int(y0), int(x0)
                out_right, out_bottom = int(x0 + rw), int(y0 + rh)

            out_imgs.append(crop_img.unsqueeze(0))
            out_msks.append(crop_m1.unsqueeze(0))

        cropped_img_b = torch.cat(out_imgs, dim=0)
        cropped_msk_b = torch.cat(out_msks, dim=0)

        return (cropped_img_b, cropped_msk_b, out_top, out_left, out_right, out_bottom)
