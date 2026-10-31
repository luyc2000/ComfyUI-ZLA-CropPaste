import os
import csv
import math
import torch
import torch.nn.functional as F

# ================== 预设读取 ==================

_PRESET_DICT = None  # name -> (w, h)

def _get_preset_file_path():
    return os.path.join(os.path.dirname(__file__), "preset.txt")

def _load_presets_if_needed():
    global _PRESET_DICT
    if _PRESET_DICT is not None:
        return

    presets = {}
    path = _get_preset_file_path()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 3:
                    continue
                name = row[0].strip()
                if name.startswith('"') and name.endswith('"'):
                    name = name[1:-1]
                try:
                    w = int(row[1])
                    h = int(row[2])
                except ValueError:
                    continue
                presets[name] = (w, h)

    _PRESET_DICT = presets

def _get_all_preset_names():
    _load_presets_if_needed()
    if not _PRESET_DICT:
        return ["NO_PRESET_FILE"]
    return list(_PRESET_DICT.keys())

def _get_preset_size(name, fallback_image_bhwc):
    _load_presets_if_needed()
    if _PRESET_DICT and name in _PRESET_DICT:
        return _PRESET_DICT[name]
    _, h, w, _ = fallback_image_bhwc.shape
    return int(w), int(h)

# ================== 形状转换 ==================

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
    if mask is None:
        return None
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
        if mask.shape[1:] == (1, H, W):
            return mask[:, 0, :, :]
        if mask.shape[1:] == (H, W, 1):
            return mask[:, :, :, 0]
        raise ValueError(f"Unsupported 4D mask shape {tuple(mask.shape)}; expect [B,1,H,W] or [B,H,W,1]")

    raise ValueError(f"Unsupported mask shape {tuple(mask.shape)}")

def _is_mask_empty(mask_bhw, thresh=0.5):
    if mask_bhw is None:
        return True
    if mask_bhw.dtype == torch.bool:
        return not mask_bhw.any().item()
    return not (mask_bhw > thresh).any().item()

# ================== bbox + cover-aspect 裁剪 ==================

def _bbox_from_mask(m_bool_hw):
    ys, xs = torch.where(m_bool_hw)
    if ys.numel() == 0:
        return None
    ymin, ymax = int(ys.min()), int(ys.max())
    xmin, xmax = int(xs.min()), int(xs.max())
    return xmin, ymin, xmax, ymax

# def _min_rect_with_aspect_cover(bw, bh, ar_w, ar_h):
#     scale = max(bw / float(ar_w), bh / float(ar_h))
#     rw = int(math.floor(scale * ar_w))
#     rh = int(math.floor(scale * ar_h))
#     return rw, rh


def _min_rect_with_aspect_cover(bw, bh, ar_w, ar_h):
    scale = max(bw / float(ar_w), bh / float(ar_h))
    rw = int(math.floor(scale * ar_w))
    rh = int(math.floor(scale * ar_h))
    rw = max(rw, bw)
    rh = max(rh, bh)
    return rw, rh



def _solve_1d(xmin, xmax, length, img_len, center_target):
    left_cover  = xmax - (length - 1)
    right_cover = xmin
    left  = max(0, left_cover)
    right = min(img_len - length, right_cover)
    if left > right:
        raise ValueError("feasible interval empty while length <= image size.")
    x0_target = int(round(center_target - length / 2.0))
    if x0_target < left:
        return left
    if x0_target > right:
        return right
    return x0_target

def _ratio_from_size(w, h):
    g = math.gcd(int(w), int(h))
    g = max(g, 1)
    return int(w // g), int(h // g)

# ================== Resize 三模式：stretch / center_crop / pad_white ==================

def _interp_mode(mode):
    mode = (mode or "bicubic").lower()
    if mode == "lanczos":
        # torch 没有真正 lanczos，这里用 bicubic 近似（你之前也这么干）
        return "bicubic", True
    if mode in ("bilinear", "bicubic"):
        return mode, True
    if mode in ("nearest", "area"):
        return mode, False
    return "bicubic", True

def _resize_stretch_image(img_bhwc, out_h, out_w, mode):
    m, need_ac = _interp_mode(mode)
    x = img_bhwc.permute(0, 3, 1, 2)
    if need_ac:
        x = F.interpolate(x, size=(int(out_h), int(out_w)), mode=m, align_corners=False)
    else:
        x = F.interpolate(x, size=(int(out_h), int(out_w)), mode=m)
    return x.permute(0, 2, 3, 1)

def _resize_stretch_mask(mask_bhw, out_h, out_w):
    x = mask_bhw.unsqueeze(1)  # [B,1,H,W]
    x = F.interpolate(x, size=(int(out_h), int(out_w)), mode="nearest")
    return x.squeeze(1)

def _resize_center_crop_image(img_bhwc, out_h, out_w, mode):
    # scale-to-cover then center crop
    B, H, W, C = img_bhwc.shape
    scale = max(out_w / float(W), out_h / float(H))
    new_w = max(1, int(math.ceil(W * scale)))
    new_h = max(1, int(math.ceil(H * scale)))
    tmp = _resize_stretch_image(img_bhwc, new_h, new_w, mode)
    x0 = (new_w - out_w) // 2
    y0 = (new_h - out_h) // 2
    return tmp[:, y0:y0+out_h, x0:x0+out_w, :]

def _resize_center_crop_mask(mask_bhw, out_h, out_w):
    B, H, W = mask_bhw.shape
    scale = max(out_w / float(W), out_h / float(H))
    new_w = max(1, int(math.ceil(W * scale)))
    new_h = max(1, int(math.ceil(H * scale)))
    tmp = _resize_stretch_mask(mask_bhw, new_h, new_w)
    x0 = (new_w - out_w) // 2
    y0 = (new_h - out_h) // 2
    return tmp[:, y0:y0+out_h, x0:x0+out_w]

def _resize_pad_white_image(img_bhwc, out_h, out_w, mode):
    # scale-to-contain then pad (white)
    B, H, W, C = img_bhwc.shape
    scale = min(out_w / float(W), out_h / float(H))
    new_w = max(1, int(math.floor(W * scale)))
    new_h = max(1, int(math.floor(H * scale)))
    tmp = _resize_stretch_image(img_bhwc, new_h, new_w, mode)

    # create white canvas
    canvas = torch.ones((B, int(out_h), int(out_w), C), dtype=tmp.dtype, device=tmp.device)
    x0 = (out_w - new_w) // 2
    y0 = (out_h - new_h) // 2
    canvas[:, y0:y0+new_h, x0:x0+new_w, :] = tmp
    return canvas

def _resize_pad_mask(mask_bhw, out_h, out_w):
    B, H, W = mask_bhw.shape
    scale = min(out_w / float(W), out_h / float(H))
    new_w = max(1, int(math.floor(W * scale)))
    new_h = max(1, int(math.floor(H * scale)))
    tmp = _resize_stretch_mask(mask_bhw, new_h, new_w)

    canvas = torch.zeros((B, int(out_h), int(out_w)), dtype=tmp.dtype, device=tmp.device)
    x0 = (out_w - new_w) // 2
    y0 = (out_h - new_h) // 2
    canvas[:, y0:y0+new_h, x0:x0+new_w] = tmp
    return canvas

def _resize_with_aspect_mode_image(img_bhwc, out_h, out_w, interp, aspect_mode):
    aspect_mode = (aspect_mode or "stretch").lower()
    if aspect_mode == "stretch":
        return _resize_stretch_image(img_bhwc, out_h, out_w, interp)
    if aspect_mode == "center_crop":
        return _resize_center_crop_image(img_bhwc, out_h, out_w, interp)
    if aspect_mode == "pad_white":
        return _resize_pad_white_image(img_bhwc, out_h, out_w, interp)
    return _resize_stretch_image(img_bhwc, out_h, out_w, interp)

def _resize_with_aspect_mode_mask(mask_bhw, out_h, out_w, aspect_mode):
    aspect_mode = (aspect_mode or "stretch").lower()
    if aspect_mode == "stretch":
        return _resize_stretch_mask(mask_bhw, out_h, out_w)
    if aspect_mode == "center_crop":
        return _resize_center_crop_mask(mask_bhw, out_h, out_w)
    if aspect_mode == "pad_white":
        return _resize_pad_mask(mask_bhw, out_h, out_w)
    return _resize_stretch_mask(mask_bhw, out_h, out_w)

def _is_all_black_mask_tensor(mask_tensor, thresh=0.5):
    """
    不要求形状匹配，直接判断是否全黑/全空
    """
    if mask_tensor is None:
        return True
    if not isinstance(mask_tensor, torch.Tensor):
        return True
    if mask_tensor.numel() == 0:
        return True
    if mask_tensor.dtype == torch.bool:
        return not mask_tensor.any().item()
    return not (mask_tensor > thresh).any().item()

def _fit_rect_to_aspect_cover(desired_left, desired_top, desired_right, desired_bottom,
                              ar_w, ar_h, W, H):
    """
    输入：期望框(左上右下，右下为exclusive)，以及目标比例 ar_w:ar_h
    输出：最终框 (x0, y0, rw, rh)，满足：
      1) 最终框覆盖期望框
      2) 最终框宽高比= ar_w:ar_h
      3) 最终框完全落在画布 [0,W]x[0,H] 内
    """
    # clamp 期望框到画布
    dl = max(0, min(int(desired_left),  W))
    dt = max(0, min(int(desired_top),   H))
    dr = max(0, min(int(desired_right), W))
    db = max(0, min(int(desired_bottom),H))

    # 保底：避免负/空框
    if dr <= dl + 1:
        dr = min(W, dl + 1)
    if db <= dt + 1:
        db = min(H, dt + 1)

    bw = dr - dl
    bh = db - dt

    # 先用你验证过的 floor 版本算“覆盖所需”的比例框
    scale = max(bw / float(ar_w), bh / float(ar_h))
    rw0 = int(math.floor(scale * ar_w))
    rh0 = int(math.floor(scale * ar_h))

    # floor 可能会让 rw0/rh0 比 bw/bh 小 1 像素，强制补回“覆盖”
    rw0 = max(rw0, bw)
    rh0 = max(rh0, bh)

    # 让比例严格成立：基于 rw0 / rh0 做两个候选，取更小但仍覆盖的
    # 候选A：固定宽 -> 计算高（向上取整保证覆盖）
    rhA = int(math.ceil(rw0 * ar_h / float(ar_w)))
    rwA = rw0
    # 候选B：固定高 -> 计算宽
    rwB = int(math.ceil(rh0 * ar_w / float(ar_h)))
    rhB = rh0

    # 选面积更小的那个（都能覆盖 bw,bh）
    if rwA * rhA <= rwB * rhB:
        rw, rh = rwA, rhA
    else:
        rw, rh = rwB, rhB

    # 如果超过画布，尝试把 rw/rh 缩到画布内（仍保持比例）
    # 这里用“contain within canvas”，但仍要覆盖期望框；如果期望框本身太大，就只能贴边
    rw = min(rw, W)
    rh = min(rh, H)

    # 如果缩小后反而不够覆盖 bw/bh，说明期望框本身在某个方向太极限；
    # 那就把 rw/rh 直接拉到画布最大可用（仍保持比例）
    if rw < bw or rh < bh:
        # 用画布限制重新算一个能覆盖期望框的比例框：尽量大
        scale2 = max(bw / float(ar_w), bh / float(ar_h))
        rw2 = int(math.floor(scale2 * ar_w))
        rh2 = int(math.floor(scale2 * ar_h))
        rw2 = max(rw2, bw)
        rh2 = max(rh2, bh)

        # 若仍超画布，就只能用画布（并贴边），这时覆盖可能无法严格满足
        # 但比直接报错更符合“能跑就行”的预期
        rw = min(rw2, W)
        rh = min(rh2, H)

    # 用期望框中心做定位
    cx = (dl + dr) / 2.0
    cy = (dt + db) / 2.0

    xmin = dl
    xmax = dr - 1  # inclusive
    ymin = dt
    ymax = db - 1

    x0 = _solve_1d(xmin, xmax, rw, W, cx)
    y0 = _solve_1d(ymin, ymax, rh, H, cy)

    return int(x0), int(y0), int(rw), int(rh), dl, dt, dr, db


# ================== 节点：有 mask 则先裁剪；无 mask 则按模式 resize 全图 ==================

class CropByMask_PresetResize:
    """
    - 如果 mask1/mask2 都未连接或都为空：跳过裁剪，对整图做 resize（stretch/center_crop/pad_white）
    - 如果任一 mask 非空：按 preset 比例裁剪且覆盖 union(mask1,mask2)，然后 resize 到 preset（此时 aspect_mode 实际无影响）
    """

    @classmethod
    def INPUT_TYPES(cls):
        preset_names = _get_all_preset_names()
        interp_modes = ["bilinear", "bicubic", "nearest", "area", "lanczos"]
        aspect_modes = ["stretch", "center_crop", "pad_white"]  # 无mask时生效
        return {
            "required": {
                "image1": ("IMAGE",),
                "preset_name": (preset_names,),
                "interpolation": (interp_modes,),
                "aspect_mode": (aspect_modes,),
                "mask_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "image2": ("IMAGE",),
                "mask1": ("MASK",),   # <- 改为 optional，允许不连接
                "mask2": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("image1", "image2", "mask1", "mask2", "top", "left", "right", "bottom")
    FUNCTION = "run"
    CATEGORY = "zla/Crop"

    def run(self, image1, preset_name, interpolation, aspect_mode, mask_threshold=0.5,
            image2=None, mask1=None, mask2=None):

        img1, B, H, W, C = _to_batch_image(image1)

        if image2 is not None:
            img2, B2, H2, W2, C2 = _to_batch_image(image2)
            if (B2 != B) or (H2 != H) or (W2 != W):
                raise ValueError("image2 must have the same batch/H/W as image1 (before crop).")
        else:
            img2 = img1

        # mask 允许不连；不连就当空
        # m1 = _to_batch_mask(mask1, B, H, W)
        # m2 = _to_batch_mask(mask2, B, H, W)

        # mask 允许“不连接 / 尺寸不匹配但全黑”
        try:
            m1 = _to_batch_mask(mask1, B, H, W)
        except Exception:
            # Load Image 那种小全黑mask：直接当没连接
            if _is_all_black_mask_tensor(mask1, thresh=mask_threshold):
                m1 = None
            else:
                raise

        try:
            m2 = _to_batch_mask(mask2, B, H, W)
        except Exception:
            if _is_all_black_mask_tensor(mask2, thresh=mask_threshold):
                m2 = None
            else:
                raise

        if m1 is None:
            m1 = torch.zeros((B, H, W), dtype=torch.float32, device=img1.device)
        if m2 is None:
            m2 = torch.zeros((B, H, W), dtype=torch.float32, device=img1.device)

        target_w, target_h = _get_preset_size(preset_name, img1)
        ar_w, ar_h = _ratio_from_size(target_w, target_h)

        out_img1, out_img2, out_m1, out_m2 = [], [], [], []
        out_top = out_left = out_right = out_bottom = 0

        for b in range(B):
            mm1 = m1[b]
            mm2 = m2[b]

            # 判断“是否有 mask”
            has_mask = (mm1 > mask_threshold).any().item() or (mm2 > mask_threshold).any().item()

            if not has_mask:
                # ===== 无mask：跳过裁剪，对全图按 aspect_mode resize =====
                full_img1 = img1[b:b+1, :, :, :]
                full_img2 = img2[b:b+1, :, :, :]
                full_m1   = mm1.unsqueeze(0)  # [1,H,W]
                full_m2   = mm2.unsqueeze(0)

                r_img1 = _resize_with_aspect_mode_image(full_img1, target_h, target_w, interpolation, aspect_mode)
                r_img2 = _resize_with_aspect_mode_image(full_img2, target_h, target_w, interpolation, aspect_mode)
                r_m1   = _resize_with_aspect_mode_mask(full_m1, target_h, target_w, aspect_mode)
                r_m2   = _resize_with_aspect_mode_mask(full_m2, target_h, target_w, aspect_mode)

                if b == 0:
                    # 没裁剪：框就是原图
                    out_top, out_left = 0, 0
                    out_right, out_bottom = int(W), int(H)

                out_img1.append(r_img1)
                out_img2.append(r_img2)
                out_m1.append(r_m1)
                out_m2.append(r_m2)
                continue

            # ===== 有mask：先裁剪到目标比例并覆盖 union(mask1,mask2)，再 resize（aspect_mode 等价）=====
            mm1b = (mm1 > mask_threshold)
            mm2b = (mm2 > mask_threshold)
            union = torch.logical_or(mm1b, mm2b)
            bbox = _bbox_from_mask(union)

            if bbox is None:
                # 理论上 has_mask=True 不会进来；兜底当无mask处理
                full_img1 = img1[b:b+1, :, :, :]
                full_img2 = img2[b:b+1, :, :, :]
                full_m1   = mm1.unsqueeze(0)
                full_m2   = mm2.unsqueeze(0)
                r_img1 = _resize_with_aspect_mode_image(full_img1, target_h, target_w, interpolation, aspect_mode)
                r_img2 = _resize_with_aspect_mode_image(full_img2, target_h, target_w, interpolation, aspect_mode)
                r_m1   = _resize_with_aspect_mode_mask(full_m1, target_h, target_w, aspect_mode)
                r_m2   = _resize_with_aspect_mode_mask(full_m2, target_h, target_w, aspect_mode)
                if b == 0:
                    out_top, out_left = 0, 0
                    out_right, out_bottom = int(W), int(H)
                out_img1.append(r_img1); out_img2.append(r_img2); out_m1.append(r_m1); out_m2.append(r_m2)
                continue

            xmin, ymin, xmax, ymax = bbox
            bw = xmax - xmin + 1
            bh = ymax - ymin + 1
            xc, yc = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0

            rw, rh = _min_rect_with_aspect_cover(bw, bh, ar_w, ar_h)
            if rw > W or rh > H:
                raise ValueError(
                    f"Preset aspect {target_w}:{target_h} cannot be satisfied while covering union(mask1,mask2). "
                    f"Need {rw}x{rh}, image is {W}x{H}."
                )

            x0 = _solve_1d(xmin, xmax, rw, W, xc)
            y0 = _solve_1d(ymin, ymax, rh, H, yc)

            if b == 0:
                out_top, out_left = int(y0), int(x0)
                out_right, out_bottom = int(x0 + rw), int(y0 + rh)

            c_img1 = img1[b:b+1, y0:y0+rh, x0:x0+rw, :]
            c_img2 = img2[b:b+1, y0:y0+rh, x0:x0+rw, :]
            c_m1   = mm1[y0:y0+rh, x0:x0+rw].unsqueeze(0)
            c_m2   = mm2[y0:y0+rh, x0:x0+rw].unsqueeze(0)

            # 裁剪后比例=目标比例，aspect_mode 三种效果相同，所以直接 stretch 即可
            r_img1 = _resize_stretch_image(c_img1, target_h, target_w, interpolation)
            r_img2 = _resize_stretch_image(c_img2, target_h, target_w, interpolation)
            r_m1   = _resize_stretch_mask(c_m1, target_h, target_w)
            r_m2   = _resize_stretch_mask(c_m2, target_h, target_w)

            out_img1.append(r_img1)
            out_img2.append(r_img2)
            out_m1.append(r_m1)
            out_m2.append(r_m2)

        out_img1 = torch.cat(out_img1, dim=0)
        out_img2 = torch.cat(out_img2, dim=0)
        out_m1   = torch.cat(out_m1, dim=0)
        out_m2   = torch.cat(out_m2, dim=0)

        return (out_img1, out_img2, out_m1, out_m2, out_top, out_left, out_right, out_bottom)



# ================== 节点：有 mask 则先裁剪；无 mask 则按模式 resize 全图 ==================

class CropByMask_PresetResize2:
    """
    - 如果 mask1/mask2 都未连接或都为空：跳过裁剪，对整图做 resize（stretch/center_crop/pad_white）
    - 如果任一 mask 非空：按 preset 比例裁剪且覆盖 union(mask1,mask2)，然后 resize 到 preset（此时 aspect_mode 实际无影响）
    """

    @classmethod
    def INPUT_TYPES(cls):
        preset_names = _get_all_preset_names()
        interp_modes = ["bilinear", "bicubic", "nearest", "area", "lanczos"]
        aspect_modes = ["stretch", "center_crop", "pad_white"]  # 无mask时生效
        return {
            "required": {
                "image1": ("IMAGE",),
                "preset_name": (preset_names,),
                "interpolation": (interp_modes,),
                "aspect_mode": (aspect_modes,),
                "mask_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),

                # ===== 新增：bbox 四边扩/缩（像素），正=扩大，负=缩小 =====
                "pad_left": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "pad_right": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "pad_top": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "pad_bottom": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "debug_log": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image2": ("IMAGE",),
                "mask1": ("MASK",),   # <- optional，允许不连接
                "mask2": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("image1", "image2", "mask1", "mask2", "top", "left", "right", "bottom")
    FUNCTION = "run"
    CATEGORY = "zla/Crop"

    def run(self, image1, preset_name, interpolation, aspect_mode,
            mask_threshold=0.5,
            pad_left=0, pad_right=0, pad_top=0, pad_bottom=0,
            debug_log=False,
            image2=None, mask1=None, mask2=None):

        img1, B, H, W, C = _to_batch_image(image1)

        if image2 is not None:
            img2, B2, H2, W2, C2 = _to_batch_image(image2)
            if (B2 != B) or (H2 != H) or (W2 != W):
                raise ValueError("image2 must have the same batch/H/W as image1 (before crop).")
        else:
            img2 = img1

        # mask 允许“不连接 / 尺寸不匹配但全黑”
        try:
            m1 = _to_batch_mask(mask1, B, H, W)
        except Exception:
            if _is_all_black_mask_tensor(mask1, thresh=mask_threshold):
                m1 = None
            else:
                raise

        try:
            m2 = _to_batch_mask(mask2, B, H, W)
        except Exception:
            if _is_all_black_mask_tensor(mask2, thresh=mask_threshold):
                m2 = None
            else:
                raise

        if m1 is None:
            m1 = torch.zeros((B, H, W), dtype=torch.float32, device=img1.device)
        if m2 is None:
            m2 = torch.zeros((B, H, W), dtype=torch.float32, device=img1.device)

        target_w, target_h = _get_preset_size(preset_name, img1)
        ar_w, ar_h = _ratio_from_size(target_w, target_h)

        out_img1, out_img2, out_m1, out_m2 = [], [], [], []
        out_top = out_left = out_right = out_bottom = 0

        for b in range(B):
            mm1 = m1[b]
            mm2 = m2[b]

            # 判断“是否有 mask”
            has_mask = (mm1 > mask_threshold).any().item() or (mm2 > mask_threshold).any().item()

            if not has_mask:
                # ===== 无mask：跳过裁剪，对全图按 aspect_mode resize =====
                full_img1 = img1[b:b+1, :, :, :]
                full_img2 = img2[b:b+1, :, :, :]
                full_m1 = mm1.unsqueeze(0)  # [1,H,W]
                full_m2 = mm2.unsqueeze(0)

                r_img1 = _resize_with_aspect_mode_image(full_img1, target_h, target_w, interpolation, aspect_mode)
                r_img2 = _resize_with_aspect_mode_image(full_img2, target_h, target_w, interpolation, aspect_mode)
                r_m1 = _resize_with_aspect_mode_mask(full_m1, target_h, target_w, aspect_mode)
                r_m2 = _resize_with_aspect_mode_mask(full_m2, target_h, target_w, aspect_mode)

                if b == 0:
                    out_top, out_left = 0, 0
                    out_right, out_bottom = int(W), int(H)

                out_img1.append(r_img1)
                out_img2.append(r_img2)
                out_m1.append(r_m1)
                out_m2.append(r_m2)
                continue

            # ===== 有mask：先裁剪到目标比例并覆盖 union(mask1,mask2)，再 resize =====
            mm1b = (mm1 > mask_threshold)
            mm2b = (mm2 > mask_threshold)
            union = torch.logical_or(mm1b, mm2b)
            bbox = _bbox_from_mask(union)

            if bbox is None:
                # 兜底当无mask处理
                full_img1 = img1[b:b+1, :, :, :]
                full_img2 = img2[b:b+1, :, :, :]
                full_m1 = mm1.unsqueeze(0)
                full_m2 = mm2.unsqueeze(0)
                r_img1 = _resize_with_aspect_mode_image(full_img1, target_h, target_w, interpolation, aspect_mode)
                r_img2 = _resize_with_aspect_mode_image(full_img2, target_h, target_w, interpolation, aspect_mode)
                r_m1 = _resize_with_aspect_mode_mask(full_m1, target_h, target_w, aspect_mode)
                r_m2 = _resize_with_aspect_mode_mask(full_m2, target_h, target_w, aspect_mode)
                if b == 0:
                    out_top, out_left = 0, 0
                    out_right, out_bottom = int(W), int(H)
                out_img1.append(r_img1); out_img2.append(r_img2); out_m1.append(r_m1); out_m2.append(r_m2)
                continue

            xmin, ymin, xmax, ymax = bbox  # inclusive

            # ===== 1) 先在 bbox 上做四边扩/缩（此时不考虑长宽比）=====
            # 转为 exclusive 便于计算：right=xmax+1, bottom=ymax+1
            l0, t0, r0, b0 = int(xmin), int(ymin), int(xmax) + 1, int(ymax) + 1

            # 正数=扩大：left/top 往外是减，right/bottom 往外是加
            l1 = l0 - int(pad_left)
            r1 = r0 + int(pad_right)
            t1 = t0 - int(pad_top)
            b1 = b0 + int(pad_bottom)

            # ===== 2) clamp，不报错 =====
            l1 = max(0, min(l1, W))
            r1 = max(0, min(r1, W))
            t1 = max(0, min(t1, H))
            b1 = max(0, min(b1, H))

            # ===== 3) max 不能 <= min：要报错 =====
            if r1 <= l1 or b1 <= t1:
                raise ValueError(
                    f"Expanded bbox invalid after clamp: LTRB=({l1},{t1},{r1},{b1}), "
                    f"pads LRTB=({pad_left},{pad_right},{pad_top},{pad_bottom})."
                )

            # ===== 4) 按目标长宽比做 cover（逻辑同原来）=====
            bw = r1 - l1
            bh = b1 - t1
            xc = (l1 + (r1 - 1)) / 2.0
            yc = (t1 + (b1 - 1)) / 2.0

            rw, rh = _min_rect_with_aspect_cover(bw, bh, ar_w, ar_h)

            if rw > W or rh > H:
                # 这代表：要覆盖 expanded bbox 且满足比例，所需框比整张图还大
                raise ValueError(
                    f"Preset aspect {target_w}:{target_h} cannot be satisfied while covering expanded bbox. "
                    f"Need {rw}x{rh}, expanded bbox is {bw}x{bh}, image is {W}x{H}."
                )

            x0 = _solve_1d(l1, r1 - 1, rw, W, xc)
            y0 = _solve_1d(t1, b1 - 1, rh, H, yc)

            if debug_log:
                print(f"[ZLA] preset={preset_name} target={target_w}x{target_h} ar={ar_w}:{ar_h} img={W}x{H}")
                print(f"[ZLA] bbox0(incl)=({xmin},{ymin})-({xmax},{ymax}) -> LTRB0(excl)=({l0},{t0},{r0},{b0}) size={r0-l0}x{b0-t0}")
                print(f"[ZLA] pads LRTB=({pad_left},{pad_right},{pad_top},{pad_bottom})")
                print(f"[ZLA] bbox1(clamped)=({l1},{t1},{r1},{b1}) size={bw}x{bh}")
                print(f"[ZLA] crop_box=({x0},{y0},{x0+rw},{y0+rh}) size={rw}x{rh}")

            if b == 0:
                out_top, out_left = int(y0), int(x0)
                out_right, out_bottom = int(x0 + rw), int(y0 + rh)

            c_img1 = img1[b:b+1, y0:y0+rh, x0:x0+rw, :]
            c_img2 = img2[b:b+1, y0:y0+rh, x0:x0+rw, :]
            c_m1 = mm1[y0:y0+rh, x0:x0+rw].unsqueeze(0)
            c_m2 = mm2[y0:y0+rh, x0:x0+rw].unsqueeze(0)

            # 裁剪后比例=目标比例，aspect_mode 三种效果相同，所以直接 stretch 即可
            r_img1 = _resize_stretch_image(c_img1, target_h, target_w, interpolation)
            r_img2 = _resize_stretch_image(c_img2, target_h, target_w, interpolation)
            r_m1 = _resize_stretch_mask(c_m1, target_h, target_w)
            r_m2 = _resize_stretch_mask(c_m2, target_h, target_w)

            out_img1.append(r_img1)
            out_img2.append(r_img2)
            out_m1.append(r_m1)
            out_m2.append(r_m2)

        out_img1 = torch.cat(out_img1, dim=0)
        out_img2 = torch.cat(out_img2, dim=0)
        out_m1 = torch.cat(out_m1, dim=0)
        out_m2 = torch.cat(out_m2, dim=0)

        return (out_img1, out_img2, out_m1, out_m2, out_top, out_left, out_right, out_bottom)


# NODE_CLASS_MAPPINGS = {
#     "CropByMask_PresetResize": CropByMask_PresetResize
# }
# NODE_DISPLAY_NAME_MAPPINGS = {
#     "CropByMask_PresetResize": "Crop + Resize by Preset (Cover Mask)"
# }
