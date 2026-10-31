import os
import csv
import math
import torch
import torch.nn.functional as F

# ================== 预设读取相关 ==================

_PRESET_DICT = None  # name -> (w, h)


def _get_preset_file_path():
    """preset.txt 放在当前 py 文件同目录"""
    return os.path.join(os.path.dirname(__file__), "preset.txt")


def _load_presets_if_needed():
    """从 preset.txt 读取所有预设到 _PRESET_DICT"""
    global _PRESET_DICT
    if _PRESET_DICT is not None:
        return

    presets = {}
    path = _get_preset_file_path()

    if not os.path.exists(path):
        _PRESET_DICT = presets
        return

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            # 每行格式："name", width, height
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
    """返回所有预设名字，用于下拉菜单"""
    _load_presets_if_needed()
    if not _PRESET_DICT:
        return ["NO_PRESET_FILE"]
    return list(_PRESET_DICT.keys())


def _get_preset_size(name, fallback_image):
    """
    根据 preset 名字返回 (w, h)。
    如果名字不存在，就用 image1 的尺寸做兜底。
    fallback_image: IMAGE tensor [B,H,W,C]
    """
    _load_presets_if_needed()
    if _PRESET_DICT and name in _PRESET_DICT:
        return _PRESET_DICT[name]
    _, h, w, _ = fallback_image.shape
    return int(w), int(h)


# ================== 基础插值 helper ==================

def _resize_image_bhwc(img_bhwc, out_h, out_w, mode):
    """
    IMAGE：输入 [B,H,W,C]，输出 [B,H,W,C]，数值范围保持不变。
    mode: 'bilinear' 'bicubic' 'nearest' 'area' 'lanczos'
    - 'lanczos' 暂用 bicubic 近似
    """
    b, h, w, c = img_bhwc.shape
    mode = (mode or "bicubic").lower()

    if mode == "lanczos":
        # 简单处理成 bicubic 近似
        mode = "bicubic"

    x = img_bhwc.permute(0, 3, 1, 2)  # [B,C,H,W]

    if mode in ("bilinear", "bicubic"):
        x = F.interpolate(
            x,
            size=(int(out_h), int(out_w)),
            mode=mode,
            align_corners=False,
        )
    elif mode in ("nearest", "area"):
        x = F.interpolate(
            x,
            size=(int(out_h), int(out_w)),
            mode=mode,
        )
    else:
        # 未知模式兜底 bicubic
        x = F.interpolate(
            x,
            size=(int(out_h), int(out_w)),
            mode="bicubic",
            align_corners=False,
        )

    return x.permute(0, 2, 3, 1)  # 回到 [B,H,W,C]


def _ensure_mask_bhw(mask):
    """
    把 MASK 统一成 [B,H,W] 形状。
    支持 [H,W] / [B,H,W] / [B,H,W,1]
    """
    if mask is None:
        return None

    if mask.dim() == 4 and mask.shape[-1] == 1:
        mask = mask[..., 0]  # [B,H,W]

    if mask.dim() == 2:
        mask = mask.unsqueeze(0)  # [1,H,W]

    if mask.dim() != 3:
        raise ValueError(f"Unexpected mask shape: {mask.shape}")

    return mask


def _resize_mask_base(mask_bhw, out_h, out_w):
    """
    基础最近邻插值，把 [B,H,W] resize 到 [B,out_h,out_w]。
    """
    b, h, w = mask_bhw.shape
    x = mask_bhw.unsqueeze(1)  # [B,1,H,W]
    x = F.interpolate(x, size=(int(out_h), int(out_w)), mode="nearest")
    x = x.squeeze(1)  # [B,H,W]
    return x


# ================== 带长宽比模式的 helper ==================

def _resize_image_with_aspect(img_bhwc, target_h, target_w, aspect_mode, interp):
    """
    对 IMAGE 做 resize，并根据 aspect_mode 决定：
      - stretch      : 直接拉伸到 target_w x target_h
      - center_crop  : 按“铺满”缩放，再居中裁切
      - pad_white    : 按“等比缩放 + 居中加白边”输出
    """
    aspect_mode = (aspect_mode or "stretch").lower()
    b, h, w, c = img_bhwc.shape

    if h <= 0 or w <= 0:
        return _resize_image_bhwc(img_bhwc, target_h, target_w, interp)

    if aspect_mode == "stretch":
        return _resize_image_bhwc(img_bhwc, target_h, target_w, interp)

    # 原始宽高比
    scale_w = target_w / float(w)
    scale_h = target_h / float(h)

    if aspect_mode == "center_crop":
        # 以较大的比例缩放，保证能铺满，再裁切
        scale = max(scale_w, scale_h)
    elif aspect_mode == "pad_white":
        # 以较小的比例缩放，保证全部内容都在，再加边
        scale = min(scale_w, scale_h)
    else:
        # 未知模式兜底拉伸
        return _resize_image_bhwc(img_bhwc, target_h, target_w, interp)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = _resize_image_bhwc(img_bhwc, new_h, new_w, interp)

    if aspect_mode == "center_crop":
        # 如果数值问题造成缩放后仍然小于目标，就回退到拉伸
        if new_w < target_w or new_h < target_h:
            return _resize_image_bhwc(img_bhwc, target_h, target_w, interp)

        top = (new_h - target_h) // 2
        left = (new_w - target_w) // 2
        return resized[:, top:top + target_h, left:left + target_w, :]

    if aspect_mode == "pad_white":
        # 白底（假设图像是 0~1，ones 就是白色）
        out = torch.ones(
            (b, target_h, target_w, c),
            dtype=img_bhwc.dtype,
            device=img_bhwc.device,
        )
        top = (target_h - new_h) // 2
        left = (target_w - new_w) // 2
        out[:, top:top + new_h, left:left + new_w, :] = resized
        return out

    # 理论不会到这里，再兜底一次拉伸
    return _resize_image_bhwc(img_bhwc, target_h, target_w, interp)


def _resize_mask_with_aspect(mask, target_h, target_w, aspect_mode):
    """
    对 MASK 做与 IMAGE 一致的几何变换（最近邻），并保持 [B,H,W]：
      - stretch      : 直接拉伸
      - center_crop  : 馈满+中心裁切
      - pad_white    : 等比缩放+黑边填充
    """
    mask_bhw = _ensure_mask_bhw(mask)
    b, h, w = mask_bhw.shape

    if h <= 0 or w <= 0:
        return _resize_mask_base(mask_bhw, target_h, target_w)

    aspect_mode = (aspect_mode or "stretch").lower()

    if aspect_mode == "stretch":
        return _resize_mask_base(mask_bhw, target_h, target_w)

    scale_w = target_w / float(w)
    scale_h = target_h / float(h)

    if aspect_mode == "center_crop":
        scale = max(scale_w, scale_h)
    elif aspect_mode == "pad_white":
        scale = min(scale_w, scale_h)
    else:
        return _resize_mask_base(mask_bhw, target_h, target_w)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = _resize_mask_base(mask_bhw, new_h, new_w)  # [B,new_h,new_w]

    if aspect_mode == "center_crop":
        if new_w < target_w or new_h < target_h:
            return _resize_mask_base(mask_bhw, target_h, target_w)
        top = (new_h - target_h) // 2
        left = (new_w - target_w) // 2
        return resized[:, top:top + target_h, left:left + target_w]

    if aspect_mode == "pad_white":
        out = torch.zeros(
            (b, target_h, target_w),
            dtype=mask_bhw.dtype,
            device=mask_bhw.device,
        )
        top = (target_h - new_h) // 2
        left = (target_w - new_w) // 2
        out[:, top:top + new_h, left:left + new_w] = resized
        return out

    # 兜底
    return _resize_mask_base(mask_bhw, target_h, target_w)


# ================== 节点本体 ==================

class PresetResize:
    """
    从 preset.txt 中选择一个分辨率，对 image1 / image2 / mask1 / mask2 统一 resize。

    输入：
      - image1: 必选 IMAGE
      - image2: 可选 IMAGE
      - mask1:  可选 MASK (B,H,W)
      - mask2:  可选 MASK (B,H,W)
      - preset_name: 下拉，从 preset.txt 读取
      - interpolation: 下拉，插值方式
      - aspect_mode: 下拉，长宽比处理模式（stretch / center_crop / pad_white）

    输出：
      - image1_resized: IMAGE
      - image2_resized: IMAGE
      - mask1_resized:  MASK (B,H,W)
      - mask2_resized:  MASK (B,H,W)
    """

    @classmethod
    def INPUT_TYPES(cls):
        preset_names = _get_all_preset_names()
        interp_modes = ["bilinear", "bicubic", "nearest", "area", "lanczos"]
        aspect_modes = ["stretch", "center_crop", "pad_white"]

        return {
            "required": {
                "image1": ("IMAGE",),
                "preset_name": (preset_names,),        # 下拉：尺寸预设
                "interpolation": (interp_modes,),      # 下拉：插值算法
                "aspect_mode": (aspect_modes,),        # 下拉：长宽比策略
            },
            "optional": {
                "image2": ("IMAGE",),
                "mask1": ("MASK",),
                "mask2": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("image1", "image2", "mask1", "mask2")
    FUNCTION = "resize"
    CATEGORY = "zla/Crop"

    def resize(
        self,
        image1,
        preset_name,
        interpolation,
        aspect_mode,
        image2=None,
        mask1=None,
        mask2=None,
    ):
        # 确保 image1 有 batch 维度
        if image1.dim() == 3:
            image1 = image1.unsqueeze(0)

        b, h, w, c = image1.shape
        device = image1.device

        # 目标尺寸（w,h）
        target_w, target_h = _get_preset_size(preset_name, image1)

        # image1
        img1_out = _resize_image_with_aspect(
            image1, target_h, target_w, aspect_mode, interpolation
        )

        # image2：如果没传，就复用 image1_out
        if image2 is not None:
            if image2.dim() == 3:
                image2 = image2.unsqueeze(0)
            img2_out = _resize_image_with_aspect(
                image2, target_h, target_w, aspect_mode, interpolation
            )
        else:
            img2_out = img1_out

        # mask1：输出 [B,H,W]，没有则输出全 0
        if mask1 is not None:
            m1_out = _resize_mask_with_aspect(
                mask1, target_h, target_w, aspect_mode
            )
        else:
            m1_out = torch.zeros(
                (b, target_h, target_w),
                dtype=torch.float32,
                device=device,
            )

        # mask2：有则单独 resize，没有则复用 mask1
        if mask2 is not None:
            m2_out = _resize_mask_with_aspect(
                mask2, target_h, target_w, aspect_mode
            )
        else:
            m2_out = m1_out

        return (img1_out, img2_out, m1_out, m2_out)
