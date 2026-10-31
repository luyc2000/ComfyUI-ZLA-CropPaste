import logging
import math
import os

import numpy as np
from PIL import Image
import folder_paths
import torch
import torch.nn.functional as F

try:
    from psd_tools import PSDImage
    from psd_tools.constants import Compression
except ImportError:
    PSDImage = None

from .crop_resize_by_mask_preset import (
    _get_all_preset_names, _to_batch_image, _to_batch_mask,
    _is_all_black_mask_tensor, _get_preset_size, _ratio_from_size,
    _resize_with_aspect_mode_image, _resize_with_aspect_mode_mask,
    _bbox_from_mask, _min_rect_with_aspect_cover, _solve_1d,
    _resize_stretch_image, _resize_stretch_mask,
)


def _manual_resize_size(ratio_w, ratio_h, resize_mode, resize_length, resize_pixels_k):
    if resize_mode == "height":
        return max(1, round(resize_length * ratio_w / ratio_h)), resize_length
    if resize_mode == "width":
        return resize_length, max(1, round(resize_length * ratio_h / ratio_w))
    if resize_mode == "long_edge":
        scale = resize_length / max(ratio_w, ratio_h)
    elif resize_mode == "short_edge":
        scale = resize_length / min(ratio_w, ratio_h)
    elif resize_mode == "total_pixels":
        scale = math.sqrt(resize_pixels_k * 1024 / (ratio_w * ratio_h))
    else:
        raise ValueError(f"Unknown manual resize mode: {resize_mode}")
    return max(1, round(ratio_w * scale)), max(1, round(ratio_h * scale))


class CropByMask_PresetResizeInfo:
    """
    - 如果 mask1/mask2 都未连接或都为空：跳过裁剪，对整图做 resize（stretch/center_crop/pad_white）
    - 如果任一 mask 非空：按预设或手动比例裁剪且覆盖 union(mask1,mask2)，然后按预设或手动设置缩放（此时 aspect_mode 实际无影响）
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
                "size_mode": (["preset", "manual"], {"default": "preset", "tooltip": "preset: use preset_name as before. manual: use aspect_width/aspect_height and the manual resize controls."}),
                "aspect_width": ("INT", {"default": 1, "min": 1, "max": 8192, "step": 1, "tooltip": "Manual crop aspect ratio width, e.g. 3 for 3:2."}),
                "aspect_height": ("INT", {"default": 1, "min": 1, "max": 8192, "step": 1, "tooltip": "Manual crop aspect ratio height, e.g. 2 for 3:2."}),
                "resize_mode": (["height", "width", "long_edge", "short_edge", "total_pixels", "none"], {"default": "long_edge", "tooltip": "Manual mode only. height/width/long_edge/short_edge use resize_length; total_pixels uses resize_pixels_k; none keeps actual crop pixels."}),
                "resize_length": ("INT", {"default": 1024, "min": 1, "max": 65536, "step": 1, "tooltip": "Target dimension in pixels for height, width, long_edge or short_edge."}),
                "resize_pixels_k": ("FLOAT", {"default": 1024.0, "min": 0.001, "max": 4194304.0, "step": 1.0, "tooltip": "Target total pixels in K pixels, where 1K = 1024 pixels. 1024K = 1,048,576 pixels."}),
                "resize_alignment": (["8", "16", "32", "64", "none"], {"default": "8", "tooltip": "Manual resize only: round output width and height up to this multiple. none disables alignment; resize_mode=none always preserves actual crop pixels."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "MASK", "INT", "INT", "INT", "INT", "ZLA_CROP_INFO")
    RETURN_NAMES = ("image1", "image2", "mask1", "mask2", "top", "left", "right", "bottom", "crop_info")
    FUNCTION = "run"
    CATEGORY = "zla/Crop"

    def run(self, image1, preset_name, interpolation, aspect_mode,
            mask_threshold=0.5,
            pad_left=0, pad_right=0, pad_top=0, pad_bottom=0,
            debug_log=False,
            image2=None, mask1=None, mask2=None,
            size_mode="preset", aspect_width=1, aspect_height=1,
            resize_mode="long_edge", resize_length=1024, resize_pixels_k=1024.0,
            resize_alignment="8"):

        img1, B, H, W, C = _to_batch_image(image1)

        if image2 is not None:
            img2, B2, H2, W2, C2 = _to_batch_image(image2)
            if (B2 != B) or (H2 != H) or (W2 != W):
                raise ValueError("image2 must have the same batch/H/W as image1 (before crop).")
        else:
            img2 = img1

        try:
            m1 = _to_batch_mask(mask1, B, H, W)
        except (TypeError, ValueError):
            if _is_all_black_mask_tensor(mask1, thresh=mask_threshold):
                m1 = None
            else:
                raise

        try:
            m2 = _to_batch_mask(mask2, B, H, W)
        except (TypeError, ValueError):
            if _is_all_black_mask_tensor(mask2, thresh=mask_threshold):
                m2 = None
            else:
                raise

        if m1 is None:
            m1 = torch.zeros((B, H, W), dtype=torch.float32, device=img1.device)
        if m2 is None:
            m2 = torch.zeros((B, H, W), dtype=torch.float32, device=img1.device)

        keep_crop_pixels = size_mode == "manual" and resize_mode == "none"
        if size_mode == "manual":
            ar_w, ar_h = _ratio_from_size(aspect_width, aspect_height)
            if keep_crop_pixels:
                target_w, target_h = W, H
            else:
                target_w, target_h = _manual_resize_size(ar_w, ar_h, resize_mode, resize_length, resize_pixels_k)
                alignment = {"none": 1, "8": 8, "16": 16, "32": 32, "64": 64}[resize_alignment]
                target_w = ((target_w + alignment - 1) // alignment) * alignment
                target_h = ((target_h + alignment - 1) // alignment) * alignment
        else:
            target_w, target_h = _get_preset_size(preset_name, img1)
            ar_w, ar_h = _ratio_from_size(target_w, target_h)

        out_img1, out_img2, out_m1, out_m2 = [], [], [], []
        out_top = out_left = out_right = out_bottom = 0
        regions = []

        for b in range(B):
            mm1 = m1[b]
            mm2 = m2[b]

            has_mask = (mm1 > mask_threshold).any().item() or (mm2 > mask_threshold).any().item()

            if not has_mask:
                full_img1 = img1[b:b+1, :, :, :]
                full_img2 = img2[b:b+1, :, :, :]
                full_m1 = mm1.unsqueeze(0)  # [1,H,W]
                full_m2 = mm2.unsqueeze(0)

                if keep_crop_pixels:
                    r_img1, r_img2, r_m1, r_m2 = full_img1, full_img2, full_m1, full_m2
                else:
                    r_img1 = _resize_with_aspect_mode_image(full_img1, target_h, target_w, interpolation, aspect_mode)
                    r_img2 = _resize_with_aspect_mode_image(full_img2, target_h, target_w, interpolation, aspect_mode)
                    r_m1 = _resize_with_aspect_mode_mask(full_m1, target_h, target_w, aspect_mode)
                    r_m2 = _resize_with_aspect_mode_mask(full_m2, target_h, target_w, aspect_mode)

                if b == 0:
                    out_top, out_left = 0, 0
                    out_right, out_bottom = int(W), int(H)

                regions.append({
                    "box": (0, 0, W, H), "aspect_mode": aspect_mode,
                    "has_mask": False, "mask1": None, "mask2": None,
                })
                out_img1.append(r_img1)
                out_img2.append(r_img2)
                out_m1.append(r_m1)
                out_m2.append(r_m2)
                continue

            mm1b = (mm1 > mask_threshold)
            mm2b = (mm2 > mask_threshold)
            union = torch.logical_or(mm1b, mm2b)
            bbox = _bbox_from_mask(union)

            xmin, ymin, xmax, ymax = bbox  # inclusive

            l0, t0, r0, b0 = int(xmin), int(ymin), int(xmax) + 1, int(ymax) + 1

            l1 = l0 - int(pad_left)
            r1 = r0 + int(pad_right)
            t1 = t0 - int(pad_top)
            b1 = b0 + int(pad_bottom)

            l1 = max(0, min(l1, W))
            r1 = max(0, min(r1, W))
            t1 = max(0, min(t1, H))
            b1 = max(0, min(b1, H))

            if r1 <= l1 or b1 <= t1:
                raise ValueError(
                    f"Expanded bbox invalid after clamp: LTRB=({l1},{t1},{r1},{b1}), "
                    f"pads LRTB=({pad_left},{pad_right},{pad_top},{pad_bottom})."
                )

            bw = r1 - l1
            bh = b1 - t1
            xc = (l1 + (r1 - 1)) / 2.0
            yc = (t1 + (b1 - 1)) / 2.0

            rw, rh = _min_rect_with_aspect_cover(bw, bh, ar_w, ar_h)

            if rw > W or rh > H:
                raise ValueError(
                    f"Crop aspect {ar_w}:{ar_h} cannot be satisfied while covering expanded bbox. "
                    f"Need {rw}x{rh}, expanded bbox is {bw}x{bh}, image is {W}x{H}."
                )

            x0 = _solve_1d(l1, r1 - 1, rw, W, xc)
            y0 = _solve_1d(t1, b1 - 1, rh, H, yc)

            if debug_log:
                print(f"[ZLA] size_mode={size_mode} preset={preset_name} target={rw if keep_crop_pixels else target_w}x{rh if keep_crop_pixels else target_h} ar={ar_w}:{ar_h} img={W}x{H}")
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

            if keep_crop_pixels:
                r_img1, r_img2, r_m1, r_m2 = c_img1, c_img2, c_m1, c_m2
            else:
                r_img1 = _resize_stretch_image(c_img1, target_h, target_w, interpolation)
                r_img2 = _resize_stretch_image(c_img2, target_h, target_w, interpolation)
                r_m1 = _resize_stretch_mask(c_m1, target_h, target_w)
                r_m2 = _resize_stretch_mask(c_m2, target_h, target_w)

            regions.append({
                "box": (x0, y0, x0 + rw, y0 + rh), "aspect_mode": "stretch",
                "has_mask": True, "mask1": c_m1.clone(), "mask2": c_m2.clone(),
            })
            out_img1.append(r_img1)
            out_img2.append(r_img2)
            out_m1.append(r_m1)
            out_m2.append(r_m2)

        output_size = tuple(out_img1[0].shape[1:3])
        if keep_crop_pixels and any(tuple(image.shape[1:3]) != output_size for image in out_img1):
            raise ValueError("No-resize crops have different sizes within the batch. Process images individually or select a resize mode to create a fixed-size IMAGE batch.")
        out_img1 = torch.cat(out_img1, dim=0)
        out_img2 = torch.cat(out_img2, dim=0)
        out_m1 = torch.cat(out_m1, dim=0)
        out_m2 = torch.cat(out_m2, dim=0)

        crop_info = {"original_size": (H, W), "target_size": output_size, "regions": regions}
        return (out_img1, out_img2, out_m1, out_m2, out_top, out_left, out_right, out_bottom, crop_info)


def _restore_crop(image, target_h, target_w, height, width, aspect_mode, interpolation):
    image = _resize_stretch_image(image, target_h, target_w, interpolation)
    coverage = image.new_ones((1, height, width))
    if aspect_mode == "pad_white":
        scale = min(target_w / width, target_h / height)
        new_w = max(1, math.floor(width * scale))
        new_h = max(1, math.floor(height * scale))
        x = (target_w - new_w) // 2
        y = (target_h - new_h) // 2
        image = image[:, y:y + new_h, x:x + new_w, :]
    elif aspect_mode == "center_crop":
        scale = max(target_w / width, target_h / height)
        new_w = max(1, math.ceil(width * scale))
        new_h = max(1, math.ceil(height * scale))
        x = (new_w - target_w) // 2
        y = (new_h - target_h) // 2
        padding = (x, new_w - target_w - x, y, new_h - target_h - y)
        image = F.pad(image.permute(0, 3, 1, 2), padding, mode="replicate").permute(0, 2, 3, 1)
        valid = image.new_ones((1, 1, target_h, target_w))
        coverage = F.interpolate(F.pad(valid, padding), size=(height, width), mode="bilinear", align_corners=False)[:, 0]
    return _resize_stretch_image(image, height, width, interpolation), coverage


def _feather_mask(mask, radius, shrink_radius):
    x = mask.unsqueeze(1)
    eroded = x
    if shrink_radius > 0:
        eroded = -F.max_pool2d(-F.pad(eroded, (shrink_radius, shrink_radius, 0, 0)), (1, 2 * shrink_radius + 1), stride=1)
        eroded = -F.max_pool2d(-F.pad(eroded, (0, 0, shrink_radius, shrink_radius)), (2 * shrink_radius + 1, 1), stride=1)
    if radius <= 0:
        return eroded[:, 0]
    offsets = torch.arange(-radius, radius + 1, device=mask.device, dtype=mask.dtype)
    kernel = torch.exp(-0.5 * (offsets / max(radius / 3.0, 0.5)).square())
    kernel = kernel / kernel.sum()
    blurred = F.conv2d(F.pad(eroded, (radius, radius, 0, 0)), kernel.view(1, 1, 1, -1))
    blurred = F.conv2d(F.pad(blurred, (0, 0, radius, radius)), kernel.view(1, 1, -1, 1))[:, 0]
    # Remove the truncated Gaussian's residual at the original boundary pixels.
    interior = -F.max_pool2d(-F.pad(x, (1, 1, 1, 1)), 3, stride=1)[:, 0] > 0
    return torch.minimum(blurred, mask) * interior



def _save_aligned_psd(original, patch, coverage, left, top, filename_prefix, psd_compression="none"):
    if PSDImage is None:
        raise RuntimeError("PSD export requires psd-tools. Install this node's requirements.txt and restart ComfyUI.")
    compression = {"none": Compression.RAW, "rle": Compression.RLE}[psd_compression]
    height, width, channels = original.shape
    if channels not in (1, 3, 4):
        raise ValueError("PSD export supports grayscale, RGB, or RGBA images.")
    folder, filename, counter, _, _ = folder_paths.get_save_image_path(
        filename_prefix, folder_paths.get_output_directory(), width, height)
    path = os.path.join(folder, f"{filename}_{counter:05}_.psd")
    psd = PSDImage.new("RGBA", (width, height), compression=compression)
    original_pixels = original.detach().clamp(0, 1).mul(255).round().to(device="cpu", dtype=torch.uint8).numpy()
    patch_pixels = patch.detach().clamp(0, 1).mul(255).round().to(device="cpu", dtype=torch.uint8).numpy()
    if channels == 1:
        original_pixels = original_pixels[:, :, 0]
        patch_pixels = patch_pixels[:, :, 0]
    psd.create_pixel_layer(Image.fromarray(original_pixels).convert("RGBA"), name="Original Image", compression=compression)
    patch_image = Image.fromarray(patch_pixels).convert("RGBA")
    # Coverage only excludes pixels removed by center_crop, independent of the blend mask.
    alpha = np.asarray(patch_image.getchannel("A"), dtype=np.float32) * coverage.detach().to(device="cpu", dtype=torch.float32).numpy()
    patch_image.putalpha(Image.fromarray(np.rint(alpha).astype(np.uint8)))
    psd.create_pixel_layer(patch_image, name="Processed Image", left=left, top=top, compression=compression)
    with open(path, "xb") as handle:
        psd.save(handle)
    logging.info("Saved aligned PSD: %s", path)


class PasteByMaskCropInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "processed_image": ("IMAGE",),
                "crop_info": ("ZLA_CROP_INFO",),
                "blend_region": (["mask1", "mask2", "union", "rectangle"], {"tooltip": "mask1 / mask2: paste through that original mask; union: both masks; rectangle: entire crop."}),
                "feather_radius": ("INT", {"default": 16, "min": 0, "max": 256, "step": 1, "tooltip": "Gaussian blur radius in original-image pixels, independent of mask_shrink. Original mask edges stay at zero when feathering is enabled. 0 disables blur."}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "interpolation": (["bilinear", "bicubic", "nearest", "area", "lanczos"],),
            },
            "optional": {
                "mask_shrink": ("INT", {"default": 16, "min": 0, "max": 256, "step": 1, "tooltip": "Mask erosion in original-image pixels, independent of feather_radius. 0 disables erosion. Use a smaller value for small masks."}),
                "save_psd": ("BOOLEAN", {"default": False, "tooltip": "Save original and aligned processed image as two PSD layers, without blend masks, feathering, or opacity adjustments."}),
                "psd_filename_prefix": ("STRING", {"default": "ZLA_Paste/Aligned", "tooltip": "PSD filename prefix within ComfyUI's output directory."}),
                "psd_compression": (["none", "rle"], {"default": "none", "tooltip": "PSD layer and merged preview compression: none (uncompressed) or rle (lossless)."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("image", "blend_mask", "shrunk_mask", "feathered_mask")
    FUNCTION = "run"
    CATEGORY = "zla/Crop"
    OUTPUT_NODE = True

    def run(self, original_image, processed_image, crop_info, blend_region="mask1",
            feather_radius=16, opacity=1.0, interpolation="bilinear", mask_shrink=16,
            save_psd=False, psd_filename_prefix="ZLA_Paste/Aligned", psd_compression="none"):
        original, batch, height, width, channels = _to_batch_image(original_image)
        processed, processed_batch, _, _, processed_channels = _to_batch_image(processed_image)
        if (height, width) != tuple(crop_info["original_size"]):
            raise ValueError("original_image size must match the image used by the crop node.")
        if channels != processed_channels:
            raise ValueError("original_image and processed_image must have the same channel count.")
        regions = crop_info["regions"]
        count = max(batch, processed_batch, len(regions))
        if any(size not in (1, count) for size in (batch, processed_batch, len(regions))):
            raise ValueError("Image batches and crop_info must have matching counts, or a single item to broadcast.")
        target_h, target_w = crop_info["target_size"]
        images, masks, shrunk_masks, feathered_masks = [], [], [], []
        for index in range(count):
            region = regions[0 if len(regions) == 1 else index]
            left, top, right, bottom = region["box"]
            crop_h, crop_w = bottom - top, right - left
            output = original[0 if batch == 1 else index].unsqueeze(0).clone()
            patch = processed[0 if processed_batch == 1 else index].unsqueeze(0).to(device=original.device, dtype=original.dtype)
            patch, coverage = _restore_crop(patch, target_h, target_w, crop_h, crop_w, region["aspect_mode"], interpolation)
            if save_psd:
                _save_aligned_psd(original[0 if batch == 1 else index], patch[0], coverage[0], left, top, psd_filename_prefix, psd_compression)
            if blend_region == "rectangle" or not region["has_mask"]:
                alpha = output.new_ones((1, crop_h, crop_w))
            elif blend_region == "union":
                alpha = torch.maximum(region["mask1"], region["mask2"]).to(device=output.device, dtype=output.dtype)
            elif blend_region in ("mask1", "mask2"):
                alpha = region[blend_region].to(device=output.device, dtype=output.dtype)
            else:
                raise ValueError(f"Unknown blend_region: {blend_region}")
            alpha = alpha.clamp(0, 1)
            shrunk = _feather_mask(alpha, 0, int(mask_shrink))
            feathered = _feather_mask(alpha, int(feather_radius), int(mask_shrink))
            alpha = feathered * coverage * opacity
            shrunk_mask = output.new_zeros((1, height, width))
            feathered_mask = output.new_zeros((1, height, width))
            shrunk_mask[:, top:bottom, left:right] = shrunk
            feathered_mask[:, top:bottom, left:right] = feathered
            shrunk_masks.append(shrunk_mask)
            feathered_masks.append(feathered_mask)
            source = output[:, top:bottom, left:right, :]
            output[:, top:bottom, left:right, :] = source * (1 - alpha.unsqueeze(-1)) + patch * alpha.unsqueeze(-1)
            blend_mask = output.new_zeros((1, height, width))
            blend_mask[:, top:bottom, left:right] = alpha
            images.append(output)
            masks.append(blend_mask)
        return torch.cat(images), torch.cat(masks), torch.cat(shrunk_masks), torch.cat(feathered_masks)
