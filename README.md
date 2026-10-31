# ComfyUI-ZLA-CropPaste

ComfyUI 双蒙版裁剪、缩放与贴回节点，支持手动比例、蒙版羽化和双图层 PSD 导出。

## 安装

在 ComfyUI 的 `custom_nodes` 目录运行：

```sh
git clone https://github.com/luyc2000/ComfyUI-ZLA-CropPaste.git
```

使用运行 ComfyUI 的 Python 环境安装本目录的 `requirements.txt`，然后重启 ComfyUI 并刷新页面。

## 节点

以下节点位于 `zla/Crop`，重启 ComfyUI 后可搜索以下名称：

- `Crop by Mask and Resize + Info ZLA`：复制自 `Crop by Mask and Resize to Preset Size 2`，保留原有参数与八个输出，增加 `crop_info` 输出。
- `Paste Processed Crop ZLA`：将 AI 处理后的裁剪图缩放回原位置，并与原图融合；输出贴回结果，以及实际融合、内缩、羽化三个阶段的蒙版。

## 连接方法

1. Dual Mask 的图像、mask1、mask2 接到新裁剪节点的同名输入。
2. 新裁剪节点的 `image1` 输出送入 AI 处理流程。
3. 原始完整图像接贴回节点的 `original_image`。
4. AI 处理结果接 `processed_image`，裁剪节点的 `crop_info` 接贴回节点的 `crop_info`。

## 贴回参数

- `blend_region`：`mask1`（默认）、`mask2`、`union`（两个蒙版的并集）、`rectangle`（整个裁剪框）。例如 mask1 画修改区域、mask2 画上下文时，选择 `mask1`。
- `mask_shrink`：蒙版内缩量，单位为原图像素，默认 16；0 不内缩。小蒙版可从 0～2 开始调整，避免内缩后消失。
- `feather_radius`：独立控制高斯羽化半径，单位为原图像素，默认 16；0 关闭羽化，但仍执行设置的内缩。启用羽化时，原蒙版边缘及外部的贴回权重为 0。
- `opacity`：贴回强度，默认 1；0 保留原图。
- `interpolation`：处理图缩放回原位置时的插值方式。

先内缩再羽化；两者都为 0 时使用原始蒙版。将 `mask_shrink` 设为与 `feather_radius` 相同的值可恢复此前的计算方式。

AI 输出分辨率可以变化，但应保持裁剪图的内容布局。原图尺寸必须与裁剪时一致。支持逐张记录批量裁剪位置，也支持单张原图/裁剪信息对应多张 AI 结果。

无有效蒙版时，默认贴回整个可还原区域。`pad_white` 会去掉补白后贴回；`center_crop` 只贴回实际保留的区域，其余保留原图。

## crop_info 格式

使用自定义端口类型 `ZLA_CROP_INFO` 传递 Python 字典：

- `original_size`：原图 `(height, width)`。
- `target_size`：裁剪输出的 `(height, width)`。
- `regions`：每张图的记录，包含 `box=(left, top, right, bottom)`（右、下边界不包含）、`aspect_mode`、`has_mask`，以及原始裁剪分辨率的 `mask1` / `mask2` 副本。

该信息自动传给贴回节点，无需手动输入坐标。

## 贴回输出

`Paste Processed Crop ZLA` 保留以下输出，便于预览和排查融合问题：

- `image`：贴回图像。
- `blend_mask`：实际融合权重，包含可还原区域限制和 opacity。
- `shrunk_mask`：仅内缩后的蒙版，尚未羽化。
- `feathered_mask`：内缩并羽化后的蒙版，尚未乘 opacity 和可还原区域限制。

所有蒙版都按原图尺寸输出，并放在原图对应的裁剪位置。将两个阶段蒙版分别接 Mask Preview（或 Mask to Image → Preview Image），即可比较蒙版变化。

独立的 `(Mask Test)` 测试节点已删除；使用测试节点的旧工作流需要替换为正式节点，参数及四个输出的顺序相同。

## PSD 导出

在 `Paste Processed Crop ZLA` 中启用 `save_psd`，即可将原图和缩放对齐后的 processed image 保存为两个独立图层：底层 `Original Image`、顶层 `Processed Image`。不应用融合 mask、内缩、羽化或节点的 opacity，两个图层均为 100% 不透明度。顶层只在对齐的裁剪位置包含图像，范围外透明；源图自带的透明度会保留。

`psd_filename_prefix` 默认是 `ZLA_Paste/Aligned`，文件保存在 ComfyUI 的 `output/ZLA_Paste/Aligned_00001_.psd`，自动编号，批量处理时每张图生成一个文件。支持 output 目录内的子目录。

`psd_compression` 可选择 `none`（默认，不压缩）或 `rle`（RLE 无损压缩），同时控制图层和合成预览，不再提供 ZIP。文件为 8 位 RGB PSD；导出时按 0～1 范围转换为 8 位像素。无蒙版的 `pad_white` 会移除补白，`center_crop` 丢失的画面区域在顶层保持透明。现有四个图像/蒙版输出继续按原参数计算，与 PSD 导出独立。

PSD 导出使用 `psd-tools`（见 requirements.txt）；关闭导出时不需要该库。旧版工作流可继续使用，新增导出默认关闭。

## 手动裁剪比例和缩放

`Crop by Mask and Resize + Info ZLA` 新增 `size_mode`：默认 `preset`，继续使用原有 preset.txt 中的 gemini、qwen 等预设；选择 `manual` 时使用以下参数：

- `aspect_width`、`aspect_height`：裁剪比例的宽、高，例如 3、2 表示 3:2。双 mask 并集、四边 pad 和裁剪定位逻辑不变。
- `resize_mode`：`height`（指定高度，即“长”）、`width`（指定宽度）、`long_edge`（指定长边）、`short_edge`（指定短边）、`total_pixels`（指定总像素）、`none`（不缩放）。
- `resize_length`：前四种模式的目标边长，单位为像素。
- `resize_pixels_k`：总像素模式的目标像素量，1K=1024 像素。例如 1024K=1,048,576 像素。

`resize_alignment` 可选择 8（手动缩放默认）、16、32、64 或 none（不对齐）。先按比例计算尺寸，再分别向上对齐宽、高，因此比例、指定边长或总像素可能略有偏差；none 仅取整到整数像素。预设模式不应用此对齐设置。`none` 直接输出实际裁剪像素，不做插值。无有效 mask 时仍不裁剪：有缩放时沿用 aspect_mode 处理整图，`none` 则保留整图原始像素。

不缩放时，批次内各张图的裁剪尺寸必须一致；尺寸不同请逐张处理，或选择固定输出尺寸的缩放模式。裁剪信息会记录实际输出尺寸，继续支持贴回和 PSD 导出。
