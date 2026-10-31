import { app } from "../../scripts/app.js";

const widgetOrder = [
    "size_mode", "preset_name", "aspect_width", "aspect_height",
    "resize_mode", "resize_length", "resize_pixels_k", "resize_alignment",
    "interpolation", "aspect_mode", "mask_threshold",
    "pad_left", "pad_right", "pad_top", "pad_bottom", "debug_log",
];

const previousOrder = [
    "preset_name", "interpolation", "aspect_mode", "mask_threshold",
    "pad_left", "pad_right", "pad_top", "pad_bottom", "debug_log",
    "size_mode", "aspect_width", "aspect_height", "resize_mode",
    "resize_length", "resize_pixels_k", "resize_alignment",
];

function arrangeWidgets(node) {
    node.widgets.sort((a, b) => {
        const aIndex = widgetOrder.indexOf(a.name);
        const bIndex = widgetOrder.indexOf(b.name);
        return (aIndex < 0 ? widgetOrder.length : aIndex) - (bIndex < 0 ? widgetOrder.length : bIndex);
    });
    node.properties.zla_crop_widget_order = 1;
}

app.registerExtension({
    name: "ZLA.CropWidgetOrder",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "CropByMask_PresetResizeInfo") return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onCreated?.apply(this, arguments);
            this._zlaCropWidgetDefaults = new Map(this.widgets.map((widget) => [widget.name, widget.value]));
            arrangeWidgets(this);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (data) {
            const result = onConfigure?.apply(this, arguments);
            if (!data.properties?.zla_crop_widget_order && Array.isArray(data.widgets_values)) {
                for (const widget of this.widgets) {
                    const index = previousOrder.indexOf(widget.name);
                    if (index < 0) continue;
                    widget.value = index < data.widgets_values.length
                        ? data.widgets_values[index] : this._zlaCropWidgetDefaults.get(widget.name);
                }
            }
            arrangeWidgets(this);
            return result;
        };
    },
});

function applyPairColor(node) {
    node.color = "#287044";
    node.bgcolor = "#193e29";
    node.properties.zla_crop_pair_color = 2;
}

app.registerExtension({
    name: "ZLA.CropPasteColors",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!["CropByMask_PresetResizeInfo", "PasteByMaskCropInfo"].includes(nodeData.name)) return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onCreated?.apply(this, arguments);
            applyPairColor(this);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (data) {
            const result = onConfigure?.apply(this, arguments);
            if (!data.properties?.zla_crop_pair_color || (
                data.properties.zla_crop_pair_color === 1
                && data.color === "#234b70" && data.bgcolor === "#182f46"
            )) applyPairColor(this);
            this.properties.zla_crop_pair_color = 2;
            return result;
        };
    },
});
