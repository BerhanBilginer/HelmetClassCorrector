import random

# Small detections benefit from surrounding head/body context before resizing.
DEFAULT_CONTEXT_CROP_CONFIG = {
    'base_context_ratio': 0.15,
    'min_context_px': 4,
    'min_context_side': 64,
    'max_context_ratio': 2.0,
}


def sanitize_bbox(x1, y1, x2, y2, image_width, image_height):
    """Clamp a bbox to valid image coordinates."""
    x1 = max(0, min(int(round(x1)), image_width))
    y1 = max(0, min(int(round(y1)), image_height))
    x2 = max(0, min(int(round(x2)), image_width))
    y2 = max(0, min(int(round(y2)), image_height))
    return x1, y1, x2, y2


def expand_bbox_with_context(
    x1,
    y1,
    x2,
    y2,
    image_width,
    image_height,
    base_context_ratio=0.15,
    min_context_px=4,
    min_context_side=64,
    max_context_ratio=2.0,
):
    """
    Expand small boxes more aggressively so tiny detections keep useful context.

    The crop is still bounded by the source image, but the requested padding grows
    when the shortest bbox side is below ``min_context_side``.
    """
    x1, y1, x2, y2 = sanitize_bbox(x1, y1, x2, y2, image_width, image_height)
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    short_side = min(box_w, box_h)

    target_extra = max(0.0, (min_context_side - short_side) / 2.0)
    pad_x = max(min_context_px, box_w * base_context_ratio, target_extra)
    pad_y = max(min_context_px, box_h * base_context_ratio, target_extra)

    pad_x = int(round(min(pad_x, box_w * max_context_ratio)))
    pad_y = int(round(min(pad_y, box_h * max_context_ratio)))

    return sanitize_bbox(
        x1 - pad_x,
        y1 - pad_y,
        x2 + pad_x,
        y2 + pad_y,
        image_width,
        image_height,
    )


def crop_with_context(image, bbox, **context_kwargs):
    """Return a dynamically expanded crop and the final bbox used."""
    image_height, image_width = image.shape[:2]
    expanded_bbox = expand_bbox_with_context(
        *bbox,
        image_width=image_width,
        image_height=image_height,
        **context_kwargs,
    )
    x1, y1, x2, y2 = expanded_bbox
    return image[y1:y2, x1:x2], expanded_bbox


class AspectRatioPadResize:
    """Resize while preserving aspect ratio, then pad to the requested canvas."""

    def __init__(
        self,
        size,
        fill=(0, 0, 0),
        interpolation=None,
        scale=1.0,
        scale_range=None,
    ):
        if isinstance(size, int):
            size = (size, size)
        self.size = (int(size[0]), int(size[1]))
        self.fill = fill
        self.interpolation = interpolation
        self.scale = scale
        self.scale_range = scale_range

    def _sample_scale(self):
        if self.scale_range is None:
            return self.scale
        low, high = self.scale_range
        return random.uniform(low, high)

    def __call__(self, image):
        import numpy as np
        from PIL import Image, ImageOps

        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))

        target_h, target_w = self.size
        fit_scale = min(target_w / image.width, target_h / image.height)
        scale = min(1.0, max(0.1, self._sample_scale()))
        resized_scale = fit_scale * scale

        new_w = max(1, min(target_w, int(round(image.width * resized_scale))))
        new_h = max(1, min(target_h, int(round(image.height * resized_scale))))
        interpolation = self.interpolation or Image.BILINEAR
        resized = image.resize((new_w, new_h), interpolation)

        pad_w = target_w - new_w
        pad_h = target_h - new_h
        padding = (
            pad_w // 2,
            pad_h // 2,
            pad_w - pad_w // 2,
            pad_h - pad_h // 2,
        )
        return ImageOps.expand(resized, border=padding, fill=self.fill)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(size={self.size}, fill={self.fill}, "
            f"scale={self.scale}, scale_range={self.scale_range})"
        )
