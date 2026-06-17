# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from functools import lru_cache

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class Resolution:
    def __init__(self, size, *args):
        if isinstance(size, str):
            if "x" in size:
                size = size.split("x")
                size = (int(size[0]), int(size[1]))
            else:
                size = int(size)
        if len(args) > 0:
            size = (size, args[0])
        if isinstance(size, int):
            size = (size, size)

        self.h = self.height = size[0]
        self.w = self.width = size[1]
        self.r = self.ratio = self.height / self.width

        self.extra_res = set()

    def match(self, width, height) -> tuple[int, int]:
        if not self.extra_res:
            return self.w, self.h

        ret_w, ret_h = self.w, self.h
        target_area = width * height
        min_area_diff = abs((self.w * self.h) - target_area)
        for res_w, res_h in self.extra_res:
            area_diff = abs((res_w * res_h) - target_area)
            if area_diff < min_area_diff:
                min_area_diff = area_diff
                ret_w, ret_h = res_w, res_h
        return ret_w, ret_h

    def __getitem__(self, idx):
        if idx == 0:
            return self.h
        elif idx == 1:
            return self.w
        else:
            raise IndexError(f"Index {idx} out of range")

    def __str__(self):
        return f"{self.h}x{self.w}"

    def __repr__(self) -> str:
        if not self.extra_res:
            return "{" + f"{self.h}x{self.w}" + "}"

        ret_str = "{" + f"[{self.h}x{self.w}]"
        for res_w, res_h in self.extra_res:
            ret_str = ret_str + f"[{res_w}x{res_h}]"
        ret_str = ret_str + "}"
        return ret_str

    def append(self, res: "Resolution"):
        self.extra_res.add((res.w, res.h))


# Baked-in extras matching the official model's
# `HunyuanImage3ImageProcessor.vae_reso_group` (image_processor.py:147-152).
# These aspect buckets are trained model vocabulary targets, so AR and DiT
# consumers must share the same table.
HUNYUAN_IMAGE3_EXTRA_RESOLUTIONS: tuple[str, ...] = (
    "1024x768",
    "1280x720",
    "768x1024",
    "720x1280",
    "512x512",
    "640x640",
    "768x768",
    "896x896",
)


class ResolutionGroup:
    def __init__(self, base_size=None, step=None, align=1, extra_resolutions=None):
        self.align = align
        self.base_size = base_size
        assert base_size % align == 0, f"base_size {base_size} is not divisible by align {align}"
        if base_size is not None and not isinstance(base_size, int):
            raise ValueError(f"base_size must be None or int, but got {type(base_size)}")
        if step is None:
            step = base_size // 16
        if step is not None and step > base_size // 2:
            raise ValueError(f"step must be smaller than base_size // 2, but got {step} > {base_size // 2}")

        self.step = step
        self.data = self._calc_by_step()

        if extra_resolutions is not None:
            for er in extra_resolutions:
                for r in self.data:
                    if r.ratio == er.ratio:
                        r.append(er)
                        break
                else:
                    self.data.append(er)

        self.ratio = np.array([x.ratio for x in self.data])
        self.attr = ["" for _ in range(len(self.data))]
        self.prefix_space = 0
        logger.debug("ResolutionGroup: %s", self)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __repr__(self):
        prefix = self.prefix_space * " "
        prefix_close = (self.prefix_space - 4) * " "
        res_str = f"ResolutionGroup(base_size={self.base_size}, step={self.step}, data="
        attr_maxlen = max([len(x) for x in self.attr] + [5])
        res_str += (
            f"\n{prefix}ID: height width   ratio {' ' * max(0, attr_maxlen - 4)}count  h/16 w/16    tokens\n{prefix}"
        )

        rows = []
        for i, x in enumerate(self.data):
            main_row = (
                f"{i:2d}: ({x.h:4d}, {x.w:4d})  {self.ratio[i]:.4f}  {self.attr[i]:>{attr_maxlen}s}  "
                f"({x.h // 16:3d}, {x.w // 16:3d})  {x.h // 16 * x.w // 16:6d}"
            )
            rows.append(main_row)
            extra_val = getattr(x, "extra_res", None)
            if extra_val:
                for sub_h, sub_w in sorted(list(extra_val)):
                    sub_ratio = sub_h / sub_w
                    sub_h16, sub_w16 = sub_h // 16, sub_w // 16
                    sub_tokens = sub_h16 * sub_w16
                    sub_row = (
                        f"    ({sub_h:4d}, {sub_w:4d})  {sub_ratio:.4f}  {' ' * attr_maxlen}  "
                        f"({sub_h16:3d}, {sub_w16:3d})  {sub_tokens:6d}"
                    )
                    rows.append(sub_row)

        res_str += ("\n" + prefix).join(rows)
        res_str += f"\n{prefix_close})"
        return res_str

    def _calc_by_step(self):
        assert self.align <= self.step, f"align {self.align} must be smaller than step {self.step}"

        min_height = self.base_size // 2
        min_width = self.base_size // 2
        max_height = self.base_size * 2
        max_width = self.base_size * 2

        resolutions = [Resolution(self.base_size, self.base_size)]

        cur_height, cur_width = self.base_size, self.base_size
        while True:
            if cur_height >= max_height and cur_width <= min_width:
                break

            cur_height = min(cur_height + self.step, max_height)
            cur_width = max(cur_width - self.step, min_width)
            resolutions.append(Resolution(cur_height // self.align * self.align, cur_width // self.align * self.align))

        cur_height, cur_width = self.base_size, self.base_size
        while True:
            if cur_height <= min_height and cur_width >= max_width:
                break

            cur_height = max(cur_height - self.step, min_height)
            cur_width = min(cur_width + self.step, max_width)
            resolutions.append(Resolution(cur_height // self.align * self.align, cur_width // self.align * self.align))

        resolutions = sorted(resolutions, key=lambda x: x.ratio)

        return resolutions

    def get_target_size(self, width, height):
        ratio = height / width
        idx = np.argmin(np.abs(self.ratio - ratio))
        w, h = self.data[idx].match(width, height)
        return w, h

    def get_base_size_and_ratio_index(self, width, height):
        ratio = height / width
        idx = np.argmin(np.abs(self.ratio - ratio))
        return self.base_size, idx


@lru_cache(maxsize=4)
def get_cached_resolution_group(base_size: int) -> ResolutionGroup:
    extra_res_tuple = tuple(Resolution(s) for s in HUNYUAN_IMAGE3_EXTRA_RESOLUTIONS)
    extra_resolutions = list(extra_res_tuple) if extra_res_tuple else None
    return ResolutionGroup(base_size=base_size, extra_resolutions=extra_resolutions)


def resize_and_crop(
    image: Image.Image,
    target_size: tuple[int, int],
    crop_type: str = "center",
) -> Image.Image:
    tw, th = target_size
    if crop_type == "resize":
        return image.resize((tw, th), resample=Image.Resampling.LANCZOS)

    w, h = image.size
    tr = th / tw
    r = h / w
    if r < tr:
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))
    image = image.resize((resize_width, resize_height), resample=Image.Resampling.LANCZOS)
    crop_top = int(round((resize_height - th) / 2.0))
    crop_left = int(round((resize_width - tw) / 2.0))
    return image.crop((crop_left, crop_top, crop_left + tw, crop_top + th))
