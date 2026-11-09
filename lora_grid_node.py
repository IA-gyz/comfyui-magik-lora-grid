"""Magik Lora Grid (ComfyUI V3 node)

Generates one image (or one batch) per LoRA strength, everything else identical
(same seed, prompts, latent, sampler settings). Outputs:
  - grid          : IMAGE batch, ordered by strength (batch_size images per strength)
  - contact_sheet : single IMAGE, all results stitched together with strength labels
  - strengths     : STRING, one line per image in `grid`, aligned with the batch
"""

import math

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import comfy.model_management
import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths
import nodes
from comfy_api.latest import io

# Hard cap on total images per run (strengths x batch_size) to protect VRAM/RAM/time.
MAX_TOTAL_IMAGES = 64


# ----------------------------------------------------------------------------
# Pure helpers (no ComfyUI runtime needed)
# ----------------------------------------------------------------------------

def build_strengths(start, end, step, max_count):
    """Return [start, start+step, ...] while not exceeding `end`.

    - `step` is a magnitude (> 0); direction is inferred from start/end, so
      end < start produces a descending sequence.
    - A small epsilon prevents float drift from dropping the last value
      (e.g. 0.0 -> 1.0 at 0.1 gives 11 values).
    """
    if step <= 0:
        raise ValueError("Magik Lora Grid: strength_step must be greater than 0.")
    direction = 1.0 if end >= start else -1.0
    count = int(math.floor(abs(end - start) / step + 1e-6)) + 1
    if count > max_count:
        raise ValueError(
            f"Magik Lora Grid: {count} strengths requested but the limit is {max_count} "
            f"(max {MAX_TOTAL_IMAGES} images total = strengths x latent batch size). "
            f"Increase strength_step or narrow the range."
        )
    return [round(start + direction * i * step, 6) for i in range(count)]


def format_strength(value):
    v = round(value, 4) or 0.0  # normalizes -0.0 to 0.0
    return f"{v:g}"


def _load_font(size):
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _to_pil(img):
    arr = (img[..., :3].clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def build_contact_sheet(images, labels, columns):
    """Stitch a [N,H,W,C] batch into one [1,H',W',3] image with a label above each cell."""
    n, h, w, _ = images.shape
    columns = max(1, min(int(columns), n))
    rows = math.ceil(n / columns)

    font_size = max(14, w // 20)
    label_h = int(font_size * 1.8)
    pad = max(2, w // 128)
    cell_w, cell_h = w + pad, h + label_h + pad

    sheet = Image.new("RGB", (columns * cell_w + pad, rows * cell_h + pad), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    font = _load_font(font_size)

    for i in range(n):
        r, c = divmod(i, columns)
        x = pad + c * cell_w
        y = pad + r * cell_h
        bbox = draw.textbbox((0, 0), labels[i], font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            (x + (w - tw) // 2 - bbox[0], y + (label_h - th) // 2 - bbox[1]),
            labels[i],
            fill=(235, 235, 235),
            font=font,
        )
        sheet.paste(_to_pil(images[i]), (x, y + label_h))

    arr = np.asarray(sheet).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


# ----------------------------------------------------------------------------
# Node
# ----------------------------------------------------------------------------

class MagikLoraGrid(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MagikLoraGrid",
            display_name="Magik Lora Grid",
            category="Magik/LoRA",
            description=(
                "Generates images with identical settings and seed, varying only the "
                "LoRA strength from starting_strength up to ending_strength in "
                "increments of strength_step."
            ),
            inputs=[
                io.Model.Input("model", tooltip="Diffusion model."),
                io.Clip.Input("clip", tooltip="CLIP encoder."),
                io.Vae.Input("vae", tooltip="VAE used to decode each result."),
                io.Latent.Input(
                    "latent",
                    tooltip="Starting latent. Its batch_size images are generated for every strength.",
                ),
                io.Combo.Input(
                    "lora",
                    options=folder_paths.get_filename_list("loras"),
                    tooltip="LoRA to sweep.",
                ),
                io.Float.Input(
                    "starting_strength", default=0.0, min=-10.0, max=10.0, step=0.05,
                    tooltip="LoRA strength for the first image(s).",
                ),
                io.Float.Input(
                    "ending_strength", default=1.0, min=-10.0, max=10.0, step=0.05,
                    tooltip="Last strength to generate (not exceeded). If lower than starting_strength the sweep descends.",
                ),
                io.Float.Input(
                    "strength_step", default=0.25, min=0.01, max=10.0, step=0.01,
                    tooltip="Increment between consecutive strengths (always positive).",
                ),
                io.String.Input("positive", multiline=True, dynamic_prompts=True),
                io.String.Input("negative", multiline=True, dynamic_prompts=True),
                io.Int.Input(
                    "seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="Same seed for every strength so the LoRA is the only variable.",
                ),
                io.Int.Input("steps", default=20, min=1, max=10000, tooltip="Sampling steps."),
                io.Float.Input("cfg", default=7.0, min=0.0, max=100.0, step=0.1, round=0.01),
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default="euler"),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="normal"),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Int.Input(
                    "sheet_columns", default=0, min=0, max=32,
                    tooltip="Contact sheet columns. 0 = auto (one row per strength if batch_size > 1, otherwise near-square).",
                ),
            ],
            outputs=[
                io.Image.Output("grid"),
                io.Image.Output("contact_sheet"),
                io.String.Output("strengths"),
            ],
        )

    @classmethod
    def execute(cls, model, clip, vae, latent, lora, starting_strength, ending_strength,
                strength_step, positive, negative, seed, steps, cfg, sampler_name,
                scheduler, denoise, sheet_columns):
        batch_size = int(latent["samples"].shape[0])
        max_strengths = MAX_TOTAL_IMAGES // batch_size
        if max_strengths < 1:
            raise ValueError(
                f"Magik Lora Grid: latent batch_size {batch_size} exceeds the "
                f"{MAX_TOTAL_IMAGES}-image limit."
            )
        strengths = build_strengths(starting_strength, ending_strength, strength_step, max_strengths)

        lora_path = folder_paths.get_full_path_or_raise("loras", lora)
        lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)

        chunks, labels, lines = [], [], []
        per_strength = 1
        for s in strengths:
            comfy.model_management.throw_exception_if_processing_interrupted()

            # Same value for model and CLIP; clones, so the input model/clip stay untouched.
            m, c = comfy.sd.load_lora_for_models(model, clip, lora_data, s, s)
            pos = c.encode_from_tokens_scheduled(c.tokenize(positive))
            neg = c.encode_from_tokens_scheduled(c.tokenize(negative))

            out_latent = nodes.common_ksampler(
                m, seed, steps, cfg, sampler_name, scheduler, pos, neg, latent, denoise=denoise
            )[0]
            imgs = vae.decode(out_latent["samples"])
            if len(imgs.shape) == 5:  # video-style output -> flatten to an image batch
                imgs = imgs.reshape(-1, imgs.shape[-3], imgs.shape[-2], imgs.shape[-1])

            per_strength = imgs.shape[0]
            tag = format_strength(s)
            for j in range(per_strength):
                labels.append(tag if per_strength == 1 else f"{tag}  #{j + 1}")
                lines.append(tag)
            chunks.append(imgs)

        grid = torch.cat(chunks, dim=0)
        columns = sheet_columns or (
            per_strength if per_strength > 1 else math.ceil(math.sqrt(grid.shape[0]))
        )
        sheet = build_contact_sheet(grid, labels, columns)
        return io.NodeOutput(grid, sheet, "\n".join(lines))
