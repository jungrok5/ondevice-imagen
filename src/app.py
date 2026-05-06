"""Gradio UI for the weekly-postcard prototype.

Run with:
    py -m src.app                 # local only, http://127.0.0.1:7860
    py -m src.app --share         # also exposes a public *.gradio.live URL
                                  # (anyone with the link can use it)

It hosts a local web UI where you can:
  1. Load the sample weekly diary (or paste your own JSON)
  2. See the auto-built prompt
  3. Render the deterministic base collage
  4. Run img2img to redraw it in the "bad doodle" style
  5. Or run pure txt2img with the same prompt for comparison
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import gradio as gr

from src import diary, prompt_builder, base_collage, generator


ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = ROOT / "data" / "sample_week.json"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def _load_default_text() -> str:
    return SAMPLE_PATH.read_text(encoding="utf-8")


def _save(img, tag: str) -> str:
    fname = f"{tag}-{int(time.time())}.png"
    path = OUT_DIR / fname
    img.save(path)
    return str(path)


def step_summarize(week_json_text: str):
    week = json.loads(week_json_text)
    summary = diary.summarize(week)
    built = prompt_builder.build(summary)
    base = base_collage.render(summary)
    return (
        diary.summary_to_text(summary),
        built.positive,
        built.negative,
        base,
        json.dumps(
            {"subjects": built.subjects, "debug": built.debug_summary},
            ensure_ascii=False,
            indent=2,
        ),
    )


def step_txt2img(prompt: str, negative: str, steps: int, guidance: float, seed: int):
    img = generator.generate_txt2img(
        prompt=prompt,
        negative_prompt=negative,
        steps=int(steps),
        guidance=float(guidance),
        seed=int(seed) if seed >= 0 else None,
    )
    path = _save(img, "txt2img")
    return img, f"saved: {path}"


def step_img2img(
    prompt: str,
    negative: str,
    init_image,
    steps: int,
    guidance: float,
    strength: float,
    seed: int,
):
    if init_image is None:
        raise gr.Error("Run the summarize step first to produce a base image.")
    img = generator.generate_img2img(
        prompt=prompt,
        init_image=init_image,
        negative_prompt=negative,
        steps=int(steps),
        guidance=float(guidance),
        strength=float(strength),
        seed=int(seed) if seed >= 0 else None,
    )
    path = _save(img, "img2img")
    return img, f"saved: {path}"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Weekly Postcard Doodle — Feasibility Test") as demo:
        gr.Markdown(
            "# Weekly Postcard Doodle\n"
            "PC feasibility prototype for an Android/iOS on-device app.\n\n"
            "1) **Summarize** — derive a prompt + base collage from a week of diary data.\n"
            "2) **img2img** — redraw the collage in the bad-doodle style (recommended).\n"
            "3) **txt2img** — prompt-only generation for comparison."
        )

        with gr.Row():
            with gr.Column(scale=1):
                week_json = gr.Code(
                    value=_load_default_text(),
                    language="json",
                    label="Weekly diary JSON",
                    lines=22,
                )
                btn_summarize = gr.Button("1) Summarize and build prompt", variant="primary")
                summary_text = gr.Textbox(label="Summary", interactive=False, lines=2)
                debug_json = gr.Code(label="Built prompt debug", language="json", interactive=False)
            with gr.Column(scale=1):
                positive = gr.Textbox(label="Prompt (editable)", lines=4)
                negative = gr.Textbox(label="Negative prompt (editable)", lines=2)
                base_img = gr.Image(label="Base collage (img2img seed)", type="pil")

        with gr.Accordion("Generation settings", open=False):
            steps = gr.Slider(1, 8, value=2, step=1, label="Inference steps (SD-Turbo: 1-4)")
            guidance = gr.Slider(0.0, 7.5, value=0.0, step=0.5, label="Guidance scale (SD-Turbo: 0)")
            strength = gr.Slider(0.3, 1.0, value=0.85, step=0.05, label="img2img strength")
            seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")

        with gr.Row():
            btn_img2img = gr.Button("2) Run img2img (recommended)", variant="primary")
            btn_txt2img = gr.Button("3) Run txt2img")

        with gr.Row():
            out_img2img = gr.Image(label="img2img result", type="pil")
            out_txt2img = gr.Image(label="txt2img result", type="pil")

        status = gr.Textbox(label="Status", interactive=False)

        btn_summarize.click(
            step_summarize,
            inputs=[week_json],
            outputs=[summary_text, positive, negative, base_img, debug_json],
        )
        btn_img2img.click(
            step_img2img,
            inputs=[positive, negative, base_img, steps, guidance, strength, seed],
            outputs=[out_img2img, status],
        )
        btn_txt2img.click(
            step_txt2img,
            inputs=[positive, negative, steps, guidance, seed],
            outputs=[out_txt2img, status],
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Weekly Postcard Doodle UI")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public *.gradio.live tunnel (link expires in ~72h).",
    )
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7860")))
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address. Use 0.0.0.0 to accept LAN connections.",
    )
    args = parser.parse_args()

    ui = build_ui()
    ui.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
