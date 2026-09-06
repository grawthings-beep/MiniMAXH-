# Third-party notices

This repository does not redistribute MiniMax H3 weights.

- MiniMax H3 weights are downloaded from `Comfy-Org/MiniMax-H3` and remain
  subject to the
  [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE).
  The license's required redistribution notice is also provided in `NOTICE`.
- ComfyUI is licensed under GPL-3.0. The container clones the unmodified
  `Comfy-Org/ComfyUI` v0.31.0 release at build time.
- `duckyshell/ComfyUI-MiniMaxH3-FirstBlockCache` is installed at pinned commit
  `725973c3bfd9de6dce249bc93dc5fe27f820df31` under MIT. It adds no model or
  Python-package dependency.
- `AIMixer/ComfyUI_MiniMaxH3_Director` is licensed under Apache-2.0. The
  container installs commit `a267324a9f88141ff4e4b0e8c1a6ed90b4e45db7`
  as a separate custom-node checkout. It applies the repository patch
  `patches/minimax_h3_director_segments_no_concat.patch`, which prevents a
  redundant full-timeline tensor allocation when Director is explicitly in
  per-segment export mode, and
  `patches/minimax_h3_director_fl2v_motion_context.patch`, which retimes the
  retained FL2V last-frame anchor for Director's integrated Motion Context.
  The upstream license remains in that checkout. Director credits
  `NikoDemon80/ComfyUI-H3-Motion-Context` for the continuation approach; the
  standalone GPL package is not installed or redistributed here.
- The I2V workflow and the upstream source for the derived R2V/EasyCache
  workflows come from `Comfy-Org/workflow_templates`; exact source revisions
  and local derivation are documented in `workflows/UPSTREAM.md`.
- The H3 text encoder is derived from Qwen3-VL-32B, licensed under Apache-2.0.
- The public LightX2V MiniMax H3 Turbo 8-step v1.0 and 4-step v1.2 768p LoRAs
  are downloaded at Pod startup from pinned Hugging Face revision
  `2f015e66b37c585cea9dc4ae6f1850ea8788e742` and are published under
  Apache-2.0.
- `ultralytics==8.4.104` is installed for CPU-only YOLO11 instance-segmentation
  inference. The separately downloaded auto-mosaic model is not redistributed;
  its current Civitai terms remain the operator's responsibility.
- The auto-mosaic contour and circular-gap implementation is adapted from
  `grawthings-beep/wan-animate-runpod` commit
  `01a73bc628cc19a1df92684349285f03d4a1f39a`.

Review all applicable licenses before building, using, or distributing a
container made from this repository.
