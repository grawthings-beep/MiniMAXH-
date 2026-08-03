# Third-party notices

This repository does not redistribute MiniMax H3 weights.

- MiniMax H3 weights are downloaded from `Comfy-Org/MiniMax-H3` and remain
  subject to the
  [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE).
  The license's required redistribution notice is also provided in `NOTICE`.
- ComfyUI is licensed under GPL-3.0. The container clones the unmodified
  `Comfy-Org/ComfyUI` v0.30.0 release at build time.
- The I2V workflow and the upstream source for the derived R2V/EasyCache
  workflows come from `Comfy-Org/workflow_templates`; exact source revisions
  and local derivation are documented in `workflows/UPSTREAM.md`.
- The H3 text encoder is derived from Qwen3-VL-32B, licensed under Apache-2.0.

Review all applicable licenses before building, using, or distributing a
container made from this repository.
