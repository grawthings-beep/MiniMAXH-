# Upstream workflow

`minimax_h3_i2v.json` is copied without modification from
`Comfy-Org/workflow_templates` commit
`cebdebc9fc2febcb97a5db0dd291f59f5300b176`.

Source:
https://github.com/Comfy-Org/workflow_templates/blob/cebdebc9fc2febcb97a5db0dd291f59f5300b176/templates/video_minimax_h3_i2v.json

`upstream_minimax_h3_r2v.json` is copied without modification from
`Comfy-Org/workflow_templates` commit
`5c75d9f137bb27706a70dd337dac6249b2e51ded`.

Source:
https://github.com/Comfy-Org/workflow_templates/blob/5c75d9f137bb27706a70dd337dac6249b2e51ded/templates/video_minimax_h3_r2v.json

`scripts/build_workflows.py` deterministically derives the mixed video-reference
R2V workflow, both EasyCache Fast variants, four Real-ESRGAN 2x post-process
variants, and both I2V Quality + selectable LoRA + 2x variants from these pinned
upstream assets.

The 2x variants use ComfyUI's built-in `UpscaleModelLoader` and
`ImageUpscaleWithModel` nodes with the official `RealESRGAN_x2plus.pth` release:
https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.1

The HMMotion variant uses ComfyUI's built-in `LoraLoaderModelOnly` node and
downloads `hmmotion_minimax-h3_epoch12.safetensors` from the following private
Hugging Face repository at Pod startup. The file is not redistributed here:
https://huggingface.co/uwgm/nikke-civitai-backup

The selectable V2 variant also uses `LoraLoaderModelOnly`, defaults to
`HMNSFW_AIO_V2.safetensors` at strength 0.5, and downloads the pinned Civitai
version/file at Pod startup. The file is not redistributed here:
https://civitai.com/models/2834417?modelVersionId=3206518
