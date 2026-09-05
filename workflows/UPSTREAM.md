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

The same builder derives the only three workflows installed in the RunPod UI:
Quality, FirstBlockCache Safe, and selectable LightX2V Turbo 4/8-step 768p. All three include the
selectable creator LoRA, 2x tail, and toggleable CPU auto-mosaic. The Fast graph
uses `duckyshell/ComfyUI-MiniMaxH3-FirstBlockCache` commit
`725973c3bfd9de6dce249bc93dc5fe27f820df31`. The Turbo graph follows the
ModelTC reference graph at commit `02e26d591f7a04d5d1a074c9566d5dd4f22f6225`
and uses the public LightX2V weight revision
`2f015e66b37c585cea9dc4ae6f1850ea8788e742`. Its local profile node switches
the pinned 8-step v1.0 and 4-step v1.2 768p LoRAs together with the scheduler
step count; both profiles retain the upstream 6/3 video/audio shifts.

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

The two ordered-story workflows are deterministically derived by
`scripts/build_story_workflows.py` from AIMixer's clean FL2V Director example at
commit `a267324a9f88141ff4e4b0e8c1a6ed90b4e45db7`:
https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/blob/a267324a9f88141ff4e4b0e8c1a6ed90b4e45db7/example_workflows/minimax_h3_director_fl2v.json

The derived graphs replace the Director card editor with the local variable-length
Ordered Storyboard input, apply the selectable V2 LoRA (and EasyCache only in the
Fast variant), force per-segment export, enable Director's integrated Motion Context
at the recommended 22-frame window, and stream each segment through Real-ESRGAN 2x
into a final ffmpeg-concatenated MP4. The container installs the Director commit
separately and applies the documented memory patch in
`patches/minimax_h3_director_segments_no_concat.patch` plus the narrow FL2V
last-frame retiming fix in `patches/minimax_h3_director_fl2v_motion_context.patch`.
