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
R2V workflow and both EasyCache Fast variants from these pinned upstream assets.
