# Paper build notes

Generate tables and copy figures after syncing Vast artifacts:

```bash
cd snake_wm_v2
python -m snake_wm_v2.export_paper_assets
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The paper is intentionally conservative: v2 results are framed as a reproducible Snake case study of world-model exploitation and grounding, not as a general claim about large-scale embodied RL.

To create a view-only W&B report link after final sync:

```bash
cd snake_wm_v2
python -m snake_wm_v2.publish_wandb_report --enable-share-link
```
