# transfer-resnet

ResNet-18 transfer learning on a custom image dataset.
Two-phase: (1) train head only, (2) fine-tune all layers.
PyTorch 2.0 noticeably faster than 1.x, `torch.compile()` is wild.


Falls back to CIFAR-10 (auto-downloaded) if no `data/` folder exists.

## Run

```bash
pip install -r requirements.txt
python train.py
```

## Results (on dogs vs cats, 1000 imgs/class)

| Approach | Val Accuracy | Time |
|----------|-------------|------|
| Train from scratch (my CIFAR CNN) | ~72% | 30 min |
| Head-only, 5 epochs | 94.1% | 3 min |
| Full fine-tune, 10 epochs | **97.2%** | 12 min |

## Transfer learning notes

- `pretrained=True` downloads ImageNet weights (~45MB) once, cached in `~/.cache/torch`
- Must use **ImageNet normalization** (mean/std) — not doing this was my bug for 2 hours
- Phase 1 (freeze backbone): fast, good starting point
- Phase 2 (fine-tune all): use **very small LR** (1e-4) or you destroy the pretrained features
- `copy.deepcopy(model.state_dict())` to save best weights during training
