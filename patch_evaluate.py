from pathlib import Path

f = Path('src/evaluation/evaluate.py')
c = f.read_text(encoding='utf-8')

# Fix checkpoint loading
old = 'epoch, val_acc, _ = load_checkpoint(model, CHECKPOINT, device)'
new = (
    'ckpt = torch.load(CHECKPOINT, map_location=device)\n'
    '    model.load_state_dict(ckpt["model_state_dict"])\n'
    '    epoch   = ckpt.get("epoch", 0)\n'
    '    val_acc = ckpt.get("val_acc", 0.0)'
)
c = c.replace(old, new)

# Fix model - unfreeze for inference
c = c.replace(
    'model = model.to(device)',
    'model.unfreeze_backbone()\n    model = model.to(device)'
)

# Remove unused import
c = c.replace('from models.model_factory import get_model, get_device, load_checkpoint',
              'from models.model_factory import get_model, get_device')

f.write_text(c, encoding='utf-8')
print('evaluate.py patched successfully.')