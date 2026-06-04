from pathlib import Path

f = Path('src/evaluation/evaluate.py')
c = f.read_text(encoding='utf-8')

c = c.replace(
    'loaders = build_dataloaders()',
    'loaders = build_dataloaders(dataset_root="COVID_19_dataset")'
)

f.write_text(c, encoding='utf-8')
print('Patched build_dataloaders call.')