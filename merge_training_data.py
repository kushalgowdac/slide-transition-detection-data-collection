"""Merge toc_1 training data with existing dataset"""

import pandas as pd
from pathlib import Path

print("\n" + "=" * 80)
print("MERGING TRAINING DATA")
print("=" * 80 + "\n")

# Load both datasets
existing = pd.read_csv('labeled_dataset.csv')
new = pd.read_csv('toc_1_training_data.csv')

print(f'Existing dataset: {len(existing):,} rows')
print(f'  - Transitions: {existing["is_transition_gt"].sum()}')
print(f'  - Non-transitions: {(~existing["is_transition_gt"]).sum()}')
print()

print(f'New data (toc_1): {len(new):,} rows')
print(f'  - Transitions: {new["is_transition_gt"].sum()}')
print(f'  - Non-transitions: {(~new["is_transition_gt"]).sum()}')
print()

# Combine
combined = pd.concat([existing, new], ignore_index=True)

# Backup original (remove old backup if present)
backup_path = Path('labeled_dataset_backup.csv')
if backup_path.exists():
	backup_path.unlink()
	print('✅ Old backup removed')

Path('labeled_dataset.csv').rename(backup_path)
print('✅ Backup saved to labeled_dataset_backup.csv')

# Save combined
combined.to_csv('labeled_dataset.csv', index=False)

print()
total_trans = (combined["is_transition_gt"] == 1).sum()
print(f'✅ Combined dataset: {len(combined):,} rows')
print(f'  - Transitions: {total_trans}')
print(f'  - Non-transitions: {len(combined) - total_trans}')
print(f'  - Increase: +{len(new)} rows (toc_1 data)')
print('\nReady to retrain model!\n')
