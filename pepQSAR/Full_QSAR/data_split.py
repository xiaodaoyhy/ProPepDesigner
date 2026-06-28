import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple


def random_split_indices(
    n_samples: int,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomly split sample indices into train / val / test.

    Parameters
    ----------
    n_samples : int
        Total number of samples
    val_ratio : float
        Validation set fraction of all data
    test_ratio : float
        Test set fraction of all data
    seed : int
        Random seed
    """
    assert 0 < val_ratio < 1 and 0 < test_ratio < 1 and val_ratio + test_ratio < 1

    indices = np.arange(n_samples)

    # First split train vs. (val + test)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=val_ratio + test_ratio,
        random_state=seed,
        shuffle=True,
        stratify=None,
    )

    # Then split (val + test) into val / test
    rel_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=rel_test_ratio,
        random_state=seed,
        shuffle=True,
        stratify=None,
    )

    return train_idx, val_idx, test_idx






def stratify_split_indices(
    y: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
    n_bins: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple multi-label stratified split based on activity values:
    - For each task, bin activity by quantiles into n_bins (NaN ignored)
    - Combine per-sample bins across tasks into a "label pattern" for stratification

    Note: This is an approximate multi-label stratification, used only to keep
          train/val/test similar in activity distribution patterns.

    Parameters
    ----------
    y : np.ndarray, shape [n_samples, n_tasks], may contain NaN
    val_ratio, test_ratio, seed, n_bins : same as above
    """
    assert 0 < val_ratio < 1 and 0 < test_ratio < 1 and val_ratio + test_ratio < 1
    n_samples, n_tasks = y.shape

    # 1. Bin each task; NaN marked as -1
    bins_per_task = np.full_like(y, fill_value=-1, dtype=int)

    for j in range(n_tasks):
        col = y[:, j]
        mask = ~np.isnan(col)
        if mask.sum() == 0:
            # No labels at all for this task
            continue

        # Compute quantile boundaries
        qs = np.linspace(0, 100, n_bins + 1)
        try:
            quantiles = np.nanpercentile(col[mask], qs)
        except Exception:
            # Skip binning if quantile computation fails
            continue

        # Slightly perturb boundaries to avoid duplicates
        quantiles[0] -= 1e-6
        quantiles[-1] += 1e-6

        # Assign bin indices
        # np.digitize returns 1..n_bins; subtract 1 to get 0..n_bins-1
        bins = np.digitize(col[mask], quantiles[1:-1])  # use inner boundaries
        bins_per_task[mask, j] = bins

    # 2. Combine per-sample bins into string labels for stratification
    labels = np.array(
        ["|".join(map(str, row)) for row in bins_per_task],
        dtype=object,
    )

    # 3. Handle singleton "rare labels": put directly in train, skip stratification
    indices = np.arange(n_samples)
    _, counts = np.unique(labels, return_counts=True)
    label_to_count = {lab: cnt for lab, cnt in zip(np.unique(labels), counts)}
    is_rare = np.array([label_to_count[lab] < 2 for lab in labels])

    rare_idx = indices[is_rare]
    strat_indices = indices[~is_rare]
    strat_labels = labels[~is_rare]

    # 4. Stratified split on remaining samples: train / (val + test)
    if strat_indices.size > 0:
        unique_labels = np.unique(strat_labels)
        # Fall back to non-stratified split if too few classes or too few samples per class
        if unique_labels.size > 1:
            _, rem_counts = np.unique(strat_labels, return_counts=True)
            if rem_counts.min() >= 2:
                stratify_labels = strat_labels
            else:
                stratify_labels = None
        else:
            stratify_labels = None

        train_idx_rem, temp_idx = train_test_split(
            strat_indices,
            test_size=val_ratio + test_ratio,
            random_state=seed,
            shuffle=True,
            stratify=stratify_labels,
        )
        # Final train set = stratified train + all rare samples
        train_idx = np.concatenate([rare_idx, train_idx_rem]) if rare_idx.size > 0 else train_idx_rem
    else:
        # All samples are rare labels; fall back to random split only
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=val_ratio + test_ratio,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )

    if len(temp_idx) == 0:
        return train_idx, np.array([], dtype=int), np.array([], dtype=int)

    # 5. Split temp (val + test) into val / test
    temp_labels = labels[temp_idx]
    unique_temp_labels, temp_counts = np.unique(temp_labels, return_counts=True)
    # Stratify only when num_classes > 1 and each class has >= 2 samples; else non-stratified
    if unique_temp_labels.size > 1 and temp_counts.min() >= 2:
        stratify_temp = temp_labels
    else:
        stratify_temp = None

    rel_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=rel_test_ratio,
        random_state=seed,
        shuffle=True,
        stratify=stratify_temp,
    )

    return train_idx, val_idx, test_idx

