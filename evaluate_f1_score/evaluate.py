"""
Adapted from: https://www.kaggle.com/code/wht1996/feedback-nn-train
"""

# Results file following the dataset format: id, class, predictionstring
# Note: Must be a subset of train.csv, a join is performed to eliminate unused essays in train.csv
RESULTS_FILE = "results_offset.csv"
# Ground truths file downloaded from Kaggle
GROUND_TRUTHS_FILE = "train.csv"

import numpy as np
import pandas as pd


def calc_overlap(row):
    """
    Given a row containing predictionstring_pred and predictionstring_gt,
    compute the overlap as the minimum of the ratios:
       overlap_pred = |set(pred) ∩ set(gt)| / |set(pred)|
       overlap_gt   = |set(pred) ∩ set(gt)| / |set(gt)|
    Here we return a single value: inter / max(|set(pred)|, |set(gt)|)
    which is a proxy for the minimum overlap.
    """
    set_pred = set(row.predictionstring_pred.split())
    set_gt = set(row.predictionstring_gt.split())
    if not set_pred or not set_gt:
        return 0
    inter = len(set_pred.intersection(set_gt))
    return inter / max(len(set_pred), len(set_gt))

def get_f1_score(test_pred, test_df, log=print, slient=True):
    """
    Evaluates F1 score per class and the macro-averaged F1.
    
    For each class, it:
      1. Filters predictions (test_pred) by that class (using column 'class')
         and ground truth by that class (using column 'discourse_type').
      2. Merges the two DataFrames on ['id'] (where id is the essay id) and on class.
      3. Computes the overlap between each prediction and ground truth pair.
      4. Marks a candidate as a potential true positive if its overlap is >= 0.5.
         (A prediction is matched to the ground truth row that maximizes the overlap.)
      5. Counts unmatched predictions as false positives and unmatched ground truths as false negatives.
      6. Computes F1 as: TP / (TP + 0.5*(FP+FN))
    
    test_pred: DataFrame from results.csv with columns [id, class, predictionstring]
    test_df: DataFrame from train.csv with columns [id, discourse_type, predictionstring, ...]
    """
    if not slient:
        SEP = "=" * 50
        log('test_pred.shape:', test_pred.shape, 'test_df.shape:', test_df.shape)
        log(SEP)
        log('pred class:\n', test_pred['class'].value_counts())
        log('total components:', test_pred['class'].value_counts().sum())
        log(SEP)
        log('true class:\n', test_df['discourse_type'].value_counts())
        log('total components:', test_df['discourse_type'].value_counts().sum())
        log(SEP)
    f1s = []

    for c in sorted(test_pred['class'].unique()):
    # for c in sorted(["claim", "lead", "concluding statement", "evidence", "position"]):
        # Filter predictions for class c.
        pred_df = test_pred.loc[test_pred['class'] == c].copy()
        # Filter ground truth for class c.
        gt_df = test_df.loc[test_df['discourse_type'] == c].copy()

        # Only retain necessary columns.
        gt_df = gt_df[['id', 'discourse_type', 'predictionstring']].reset_index(drop=True)
        pred_df = pred_df[['id', 'class', 'predictionstring']].reset_index(drop=True)
        pred_df['pred_id'] = pred_df.index
        gt_df['gt_id'] = gt_df.index
        
        # Merge on essay id and matching class labels.
        joined = pred_df.merge(gt_df,
                               left_on=['id', 'class'],
                               right_on=['id', 'discourse_type'],
                               how='outer',
                               suffixes=('_pred', '_gt'))
        joined['predictionstring_gt'] = joined['predictionstring_gt'].fillna(' ')
        joined['predictionstring_pred'] = joined['predictionstring_pred'].fillna(' ')

        joined['min_overlaps'] = joined.apply(calc_overlap, axis=1)
        joined['potential_TP'] = joined['min_overlaps'] >= 0.5

        # print(joined[joined['potential_TP'] == True][['id', 'predictionstring_pred', 'predictionstring_gt', 'min_overlaps', 'potential_TP']])

        # Unique pred_id and gt_id that achieved a potential match.
        matched_pred_ids = joined.query('potential_TP')['pred_id'].unique()
        fp_pred_ids = [p for p in joined['pred_id'].unique() if p not in matched_pred_ids]

        matched_gt_ids = joined.query('potential_TP')['gt_id'].unique()
        fn_gt_ids = [g for g in joined['gt_id'].unique() if g not in matched_gt_ids]

        TP = len(matched_gt_ids)
        FP = len(fp_pred_ids)
        FN = len(fn_gt_ids)
        # F1 formula: TP / (TP + 0.5*(FP+FN))
        f1_score = TP / (TP + 0.5 * (FP + FN)) if (TP + 0.5 * (FP + FN)) > 0 else 0
        if not slient:
            log(f'{c:<20} f1 score:\t{f1_score}')
        f1s.append(f1_score)
    overall_f1 = np.mean(f1s) if f1s else 0
    log('\nOverall f1 score \t', overall_f1)
    return overall_f1

if __name__ == '__main__':
    # Read predictions from results_real.csv.
    test_pred = pd.read_csv(RESULTS_FILE, quoting=1)  # quoting=1 corresponds to csv.QUOTE_ALL
    # Read ground truth from train.csv.
    test_df = pd.read_csv(GROUND_TRUTHS_FILE, quoting=1)
    
    # Remove any ".txt" extension from the id column.
    test_pred['id'] = test_pred['id'].apply(lambda x: x.rsplit('.', 1)[0] if isinstance(x, str) else x)
    test_df['id'] = test_df['id'].apply(lambda x: x.rsplit('.', 1)[0] if isinstance(x, str) else x)
    
    # Normalize class labels in predictions and ground truth.
    test_pred['class'] = test_pred['class'].str.strip().str.lower()
    test_df['discourse_type'] = test_df['discourse_type'].str.strip().str.lower()
    
    # Filter ground truth to only those ids that appear in predictions.
    valid_ids = set(test_pred['id'].unique())
    test_df = test_df[test_df['id'].isin(valid_ids)]
    
    get_f1_score(test_pred, test_df, slient=False)
