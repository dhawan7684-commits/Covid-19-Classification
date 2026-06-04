# Duplicate Detection Report

## Objective

To identify duplicate images within individual dataset splits.

## Results

Train Duplicate Groups: 27

Validation Duplicate Groups: 0

Test Duplicate Groups: 0

Total Duplicate Groups: 27

## Validation

Manual visual inspection was performed on detected duplicate groups.

The inspected image pairs were confirmed to be true duplicates (identical images).

## Observations

* Duplicate images exist only in the training split.
* No duplicates were detected in validation or test splits.
* Most duplicate groups were found in the COVID class.
* Several groups consisted of exact copies of the same image.

## Recommendation

Remove duplicate copies while retaining one representative image from each duplicate group.

## Conclusion

The dataset contains a small number of true duplicate images in the training set. These duplicates should be removed prior to model training to improve dataset quality and experimental rigor.
