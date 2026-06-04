# Data Leakage Detection and Cleanup Report

## Objective

To identify and eliminate cross-split data leakage between the training, validation, and test sets before model training.

Data leakage occurs when identical images appear in multiple dataset splits, resulting in overly optimistic evaluation metrics and invalid experimental results.

---

## Methodology

A perceptual hashing (pHash) based comparison was performed across all dataset splits.

The following comparisons were evaluated:

* Train ↔ Validation
* Train ↔ Test
* Validation ↔ Test

Potential leakage cases were manually inspected to verify whether the images were truly identical.

---

## Results

### Train ↔ Validation Leakage

Leakage Cases Found: **4**

### Train ↔ Test Leakage

Leakage Cases Found: **5**

### Validation ↔ Test Leakage

Leakage Cases Found: **2**

### Total Leakage Cases

**11**

---

## Manual Verification

All identified leakage pairs were manually inspected.

Observations:

* Images were confirmed to be visually identical.
* Several leakage cases involved consecutively numbered filenames.
* Leakage was observed exclusively within the COVID class.
* The detected cases represent true cross-split duplicates rather than merely similar images.

---

## Cleanup Strategy

The following policy was adopted:

### Train ↔ Validation

* Retain the Training image.
* Remove the Validation image.

### Train ↔ Test

* Retain the Training image.
* Remove the Test image.

### Validation ↔ Test

* Retain the Validation image.
* Remove the Test image.

This approach maximizes available training data while ensuring that validation and test sets remain completely independent.

---

## Files Removed

### Validation Images Removed

* COVID-3375.png
* COVID-898.png
* COVID-1267.png
* COVID-1416.png

### Test Images Removed

* COVID-2716.png
* COVID-230.png
* COVID-114.png
* COVID-487.png
* COVID-315.png
* COVID-2678.png
* COVID-56.png

Total Removed: **11 Images**

---

## Impact on Dataset Quality

Before Cleanup:

* Train ↔ Validation Leakage: 4
* Train ↔ Test Leakage: 5
* Validation ↔ Test Leakage: 2
* Total Leakage Cases: 11

After Cleanup:

* Train ↔ Validation Leakage: 0
* Train ↔ Test Leakage: 0
* Validation ↔ Test Leakage: 0
* Total Leakage Cases: 0

---

## Importance for Model Evaluation

Removing leakage ensures that:

* Accuracy reflects true generalization.
* Precision reflects genuine predictive performance.
* Recall is not artificially inflated.
* F1-score remains trustworthy.
* Explainability results are evaluated on previously unseen images.

---

## Conclusion

A total of 11 cross-split leakage cases were identified and verified through manual inspection. All leakage instances were removed according to a predefined cleanup policy. The dataset now satisfies the requirement that training, validation, and test sets remain independent, ensuring reliable model evaluation and scientifically valid experimental results.

## Verification Checklist

* [x] Train ↔ Validation comparison completed
* [x] Train ↔ Test comparison completed
* [x] Validation ↔ Test comparison completed
* [x] Manual verification performed
* [x] Leakage images identified
* [x] Leakage images removed
* [x] Cleanup log generated
* [x] Dataset re-verified
* [x] Zero leakage cases remaining
