# DATA WARS 2026: Predicting 30-Day Hospital Readmission

We worked with hospital records of diabetic patients. We cleaned the data, explored it with graphs, and built models that predict **which patients come back to hospital within 30 days of leaving**.

Every number in this README comes from our own notebook outputs.

---

## 1. What we were given

| File | What it is |
|---|---|
| `diabetic_data.csv` | The original dataset: **101,766 rows** (one row = one hospital visit, called an *encounter*) and **50 columns** |
| `IDs_mapping.csv` | A lookup table that explains the numeric codes (admission type, discharge destination, admission source) |
| Data dictionary (image) | A description of every column |

**What each row contains**

- **Patient information:** race, gender, age group, weight
- **The hospital stay:** admission type, where the patient came from, where they went after, days in hospital, doctor's specialty, insurance (payer) code
- **What happened during the stay:** number of lab tests, procedures and medications, and up to three diagnoses
- **History:** how many outpatient, emergency and inpatient visits the patient had in the year before
- **Diabetes information:** glucose and A1c test results, insulin and other diabetes medicines, and whether medicines were changed
- **The answer column, `readmitted`:** `<30` (came back within 30 days), `>30` (came back later) or `NO` (did not come back)

**Our goal:** predict whether a patient will be readmitted **within 30 days**, using only information known **when the patient leaves the hospital**.

**Why this matters:** a hospital could use it to decide which patients need extra follow-up.

---

## 2. Phase 1: CLEAN (fixing the data)

We never changed the original file. We worked on a copy, so every change can be proved.

| Problem we found | How big | What we did | Why |
|---|---|---|---|
| Missing data hidden behind `?` | `weight` 96.86%, medical specialty 49.08%, payer code 39.56%, race 2.23% | Turned `?` into real missing values (192,852 in total) | Computers cannot see that `?` means "missing" |
| `weight` almost empty | 96.86% missing | Dropped the column | It cannot teach a model anything |
| Other gaps | 10 text columns; 2,273 missing race values | Filled 143 race values from the same patient's other visits; labelled the rest `Unknown` | We never invent values |
| `None` in two lab columns | glucose 96,420, A1c 84,748 | Kept as its own category, "test not done" | It is real information, not missing data |
| Diagnosis codes lost leading zeros | 2,646 / 1,560 / 1,526 codes | Padded them back to 3 digits | So the same code is not split into two |
| Numeric codes are unreadable | 3 ID columns | Added readable label columns using `IDs_mapping.csv` | Easier to read and explain |
| Patients who died | 1,652, none readmitted | Flagged them and left them out of modelling | A patient who died cannot come back |
| Duplicates | 0 | Nothing to remove | We checked three different ways |
| Repeat patients | 16,773 patients appear more than once (47,021 rows) | Noted it for Phase 3 | Needed to split train and test safely |
| Outliers | High-visit patients are readmitted 25.66% of the time vs 11.16% overall | Kept them | They are real patients and carry signal |
| Two medicine columns that are always `No` | - | Dropped them | No information |

**Result:** the file went from 101,766 x 50 to **101,766 x 53**, with 0 missing cells and 0 duplicates, and the target column is identical to the original. The cleaned file is `diabetic_cleaned.csv`.

---

## 3. Phase 2: VISUALIZE (looking for patterns)

We used the 100,114 encounters that remain after removing patients who died. **The overall 30-day readmission rate is 11.34%**, so only about **1 in 9** patients comes back.

**Main findings**

| Finding | Numbers |
|---|---|
| **Prior inpatient stays** are the strongest signal | 0 stays: 8.6%. 1: 13.2%. 2: 17.9%. 3 or more: 26.2% |
| **Prior emergency visits** also matter | 10.6% with none, rising to 25.2% with 3 or more |
| **Where the patient goes after** matters | Home 9.3%, another rehab facility 27.7% |
| **Two clues together** separate risk best | From 6.6% (no prior stays, went home) to 28.1% (3+ stays, other destinations; a small group) |
| Longer hospital stays go with higher rates | About 8.4% rising to about 14.5% |
| Insulin changes | Dose lowered 14.1%, raised 13.3%, steady 11.3%, no insulin 10.2% |
| Age, admission source, outpatient visits, A1c | Weak signals |
| Strongest single link to the answer | Correlation 0.141 (prior inpatient stays), so a model would be modest |

The graphs include the target distribution, an overview of nine columns, prior outpatient, emergency and inpatient visits, admitting specialty, admission source, glucose test, diabetes medication, A1c, medication change, insulin, and a heatmap of prior stays by discharge destination.

---

## 4. Phase 3: PREDICT (building the models)

**The question:** will this patient be readmitted within 30 days? (yes = 1, no = 0). This is a **classification** problem.

### Step by step

1. **Check the data.** We loaded the cleaned file and created the yes/no answer column `readmit_30` from `readmitted`.
2. **Choose inputs (X) and the answer (y).** We removed patients who died, leaving **100,114 rows** (11,357 readmitted). We removed the answer column, the original `readmitted` column (it is the same answer, which would be *leakage*) and the ID columns. We also removed four duplicate or constant columns, leaving **46 input columns**.
3. **Split into train and test.** 80% train (80,239 rows) and 20% test (19,875 rows). The split is **by patient**, so nobody appears in both sets (**0 shared patients**), and both sets have about the same share of readmitted patients (11.36% and 11.27%).
4. **Prepare the data.** Numbers are scaled, and text columns become 0/1 columns. "Not measured" is used for the lab columns. Everything is learned from the training data only.
5. **Build a baseline.** A "model" that always says "not readmitted". It scores **88.7% accuracy but catches 0 of 2,240 readmitted patients**. This is why we do not judge by accuracy.
6. **Train two models:**
   - **Logistic Regression** is simple and easy to explain.
   - **Random Forest** is many decision trees voting together.
   Both give extra weight to the rare readmitted patients.
7. **Test them** on the 19,875 patients they had never seen.

### Results (test set, 2,240 patients were really readmitted)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Caught | False alarms |
|---|---|---|---|---|---|---|---|
| Baseline (always No) | 0.8873 | 0 | 0 | 0 | 0.500 | 0 | 0 |
| Logistic Regression | 0.6572 | 0.1750 | 0.5496 | 0.2654 | 0.6530 | 1,231 | 5,805 |
| Random Forest | 0.6296 | 0.1724 | 0.6018 | 0.2681 | 0.6579 | 1,348 | 6,469 |

**What the words mean**

- **Recall:** of the patients who really came back, how many did we catch?
- **Precision:** of the patients we flagged, how many really came back?
- **F1:** one number that balances recall and precision.
- **ROC-AUC:** how well the model ranks risky patients above safe ones (0.5 is guessing, 1.0 is perfect).

**What the results show**

- Both models beat the baseline: they catch **55% to 60%** of readmitted patients.
- The two models are **almost tied** (ROC-AUC 0.6530 vs 0.6579), so we do not call one the winner. Random Forest catches more patients, and Logistic Regression raises fewer false alarms.
- Accuracy drops because the models flag many patients. That is a trade-off, not a mistake.
- Train scores are only about 0.05 higher than test scores, so overfitting is mild.

### Which features mattered most

We shuffled one column at a time and measured how much the Random Forest got worse.

| Feature | Drop in ROC-AUC |
|---|---|
| Prior inpatient stays (`number_inpatient`) | 0.0460 |
| Discharge destination (`discharge_disposition_id`) | 0.0379 |
| Prior emergency visits | 0.0046 |
| Insulin | 0.0029 |

These features are **associated with** the prediction. We do not claim they **cause** readmission.

### Where the model fails

- **First-time patients are often missed.** With no prior inpatient stays, recall is only 0.353. With 1 or more it is 0.796 to 0.940.
- **Many false alarms:** about 4.8 wrong flags for every patient caught (6,469 / 1,348), and only about 17% of flagged patients are really readmitted.
- **Older patients are flagged more often** (0.261 at age 20-30 to 0.597 at age 90-100), although their real readmission rate is similar. We have not verified why.

---

## 5. Final takeaway

Readmission can be predicted better than guessing, mainly from **prior hospital use** and **discharge destination**. The performance is modest, so the model should be used as a **screening aid** to help staff prioritise follow-up, never as a decision maker.

## 6. Limitations

- One train/test split only, with no tuning and no cross-validation.
- The default 0.5 cut-off was used.
- We did not compute PR-AUC.
- The data has no information about home support or follow-up care.
- Feature importance shows association, not cause.

## 7. Files in this project

| File | Purpose |
|---|---|
| `diabetic_data.csv` | Original data |
| `diabetic_cleaned.csv` | Cleaned data from Phase 1 |
| `IDs_mapping.csv` | Code lookup |
| `datawars_presentation.html` | 5-minute presentation |
| `datawars_solarflare.html` | Interactive results page with data traces |

## 8. How to run

1. Open Google Colab and upload `diabetic_cleaned.csv` with `files.upload()`.
2. Run the Phase 3 blocks in order: load and verify, define X and y, split, preprocessing, baseline, models, evaluation, comparison, feature importance, error analysis.
3. Fixed seed `random_state=42` is used for the split and the models.
