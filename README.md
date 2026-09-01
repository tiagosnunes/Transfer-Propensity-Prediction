
# Transfer Purchase Propensity Prediction

A machine learning model that predicts which guests of a short-term rental company ("Feels Like Home") are likely to purchase an airport transfer service — turning a reactive booking process into a proactive, data-driven guest communication strategy.

## Business Context

Feels Like Home (FLH) is a short-term rental property management company responsible for guest check-in. Since most guests don't travel by car, FLH advertises an airport transfer service operated by a partner company — improving guest comfort while earning a commission on each transfer booked.

Around 10% of reservations convert into a transfer, and FLH wants to increase that share. The goal of this project was to build a model that predicts which guests are more likely to book a transfer, so the marketing team can target its outreach — moving from a reactive approach ("send a transfer to whoever asks") to a proactive one ("anticipate who will want it").

**Key constraint:** the prediction must happen in the window between a reservation being received (when guest information becomes available) and the check-in date (when the transfer service can actually be offered) — the model can only use information available at booking time.

The challenge: only **7.6%** of historical reservations included a transfer purchase in this dataset, making this a highly imbalanced classification problem.

## Data

Four related tables were combined and engineered into a single modeling dataset:

- **Apartments** — property characteristics: region, neighbourhood, elevator, bed configuration, typology, max occupation
- **Reservations** — booking details: check-in date, length of stay, booking channel (OTA), reservation value, booking date, guest country, number of guests
- **Transfers** — historical transfer bookings: date/time, number of guests, return-trip flag, request date — used to build the binary target variable (`HasTransfer`)
- **NewReservations** — 393 bookings made after the training data was collected, with no transfer history yet; this is the set the final model is applied to

## Methodology

**Feature engineering & cleaning**
- Built the binary target variable (`Tem_Transfer`) by matching reservations against historical transfer records
- Cleaned and consolidated OTA channel categories, capped outliers (99th percentile) on length of stay and reservation value, and engineered a booking-lead-time feature (`Antecedencia_dias`)
- Merged reservation and apartment data into a single modeling table

**Model progression**
Four models were built and compared, in a deliberate progression from baseline to tuned:

| Model | Result (test) | Diagnosis |
|---|---|---|
| XGBoost (baseline, `scale_pos_weight`) | FLH 0.759 / AUC 0.841 | Severe overfitting (train-test FLH gap: 12.4pp) |
| GBT (baseline, balanced weights) | FLH 0.771 / AUC 0.847 | Moderate overfitting (FLH gap: 7.5pp) |
| GBT + regularization + GridSearch | FLH 0.771 / AUC 0.848 | Overfitting nearly eliminated (FLH gap: 3.5pp) |
| **XGBoost + regularization + GridSearch (selected)** | **FLH 0.772 / AUC 0.845** | **No overfitting (FLH gap: 2.0pp)** |

*FLH — the project's custom central evaluation metric, chosen because it balances the identification of guests with and without a transfer more fairly than plain accuracy, which would be inflated by simply predicting the majority class (92% no-transfer) every time.*

**Model selection**
The regularized XGBoost model was chosen as the most stable, reliable, and interpretable of the four: smaller overfitting gaps across every metric, and a simpler solution (200 vs. 300 estimators) than its closest competitor, GBT — a meaningful advantage when the model needs to be explained to non-technical teams.

**Threshold decision**
The classification threshold was set at **0.48**, derived from ROC curve and Youden's index analysis to balance sensitivity and specificity. This reflects a deliberate business assumption: the cost of missing an interested guest (false negative) is higher than the cost of contacting someone who won't convert (false positive) — so a slightly more inclusive threshold was preferred. A lower threshold (0.30) was also discussed as a valid alternative, trading a small increase in unnecessary contacts for capturing more true buyers.

## Results

Applied to the 393 unseen `NewReservations`, the model flagged **83 guests (21.1%)** as likely to purchase a transfer — nearly 3x the historical base rate of 7.6%. This isn't a contradiction: the model isn't predicting universal conversion, it's concentrating the team's attention on the highest-probability profiles within a group that has no purchase history to rely on.

Business impact: instead of contacting all 393 guests, the communications team can focus on the 83 flagged as high-propensity — reducing operational cost and communication fatigue while increasing expected conversion per contact.

## Tech Stack

- **Data handling:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn` (`GridSearchCV`, `DecisionTreeClassifier`, model evaluation), `xgboost`
- **Visualization:** `matplotlib`, `seaborn`
- **Utilities:** `pycountry`, `joblib`
- **Environment:** Google Colab / Jupyter Notebook

## How to Run

1. Clone this repository
2. Open the notebook in Google Colab or Jupyter
3. Upload the dataset file when prompted (`CPP_Transfer_Dataset.xlsx`, containing the `Transfers`, `Reservations`, `Apartments`, and `NewReservations` sheets)
4. Run all cells in order

## Next Steps

As outlined in the project conclusion: monitor model performance in production, compare predictions against actual outcomes, and periodically retrain as new booking data accumulates.

## Context

Group project developed for the course *"Classificação, Perfis e Propensões"* (Classification, Profiling and Scoring), part of the Postgraduate Programme in Analytics for Business at INDEG-ISCTE Executive Education.

## Authors

- Tiago Nunes
- Ana Rita Taborda
- Artur Albuquerque
- Pedro Franco
