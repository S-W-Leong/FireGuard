# FireGuard boosted tree comparison protocol

11 October 2026. The user approved one short nonlinear model comparison after
reviewing the logistic model's limited improvement over recent fire history.
This is an explicitly authorized extension of the earlier stopping rule.

Use scikit-learn 1.7.2 HistGradientBoostingClassifier with one fixed configuration:
log-loss, learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
min_samples_leaf=100, l2_regularization=10, early_stopping=False,
class_weight=None and random_state=17. Disable automatic early stopping to avoid
its random validation split. Limit native computation to two threads. No
algorithm, feature or hyperparameter search follows this fit.

Use the same 21 named inputs as the environmental logistic candidate: preceding
FWI summaries, prior detection counts, FFMC/DC, peat and measured land-cover and
oil-palm fractions, and calendar seasonality. Reuse the verified feature
construction. Coordinates, cell identity and exposure hectares/population are
excluded from fitting. Cropland and plantation fractions describe the landscape;
hectares remain separate exposure inputs.

Train on 2022 and purge outcomes reaching 2023. Fit a sigmoid calibrator to
2023 logits and labels, with no base-model refit. Evaluate the existing logistic
model, FWI, prior 7/30-day detection counts and 2022 historical frequency on the
same 949 cells. The 2023 adoption gate is: higher average precision than the
incumbent logistic model AND at least its risk-only Recall@50. Record the choice
before examining 2024 outcomes. A tie retains the incumbent.

2024 remains exploratory because prior project decisions have examined it.
Report all results, even when they disagree with the 2023 choice. The supplied
dataset still has only one base-model training year; neither model is a verified
live seven-day forecast or an exact occurrence probability.

Retain the incumbent package while the dashboard is built. Save the challenger
under `data/experiments/boosted-tree-v1`, with protocol/input hashes, feature
names, the model as numeric tree nodes in JSON, calibration, weekly scores,
monitoring metrics and the recommendation. Provide inference code that requires
no pickle or retraining. Exported scores must match sklearn on every evaluation
row. Compare top-K risk-only and exposure-weighted policies for population,
annual cropland and closed-canopy oil palm separately. Do not combine their
units or interpret them as losses avoided.

Verify threshold boundary traversal, finite input rejection, export equivalence,
chronological training, independence from 2024 labels, the 2023 adoption gate,
fixed cell/week coverage and preservation of incumbent artifacts. Follow the
[HistGradientBoostingClassifier documentation](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
for the installed version.
