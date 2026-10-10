# MVP model selection rule

Recorded before inspecting the environmental experiment's results on 11 October
2026. Recommend the new environmental logistic model if its 2023 average
precision is at least that of the frozen earlier model without coordinates,
evaluated on the same 949-cell cohort. Otherwise recommend the earlier model
without coordinates. Both are logistic models; no additional algorithm or
feature search follows the 2024 evaluation. The choice is made by development
data, not the exploratory 2024 outcome.

The previous model receives sigmoid calibration on the same 2023 cohort for
exposure allocation. Its base coefficients remain frozen. Compare 2024 results
honestly even if they disagree with the development choice. This rule selects
an MVP ranking model; it does not establish operational readiness.
