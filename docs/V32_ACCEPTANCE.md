# V3.2 canonical acceptance matrix

This matrix applies to the 2017–2024 canonical dataset and its fixed 141-pair 2023
`validation_temporal` protocol. Reports from the historical 463-pair protocol are not comparable.

| Stage | Required evidence | Stop condition |
| --- | --- | --- |
| 64-patch connectivity | finite forward/backward, fixed-seed reproducibility, frozen physical unchanged, and exact zero residual gradient for ineligible/`delta_days>1` patches | any disconnected branch, non-finite value, physical drift, or temporal-mask leak |
| 1k pilot | fixed quick32 panels; Visual RGB RMSE `<=1.10x` Physical; pre-projection violation `<=1%`; no checkerboard, colored noise, or black collapse; new component has measurable nonzero output | automated guardrail fails or the new component is numerically inert |
| 5k pilot | complete 141-pair validation; Visual RGB RMSE `<=1.05x`; LPIPS and DISTS each improve `>=3%`; Edge F1 and optical PSD both improve; most-scene risk gates pass | any gate fails; do not launch the long run |
| final validation | all Physical hard gates; Optical LPIPS/DISTS each improve `>=5%`; Edge/PSD/bounds and scene-risk gates pass; SAR bias/PSD/ENL/histogram/P01/P99 gates pass | do not create `best_visual.pt` or `best_joint.pt` |
| closed tests | validation-selected checkpoint and fixed preregistered seed; spatial, temporal, and joint splits all support the claim | no checkpoint reselection or best-of-K after tests are opened |

Physical hard gates:

```text
SAR -> Optical RMSE <= 0.03909
SAR -> Optical SAM  <= 5.716 degrees
Optical -> SAR RMSE <= 5.0 dB
Optical -> SAR bias <= 0.5 dB
```

Final Optical majority/risk gates additionally require at least 70% of scenes to improve Edge F1
or DISTS, no more than 10% of scenes to exceed 5% RMSE degradation, and pre-projection violation
`<=0.1%`. Texture-rich/sparse and land-cover slices must agree with the aggregate conclusion.

The primary distortion result always uses one deterministic Physical output. Stochastic evaluation
uses a preregistered fixed seed plus multi-seed distribution calibration; best-of-K is prohibited.
The three closed tests remain unread until validation selection creates `best_joint.pt`.
