# V3.2 acceptance matrix

| Stage | Required evidence | Stop condition |
| --- | --- | --- |
| 64 patches | each branch has gradients and fixed seed; detail MAE improves >=30% | any branch disconnected or delta>1 residual gradient nonzero |
| 1k flow | 32 fixed panels; Visual RMSE <=1.10x Physical; pre-projection violation <=1% | automated check or manual noise/black-output review fails |
| 5k flow | all 463 pairs; Visual RMSE <=1.05x; LPIPS/DISTS improve >=3%; Edge F1 and PSD improve | any gate fails; do not launch 40k |
| final | Physical hard gates; Optical perceptual gains >=5%; SAR bias/PSD/ENL/histogram gates | do not create `best_joint.pt` |

The main paper result uses one deterministic Physical output. No best-of-K stochastic sample may
be reported as the primary result. The three closed tests are read only after validation has
created `best_joint.pt`.
