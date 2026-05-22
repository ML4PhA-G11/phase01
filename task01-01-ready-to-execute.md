# Task 01 — Reproduce the LBM → Neural-Network → LBM pipeline

## Purpose
This project is a **glue project**: it ties together three independent
repositories so the full workflow can be reproduced end to end. The workflow is:

1. **Generate data** from physics simulators (Lattice-Boltzmann, LBM).
2. **Train neural-network models** on that simulation data.
3. **Apply the trained models back** to the corresponding simulators.

This document is self-contained: it lists every repository, where each lives on
the local machine, and the exact commands to run for the first experiment.

---

## Source repositories (remote)
| Role | Repository |
|------|------------|
| Simulator (data generation) | `git@github.com:ML4PhA-G11/phase01.git` |
| Model training | `git@github.com:ML4PhA-G11/model-experiments.git` |
| Applying models back | `git@github.com:ML4PhA-G11/Apply-NN-KarmanVortexStreet.git` |
| Data (Git-LFS / GPFS based) | `git@github.com:ML4PhA-G11/data.git` |

> The **data** repository already contains both the data generated from the
> simulator and the trained models. If it is available, you do **not** need to
> re-run the simulators or re-train the models.

## Local checkout paths
These are the working copies expected on the local machine, relative to this
`phase01/` directory:

| Role | Local path |
|------|------------|
| Simulator | `../simulators/` |
| Training | `../model-experiments/` |
| Applying models back | `../Apply-NN-KarmanVortexStreet/` |
| Data | `../data/` |

---

## Experiment 01 — Kármán vortex street, BGK, GAVG

**Configuration**
- **Simulator:** BGK (single-relaxation-time LBM) for a Kármán vortex street.
- **Neural-network architecture:** GAVG (group-averaged / D4-equivariant).

### Prerequisites
- Local checkouts of all four repositories at the paths listed above.
- A Python environment with the dependencies of each repository installed.
- Two repositories must be on the **`dev-C04-helper-scripts`** branch:
  - `../model-experiments/`
  - `../Apply-NN-KarmanVortexStreet/`

### Steps

**1. Check whether the simulation data already exists.**
Look for the pre-generated dataset at:
```
/gpfs/scratch1/shared/scur0076/output-lbm-data-02-30000steps-data.per.step-npy
```

**2a. If the data exists** → skip generation and use it directly for training
(go to step 3).

**2b. If the data does NOT exist** → re-create it from the simulator:
```bash
cd ../simulators
python lbm_karman-ng.py --save-every 1
```

**3. Train the model** using `model-experiments`
(branch: `dev-C04-helper-scripts`):
```bash
cd ../model-experiments
git checkout dev-C04-helper-scripts
scripts/train-d4equivariant.sh
```

**4. Apply the trained model back** to the simulator
(branch: `dev-C04-helper-scripts`):
```bash
cd ../Apply-NN-KarmanVortexStreet
git checkout dev-C04-helper-scripts
python apply-nn.py
```
