this project is a glue project to reproduce what we have done for creating data from simulators, using the data to train models, and then apply the models one by one back to the corresponding simulators.

# components of source code: remote repositories
simulator: git@github.com:ML4PhA-G11/phase01.git
training: git@github.com:ML4PhA-G11/model-experiments.git
applying models back: git@github.com:ML4PhA-G11/Apply-NN-KarmanVortexStreet.git
data: git@github.com:ML4PhA-G11/data.git (data includes data generated from simulator and trained models, so we do not need to re-run simulators and training) (the repo is gfs based)

# corresponding source paths on local
simulator: ../simulators/
trainning: ../model-experiments/
applying models back: ../Apply-NN-KarmanVortexStreet/
data: ../data

# Exp01: Karman vortex street, BGK, and GAVG
- simulator: BGK for Karman vortex street
- neural network architecture: GAVG

## Steps
1. check if we have the data in /gpfs/scratch1/shared/scur0076/output-lbm-data-02-30000steps-data.per.step-npy
2. if yes, we will use the data for training. if not, let us re-create the data by `python lbm_karman-ng.py --save-every 1`
3. train the data with model-experiments (branch: dev-C04-helper-scripts) by running scripts/train-d4equivariant.sh
4. apply the data by `python apply-nn.py` (branch: dev-C04-helper-scripts)

