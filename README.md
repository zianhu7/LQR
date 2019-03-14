# Efficient Input Experiment Design

This repo contains the code and results from "Paper Title".
Words words words

## Installation and Running
If you have conda installed, run `conda env create -f environment.yml`. Then run
`source activate lqr` and subsequently `python setup.py develop`. This installs
the needed libraries and appends the environments to the path. Then, the script is
`gen_script.py` which will run the training for the sampler.

If you do not have conda, run `pip install -r requirements.txt` and then rnu `python setup.py
develop`.

## Generating the graphs
In the spirit of reproducibility, we've uploaded all of the trained models to this repository.
To run one of the trained models and generate all the graphs for the paper, run `TBD`

## Understanding the codebase
The main file is GenLQREnv.py which contains the environment used for training.
For each "horizon" number of steps it samples random A and B matrices, allows the system to perform
N rollouts of length `exp_length` where N=horizon/exp_length, and at the end uses the input, output
pairs to do a Least Squares estimation of A and B. The Ricatti equation is then solved to
generate a feedback matrix that is used to rollout for horizon steps and compute the LQR cost.
The  main parameters used to control it are the dimension (which, unfortunately
needs to be set manually as it was mistakenly hard-coded in) and the following environment
configs that can be set in `gen_script.py`
- 'elem_sample': if true all the elements of A and B are randomly sampled, if false eigenvalues
are sampled and then rotation matrices are applied.
- 'eval_matrix': this references the Q and R matrices that were taken from "On The Sample Complexity
    Of The Linear Quadratic Regulator"
- 'eigv_high': If 'elem_sample' is true the eigenvalues are bounded between -eigv_high and eigv_high.
    If 'elem_sample' is false this is the maximum eigenvalue of the sampled matrices.
- 'eigv_low': If 'elem_sample' is false, this is the minimal eigenvalue of the sampled matrices.
- 'reward_threshold': If the reward goes below reward_threshold, the reward is clipped at that value.
- 'full_ls': If this is true all of the input, output samples are used in the least squares estimation.
If it is false, only the last input-output pair of each rollout is used.
