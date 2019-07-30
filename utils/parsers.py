import argparse

def GenLQRParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_title", type=str, help="Name of experiment. The results will be saved to a folder"
                                                    "with this name")
    parser.add_argument("--horizon", type=int, default=120, help="Maximum number of total steps for an identification "
                                                                 "experiment.")
    parser.add_argument("--exp_length", type=int, default=6, help="The total length of an identification trial")
    parser.add_argument("--reward_threshold", type=float, default=10, help="Below this value the total reward is "
                                                                           "clipped to prevent gradient explosion")
    parser.add_argument("--eigv_low", type=float, default=0.5, help="Minimum absolute value of eigenvalue A can have. "
                                                                    "If A has an eigenvalue below this, we "
                                                                    "sample a new A.")
    parser.add_argument("--eigv_high", type=float, default=2.0, help="Maximum absolute value of eigenvalue A can have. "
                                                                     "If A has an eigenvalue above this, we "
                                                                     "sample a new A.")
    parser.add_argument("--elem_sample", type=bool, default=1, help="If true, the elements of A and B are sampled "
                                                                    "randomly")
    parser.add_argument("--eval_matrix", type=bool, default=1, help="If this is true, the A and B matrices are fixed to"
                                                                    "the values from 'On the sample complexity of the "
                                                                    "linear quadratic regulator")
    parser.add_argument("--full_ls", type=bool, default=1, help="If true, we use all the samples from the rollouts for"
                                                                "Least Squares. If not, we just use the last input"
                                                                "output pair")
    parser.add_argument("--dim", type=int, default=3, help="Dim of A and B matrices")
    parser.add_argument("--eval_mode", type=bool, default=0, help="Disable a few peculiarities of training for "
                                                                  "evaluation time. If evaluating, SET TO TRUE")
    parser.add_argument("--analytic_optimal_cost", type=bool, default=0, help="If true, compute the optimal cost of "
                                                                              "the synthesized system analytically")
    parser.add_argument("--gaussian_actions", type=bool, default=0, help="If true, the actions are sampled from a "
                                                                         "Gaussian with mean 0 and identity cov matrix "
                                                                         "Otherwise, they are taken from the neural"
                                                                         "network")
    parser.add_argument("--rand_num_exp", type=bool, default=1, help="If true, the max number of trials is sampled"
                                                                     "uniformly from 2 * dim to (horizon / exp_length "
                                                                     "Otherwise, the max num is horizon / exp length")
    parser.add_argument("--num_cpus", type=int, default=2, help="Number of CPUs to use in the trial")
    parser.add_argument("--multi_node", type=bool, default=0, help="If true, run RLlib in cluster mode")
    parser.add_argument("--use_s3", type=bool, default=0, help="Upload results to s3")
    return parser
