import argparse
import json


def env_args(parser):
    # ================================================================================================
    #                    ENV PARAMS
    # ================================================================================================
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
    parser.add_argument("--elem_sample", type=int, default=1, help="If true, the elements of A and B are sampled "
                                                                    "randomly")
    parser.add_argument("--eval_matrix", type=int, default=0, help="If this is true, the A and B matrices are fixed to"
                                                                    "the values from 'On the sample complexity of the "
                                                                    "linear quadratic regulator")
    parser.add_argument("--full_ls", type=int, default=1, help="If true, we use all the samples from the rollouts for"
                                                                "Least Squares. If not, we just use the last input"
                                                                "output pair")
    parser.add_argument("--dim", type=int, default=3, help="Dim of A and B matrices")
    # TODO(@evinitsky) remove this param, it seems ultra risky to have both it and eval_matrix.
    parser.add_argument("--eval_mode", type=int, default=0, help="Disable a few peculiarities of training for "
                                                                  "evaluation time. If evaluating, SET TO TRUE")
    parser.add_argument("--analytic_optimal_cost", type=int, default=1, help="If true, compute the optimal cost of "
                                                                              "the synthesized system analytically")
    parser.add_argument("--gaussian_actions", type=int, default=0, help="If true, the actions are sampled from a "
                                                                         "Gaussian with mean 0 and identity cov matrix "
                                                                         "Otherwise, they are taken from the neural"
                                                                         "network")
    parser.add_argument("--rand_num_exp", type=int, default=1, help="If true, the max number of trials is sampled"
                                                                     "uniformly from 2 * dim to (horizon / exp_length "
                                                                     "Otherwise, the max num is horizon / exp length")


def GenLQRParserRLlib():
    parser = argparse.ArgumentParser()
    env_args(parser)
    # ================================================================================================
    #                    RLLIB PARAMS
    # ================================================================================================
    parser.add_argument("exp_title", type=str, help="Name of experiment. The results will be saved to a folder"
                                                    "with this name")
    parser.add_argument("--num_cpus", type=int, default=2, help="Number of CPUs to use in the trial")
    parser.add_argument("--multi_node", type=int, default=0, help="If true, run RLlib in cluster mode")
    parser.add_argument("--use_s3", type=int, default=0, help="Upload results to s3")
    parser.add_argument("--checkpoint_freq", type=int, default=1, help="How many iterations are required for a model "
                                                                       "checkpoint.")
    parser.add_argument("--num_iters", type=int, default=1, help="Total number of gradient steps")
    parser.add_argument("--num_samples", type=int, default=1, help="How many times to repeat each experiment")
    parser.add_argument("--grid_search", type=int, default=0, help="Do a tune grid search if true")
    parser.add_argument("--train_batch_size", type=int, default=30000, help="How many steps in a gradient batch")
    return parser


def GenLQRParserBaseline():
    parser = argparse.ArgumentParser()
    env_args(parser)
    # ================================================================================================
    #                    PARAMS
    # ================================================================================================
    parser.add_argument("exp_title", type=str, help="Name of experiment. The results will be saved to a folder"
                                                    "with this name")
    parser.add_argument("--num_cpus", type=int, default=1, help="Number of CPUs to use in the trial")
    parser.add_argument('--num_steps', type=int, default=5000, help='How many total steps to perform learning over')
    parser.add_argument('--rollout_size', type=int, default=1000, help='How many steps are in a training batch.')
    parser.add_argument('--use_s3', action='store_true', help="If true, will upload to an s3 bucket. WARNING: "
                                                              "the path is pretty hardcoded")
    parser.add_argument('--checkpoint_freq', type=int, default=1, help="How often to check if the model has improved "
                                                                       "and consequently save it")
    parser.add_argument('--callback', type=int, default=1, help="Whether to save the model using a callback or just"
                                                                 "at the end of the training")
    return parser


def RolloutParser():
    """Used to rollout an environment a specified number of times"""
    parser = argparse.ArgumentParser()
    env_args(parser)
    parser.add_argument("--steps", type=int, default=10000, help="Number of steps to roll out.")
    parser.add_argument("--out", help="Output filename.")
    parser.add_argument("--append", type=int, default=1, help="If absent, files are written instead of appended")
    parser.add_argument("--eigv_gen", type=int, default=False,
                        help="Eigenvalue generalization tests for eigenvalues of A. Sample matrices randomly and "
                             "see how we perform relative to the top eigenvalue")
    parser.add_argument("--opnorm_error", type=int, default=False,
                        help="Operator norm error of (A-A_est)")
    return parser


def ReplayParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="path to the desired pre-trained model")
    parser.add_argument(
        "--config",
        default="{}",
        type=json.loads,
        help="Algorithm-specific configuration (e.g. env, hyperparams). "
             "Supresses loading of configuration from checkpoint.")
    parser.add_argument(
        "--run",
        type=str,
        default='PPO',
        help="The algorithm or model to train. This may refer to the name "
             "of a built-on algorithm (e.g. RLLib's DQN or PPO), or a "
             "user-defined trainable function or class registered in the "
             "tune registry.")
    env_args(parser)

    return parser