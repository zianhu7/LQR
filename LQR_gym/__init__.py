from gym.envs.registration import register

register(
        id='LQR_env',
        entry_point='LQR.LQR_gym:LQR_env',
        )
