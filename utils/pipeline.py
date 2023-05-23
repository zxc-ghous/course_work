from stable_baselines3 import A2C, DQN, PPO
from simulations.training_simulation import create_env
from stable_baselines3.common.env_util import make_vec_env
from simulations.training_simulation import args
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

out_csv_name_ppo = r"C:\Users\sskil\PycharmProjects\course_work\saved_history\sb3_PPO\history"
save_path_ppo = r"C:\Users\sskil\PycharmProjects\course_work\saved_models\sb3_PPO"
tensorboard_log_path_ppo = r"C:\Users\sskil\PycharmProjects\course_work\tensorboard_log\sb3_PPO"

out_csv_name_a2c = r"C:\Users\sskil\PycharmProjects\course_work\saved_history\sb3_A2C\history"
save_path_a2c = r"C:\Users\sskil\PycharmProjects\course_work\saved_models\sb3_A2C"
tensorboard_log_path_a2c = r"C:\Users\sskil\PycharmProjects\course_work\tensorboard_log\sb3_A2C"

out_csv_name_dqn = r"C:\Users\sskil\PycharmProjects\course_work\saved_history\sb3_DQN\history"
save_path_dqn = r"C:\Users\sskil\PycharmProjects\course_work\saved_models\sb3_DQN"
tensorboard_log_path_dqn = r"C:\Users\sskil\PycharmProjects\course_work\tensorboard_log\sb3_DQN"


class Pipeline:
    def __init__(self, n_envs=4, total_timesteps=300_000):
        self.n_envs = n_envs
        self.total_timesteps = total_timesteps
        self.A2C_env = None
        self.PPO_env = None
        self.DQN_env = None
        self.__eval_env = make_vec_env(create_env, env_kwargs=dict(net_file=args.net_file_2single,
                                                                   route_file=args.route_file))
        self.__eval_env_a2c = make_vec_env(create_env, vec_env_cls=SubprocVecEnv,
                                           env_kwargs=dict(net_file=args.net_file_2single,
                                                           route_file=args.route_file))
        self.model_a2c = None
        self.model_ppo = None
        self.model_dqn = None

    def set_a2c(self, net_file, route_file, out_csv_name, num_seconds=3600):
        self.A2C_env = make_vec_env(create_env, n_envs=self.n_envs, vec_env_cls=SubprocVecEnv,
                                    env_kwargs=dict(net_file=net_file,
                                                    route_file=route_file,
                                                    out_csv_name=out_csv_name,
                                                    num_seconds=num_seconds))

    def set_ppo(self, net_file, route_file, out_csv_name, num_seconds=3600):
        self.PPO_env = make_vec_env(create_env, n_envs=self.n_envs,
                                    env_kwargs=dict(net_file=net_file,
                                                    route_file=route_file,
                                                    out_csv_name=out_csv_name,
                                                    num_seconds=num_seconds))

    def set_dqn(self, net_file, route_file, out_csv_name, num_seconds=3600):
        self.DQN_env = make_vec_env(create_env, n_envs=self.n_envs,
                                    env_kwargs=dict(net_file=net_file,
                                                    route_file=route_file,
                                                    out_csv_name=out_csv_name,
                                                    num_seconds=num_seconds))

    def start_learning_a2c(self):
        if self.A2C_env:
            self.model_a2c = A2C("MlpPolicy", self.A2C_env, device="cpu", verbose=1,
                                 tensorboard_log=tensorboard_log_path_a2c, learning_rate=0.0009)
        else:
            print("no a2c model set")
            return

        stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=3, verbose=1)
        eval_callback = EvalCallback(self.__eval_env_a2c, eval_freq=60_000, n_eval_episodes=2,
                                     callback_after_eval=stop_train_callback,
                                     verbose=1)
        print("start A2C learning")
        self.model_a2c.learn(self.total_timesteps, progress_bar=True, callback=eval_callback)
        self.model_a2c.save(save_path_a2c)

    def start_learning_ppo(self):
        if self.PPO_env:
            self.model_ppo = PPO("MlpPolicy", self.PPO_env, verbose=1,
                                 tensorboard_log=tensorboard_log_path_ppo, learning_rate=0.0009)
        else:
            print("no ppo model set")
            return

        stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=3, verbose=1)
        eval_callback = EvalCallback(self.__eval_env, eval_freq=60_000, n_eval_episodes=2,
                                     callback_after_eval=stop_train_callback,
                                     verbose=1)
        print("start PPO learning")
        self.model_ppo.learn(self.total_timesteps, progress_bar=True, callback=eval_callback)
        self.model_ppo.save(save_path_ppo)

    def start_learning_dqn(self):
        if self.DQN_env:
            self.model_dqn = DQN("MlpPolicy", self.DQN_env, verbose=1,
                                 tensorboard_log=tensorboard_log_path_dqn, learning_rate=0.0009)
        else:
            print("no dqn model set")
            return

        stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=3, verbose=1)
        eval_callback = EvalCallback(self.__eval_env, eval_freq=60_000, n_eval_episodes=2,
                                     callback_after_eval=stop_train_callback,
                                     verbose=1)

        print("start DQN learning")
        self.model_dqn.learn(self.total_timesteps, progress_bar=True, callback=eval_callback)
        self.model_dqn.save(save_path_dqn)


# TODO: установить другой callback на остановку когда нагарада переходит порог
#       почему то обучение расходится, сначала награда хорошая, потом слабая
if __name__ == "__main__":
    pipe = Pipeline()
    pipe.set_a2c(args.net_file_2single, args.route_file, out_csv_name_a2c)
    # pipe.set_ppo(args.net_file_2single, args.route_file, out_csv_name_ppo)
    # pipe.set_dqn(args.net_file_2single, args.route_file, out_csv_name_dqn)
    pipe.start_learning_a2c()
    # pipe.start_learning_ppo()
    # pipe.start_learning_dqn()
