from stable_baselines3 import PPO
from simulations.training_simulation import create_2single_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv
import time


out_csv_name = r"C:\Users\sskil\PycharmProjects\course_work\saved_history\sb3_PPO\history"
save_path = r"C:\Users\sskil\PycharmProjects\course_work\saved_models\sb3_PPO"
tensorboard_log_path = r"C:\Users\sskil\PycharmProjects\course_work\tensorboard_log\sb3_PPO"

if __name__ == "__main__":
    vec_env = make_vec_env(create_2single_env, 4)
    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=tensorboard_log_path)
    start = time.time()
    model.learn(total_timesteps=300_000)
    print(f"training time {time.time() - start}")
    model.save(save_path)
