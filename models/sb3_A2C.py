from stable_baselines3 import A2C
from simulations.training_simulation import create_2single_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv
import time

# https://github.com/araffin/rl-tutorial-jnrr19/tree/sb3/


out_csv_name = r"C:\Users\sskil\PycharmProjects\course_work\saved_history\sb3_A2C\history"
save_path = r"C:\Users\sskil\PycharmProjects\course_work\saved_models\sb3_A2C"
tensorboard_log_path = r"C:\Users\sskil\PycharmProjects\course_work\tensorboard_log\sb3_A2C"


# step = 260k minimum loss
if __name__ == "__main__":
    vec_env = make_vec_env(create_2single_env, 4, vec_env_cls=SubprocVecEnv)
    model = A2C("MlpPolicy", vec_env, verbose=1, tensorboard_log=tensorboard_log_path, device="cpu")
    start = time.time()
    model.learn(total_timesteps=1000000)
    print(f"training time {time.time() - start}")
    model.save(save_path)