from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from simulations.training_simulation import create_env
from simulations.training_simulation import args

# TODO: сглаживание для графиков
if __name__ == '__main__':
    model = A2C.load(r"C:\Users\sskil\PycharmProjects\course_work\saved_models\sb3_A2C.zip")
    test_env = create_env(args.net_file_2single, args.route_file,use_gui=True)
    obs, _ = test_env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = test_env.step(action)

