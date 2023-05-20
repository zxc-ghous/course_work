from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from simulations.training_simulation import create_2single_env

# TODO: сглаживание для графиков
#       парсер summary файлов из sumo и тоже для них графики
#       графики для того чтобы сравнить результаты работы алгоритмов
#       время жизни demand элементов в сетях и увеличить интенсивность отдельно для tomsk.rou
#       посмотреть на то какие результаты у single_env
#       сделать MARL сети
if __name__ == '__main__':
    model = A2C.load(r"C:\Users\sskil\PycharmProjects\course_work\saved_models\sb3_A2C.zip")
    test_env = create_2single_env(True)

    mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=100)

    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")