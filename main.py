from models.DDQN import Agent
from simulations.training_simulation import create_env
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = create_env(False, 5000)
    agent = Agent(env)
    agent.train(30)

    fig, ax = plt.subplots(figsize=(15, 6))
    agent.model.save('3road_model')
    sns.lineplot(x=list(range(len(agent.reward_history))),
                 y=agent.reward_history)
    plt.show()

# TODO: выбрать сколько должно быть машин в час на дорогах среды (1800/час не много ли?)
#       add_per_agent_info установить на False?
