from maze_env import Maze
from QLearn.RL_brain import QLearningTable


def update():
    for episode in range(100):
        # 获取初始位置坐标信息（1，1）
        observation = env.reset()

        while True:
            # 刷新环境
            env.render()

            # RL 选择基于观测的下一个行为
            action = RL.choose_action(str(observation))
            # RL 根据采取的行为获取下一个观测和当前的奖励
            observation_, reward, done = env.step(action)
            # RL 更新Q-table
            RL.learn(str(observation), action, reward, str(observation_))

            observation = observation_

            if done:
                break

    print('end of game')
    env.destory()


if __name__ == '__main__':
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
