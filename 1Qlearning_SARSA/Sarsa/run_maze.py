from maze_env import Maze
from Sarsa.RL_brain import SarsaTable
from Sarsa.RL_brain import SarsaLambdaTable


def update():
    for episode in range(100):
        # 获取初始位置坐标信息（1，1）
        observation = env.reset()

        # RL 选择基于观测的下一个行为
        action = RL.choose_action(str(observation))

        while True:
            # 刷新环境
            env.render()

            # RL 根据采取的行为获取下一个观测和当前的奖励
            observation_, reward, done = env.step(action)

            # RL 基于刚刚的得到的观测选择再下一个行为
            action_ = RL.choose_action(str(observation_))

            # RL 更新Sarsa-table
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # 这里下一次行为就会用刚刚选择的
            observation = observation_
            action = action_

            if done:
                break

    print('end of game')
    env.destory()


if __name__ == '__main__':
    env = Maze()
    RL = SarsaTable(actions=list(range(env.n_actions)))
    # RL = SarsaLambdaTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
