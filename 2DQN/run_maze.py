from DQN.maze_env import Maze
from DQN.RL_brain import DeepQNetwork


def update():
    step = 0
    for episode in range(300):
        # 获取初始位置坐标信息（1，1）
        observation = env.reset()

        while True:
            # 刷新环境
            env.render()

            # RL 选择基于观测的下一个行为
            action = RL.choose_action(observation)

            # RL 根据采取的行为获取下一个观测和当前的奖励
            observation_, reward, done = env.step(action)

            # RL 更新replay buffer
            RL.store_transition(observation, action, reward, observation_)

            # 首先要有些记忆，所以迭代了一段时间后开始学习，再每5步学习一下
            if (step > 200) and (step % 5 == 0):
                RL.learn()

            observation = observation_

            if done:
                break
            step += 1

    print('end of game')
    env.destroy()


if __name__ == '__main__':
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000
                      # output_graph=True
                      )

    env.after(100, update)
    env.mainloop()
    RL.plot_cost()
