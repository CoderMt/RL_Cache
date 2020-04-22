# main.py
import numpy as np

# 导入环境和学习方法
from paper.cache_evn import CacheEnv
from paper.rl import DDPG

# 设置全局变量
MAX_EPISODES = 500#最大回合数
MAX_EP_STEPS = 200 #每回合最大步骤数
ON_TRAIN = True #是否正在测试

# 设置环境
env = CacheEnv()
s_dim = env.state_dim #状态集
a_dim = env.action_dim #动作集
a_bound = env.action_bound

# 设置学习方法 (这里使用 DDPG)
rl = DDPG(a_dim, s_dim, a_bound) #神经网络输入

#测试时候运行train()
def train():
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_r = 0.
        for j in range(MAX_EP_STEPS):
            a = rl.choose_action(s)
            s_, r, done = env.step(a)
            rl.store_transition(s, a, r, s_)
            ep_r += r

            if rl.memory_full:
                # start to learn once has fulfilled the memory
                rl.learn()
            s = s_

            if done or j == MAX_EP_STEPS-1:
                print('Ep: %i | %s | ep_r: %.1f | steps: %i' % (i, '---' if not done else 'done', ep_r, j))
                break


if __name__ == '__main__':

    train()

