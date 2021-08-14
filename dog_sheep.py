import gym
import numpy as np
from gym.envs.classic_control import rendering
from gym.envs.registration import register
from gym import spaces,core


class Dog_sheep(core.Env):
    # 如果你不想改参数，下面可以不用写
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):
        self.viewer = rendering.Viewer(800, 600)  # 1000x800 是画板的长和框
        self.r=100
        self.sheep_x = 0
        self.sheep_y = 0
        self.dog_theta=0
        self.sheep_v=5
        self.dog_v=10
        self.state=np.array([0.0,0.0,0.0])
        self.eps=0.5
        self.lr=0.05
        self.pi=np.pi
        pos=np.array([self.r,self.r,self.pi],dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.pi,
            high=self.pi, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-pos,
            high=pos,
            dtype=np.float32
        )


    def get_reward(self):
        if self.sheep_x**2+self.sheep_y**2<self.r**2:
            return 1

        dx=self.r*np.cos(self.dog_theta)-self.sheep_x
        dy=self.r*np.sin(self.dog_theta)-self.sheep_y
        d=np.sqrt(dx**2+dy**2)
        print(d)
        if d<=self.eps:
            return -500
        else:
            return 500

    def reset(self):
        r = np.random.uniform(-self.r,self.r)
        sheep_theta=np.random.uniform(-self.pi,self.pi)
        self.dog_theta=np.random.uniform(0,2*self.pi)
        self.sheep_x=r*np.cos(sheep_theta)
        self.sheep_y=r*np.sin(sheep_theta)
        # self.sheep_y=0
        # self.sheep_x=0
        # self.dog_theta=self.pi
        self.state = np.array([self.sheep_x,self.sheep_y,self.dog_theta])
        return self.state

    def trans_angle(self,x,y):
        pi=3.1415926
        if x>0:
            return np.arctan(y/x)
        elif x<0:
            return pi+np.arctan(y/x)
        elif y>0:
            return pi/2
        elif y<0:
            return pi*1.5
        else:
            return 0
    def get_dog_delta(self,state):
        pi=self.pi
        delta=self.dog_v * self.lr / self.r
        x=state[0]
        y=state[1]
        sheep_theta=self.trans_angle(x,y)
        if self.dog_theta<sheep_theta:
            if sheep_theta-self.dog_theta<pi:
                return delta
            else:
                return -delta
        elif self.dog_theta>sheep_theta:
            if self.dog_theta-sheep_theta<pi:
                return -delta
            else:
                return delta
        else:
            return 0
    def is_terminal(self,state):
        temp_x=state[0]
        temp_y=state[1]
        return temp_x**2+temp_y**2>=self.r**2

    def step(self, action):
        # 系统当前状态
        state = self.state
        if self.is_terminal(state):
            reward=self.get_reward()
            return state, reward, True, {}
        print(state)
        # 状态转移
        # print(action)
        next_sheep_x=self.sheep_x+self.lr*self.sheep_v*np.cos(action[0])
        next_sheep_y=self.sheep_y+self.lr*self.sheep_v*np.sin(action[0])
        next_dog_theta=self.dog_theta+self.get_dog_delta(state)
        if next_dog_theta>=self.pi*2:
            next_dog_theta-=self.pi*2
        if next_dog_theta<0:
            next_dog_theta+=self.pi*2
        print(self.dog_theta)
        next_state=np.array([next_sheep_x,next_sheep_y,next_dog_theta])
        self.state = next_state
        self.sheep_x=next_sheep_x
        self.sheep_y=next_sheep_y
        self.dog_theta=next_dog_theta
        terminal = False
        if self.is_terminal(next_state):
            terminal = True
        reward=self.get_reward()
        # print(reward)
        return next_state, reward, terminal, {}

    def render(self, mode='human', close=False):
        circle = rendering.make_circle(self.r,filled=False)  # 3 *注意下面还做了平移操作
        sheep = rendering.make_circle(5)
        dog = rendering.make_circle(5)
        sheep.set_color(0,1,0)
        dog.set_color(1,0,0)
        circle.set_color(0, 0, 0)
        circle.set_linewidth(1)  # 设置线宽

        # 添加一个平移操作
        transform_circle = rendering.Transform(translation=(400, 300))  # 相对偏移
        transform_sheep  = rendering.Transform(translation=(400+self.sheep_x, 300+self.sheep_y))  # 羊的
        transform_dog    = rendering.Transform(translation=(400+self.r*np.cos(self.dog_theta), 300+self.r*np.sin(self.dog_theta)))  # 狗的
        # 让圆添加平移这个属性
        circle.add_attr(transform_circle)
        sheep.add_attr(transform_sheep)
        dog.add_attr(transform_dog)

        self.viewer.add_onetime(circle)
        self.viewer.add_onetime(sheep)
        self.viewer.add_onetime(dog)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()

if __name__ == '__main__':
    t = dog_sheep()
    while True:
        t.render()