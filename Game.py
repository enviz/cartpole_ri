import gym

class Game:
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.env.reset()
                                       ##https://gym.openai.com/docs/
    def play(self):                    #This will run an instance of the CartPole-v0 environment for 1000 timesteps, rendering the environment at each step.
        for _ in range(1000):               
            self.env.render()
            observation, reward, done, info = self.env.step(self.env.action_space.sample())
            if done:
                break
        self.env.close()
