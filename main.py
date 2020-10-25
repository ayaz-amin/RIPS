import torch 
import torch.optim as optim

from model.model import Model
from schema_games.breakout.games import StandardBreakout

model = Model(input_channels=3, object_types=5, action_types=3, num_programs=10)

env = StandardBreakout(return_state_as_image=True)

obs = env.reset()

for epoch in range(100):
    while True:
        env.render()
        action_dist = model(obs)
        #action = env.action_space.sample()
        obs, reward, done, _ = env.step(action_dist.sample().item())

        print(reward)

        if done:
            break
