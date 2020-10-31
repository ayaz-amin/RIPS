import torch 
import torch.optim as optim

from model.model import Model
from schema_games.breakout.games import StandardBreakout

model = Model(input_channels=3, object_types=5, action_types=3, num_programs=10)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

env = StandardBreakout(return_state_as_image=True)

obs = env.reset()

while True:
    env.render()
    action_dist = model(obs)
    action = action_dist.sample()
    obs, reward, done, _ = env.step(action.item())

    action_log_prob = -action_dist.log_prob(action) * reward 

    print(action.item())

    if done:
        print("DONE")

