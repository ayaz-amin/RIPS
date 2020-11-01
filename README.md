# RIPS
Reward Induced Program Synthesis for sample efficient and zero-shot reinforcement learning. The current version in the ```master branch``` is the most recent version of RIPS.

## Dev Log
RIPS #001 "DOPPLER" (2020-10-24 to 2020-10-25):
- Pros: solved entity extractor that parses environment details into objects.
- Cons: very, very slow (~ 1 minute for each step) due to enumerating over all the cells of the input space.

RIPS #002 "STRANGE MATTER" (2020-10-25 to 2020-10-30):
- Pros: much faster (~ 5 seconds per step) since we now use convolutional filters to downsample high-dimensional data.
- Cons: not much really, the solved entity extractor would have saved training time (which we haven't gotten into yet), but using convolutional filters satisfies real world scenarios, where having a solved vision system is pretty much impossible.

RIPS #003 "RED SHIFT" (2020-10-30):
- Pros: small addition to sub-program, which happens to be the ```not``` logical operator. Should increase the expressiveness of the policy.
- Cons: Same as "STRANGE MATTER" really, nothing much.
