# RIPS
Reward Induced Program Synthesis for sample efficient and zero-shot reinforcement learning.

## Dev Log
RIPS #001 "DOPPLER" (2020-10-24):
- Pros: solved entity extractor that parses environment details into objects.
- Cons: very, very slow (~ 1 minute for each step) due to enumerating over all the cells of the input space.

RIPS #002 "SIDEREAL" (2020-10-25 to now):
- Pros: much faster (~ 5 seconds per step) since we now use convolutional filters to downsample high-dimensional data.
- Cons: not much really, the solved entity extractor would have saved training time (which we haven't gotten into yet), but using convolutional filters satisfies real world scenarios, where having a solved vision system is pretty much impossible.
