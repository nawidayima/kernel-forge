#!POPCORN leaderboard vectoradd
#!POPCORN gpu T4

from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    A, B = data
    return A + B
