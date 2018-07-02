"""
loss
~~~~

Loss related functions
"""


def simple_undersample(loss, freq):
    """
    Downsample the loss signal by sampling it at every nth time we desire.

    :param loss: The array of loss values
    :param freq: The rate to sample loss. Example 10 would take every 10th value
    """
    new = []
    for i in range(0, len(loss), freq):
        new.append(loss[i])
    # just append the last val for posterity
    if i != len(loss) - 1:
        new.append(loss[i])
    return new
