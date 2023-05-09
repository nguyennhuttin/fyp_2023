import numpy as np

# uses unbiased std dev like in matlab
def unbiased_std(x):
    result = 0
    mean = np.mean(x)
    for i in range(len(x)):
        result += (x[i]-mean)**2
    return float(np.sqrt(result/(len(x)-1)))

# handles dividing by zero
def divide(x, y):
    if y:
        return x/y
    else:
        return 'NaN'

# allows x to be initialised as np.array([])
def vstack(x, y):
    return np.vstack((x, y)) if x.size else y