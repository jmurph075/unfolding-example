import numpy as np
from unfold import Unfolding


# Toy response matrix (detector x energy bin)
R = np.array([
    [0.25, 0.50, 0.25],
    [0.10, 0.30, 0.60],
])

# Synthetic measurement
N = np.array([1200, 800])

# Flat prior
prior = np.ones(R.shape[1])

test = Unfolding(R, N)
result = test.gravel(prior)

print("Unfolded spectrum:")
print(result)
