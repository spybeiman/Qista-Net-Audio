import numpy as np
def SNR(ref,target):
    target.astype(np.float32)
    ref.astype(np.float32)
    gap = np.sum((target - ref) ** 2)
    ref_power = np.sum(ref ** 2)
    return 10*np.log10(ref_power/gap)