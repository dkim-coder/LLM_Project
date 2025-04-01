import sys
import torch
import numpy as np

def main(args):
    np_array = np.array([1, 2, 3])
    x_data = torch.from_numpy(np_array)
    print(x_data)

if __name__ == "__main__":
    main(sys.argv)
