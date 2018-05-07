import h5py
import numpy as np

with h5py.File('E://David//学习//Semester3//COMP5329//ass1//Assignment-1-Dataset//Assignment-1-Dataset//train_128.h5', 'r') as H:
    data = np.copy(H['data'])
    np.savetxt('test.csv', data, delimiter=',')