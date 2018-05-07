import h5py
import numpy as np

with h5py.File('E://David//学习//Semester3//COMP5329//ass1//Assignment-1-Dataset//Assignment-1-Dataset//train_128.h5', 'r') as H:
    data = np.copy(H['data'])
    np.savetxt('train.csv', data,  delimiter=',')
    #print(data)
    #print(data.view(dtype=np.float32, type=np.matrix))

with h5py.File('E://David//学习//Semester3//COMP5329//ass1//Assignment-1-Dataset//Assignment-1-Dataset//train_label.h5', 'r') as H:
    label = np.copy(H['label'])
    np.savetxt('train_label.csv', label, delimiter=',')

    print(label)

#print(data.keys())