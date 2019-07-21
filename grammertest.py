import numpy as np

a = np.arange(32)
v = a.reshape([4,-1])


print(v.shape)
print(v[1,:].shape)
print(v.shape)
sum = np.sum((v-v[1,:]) ,  axis=-1)
print(v)
print('-----------------------------')
print(v[1,:])
print('-----------------------------')
print(v-v[1,:])
print('-----------------------------')
print('-----------------------------')
print(sum)