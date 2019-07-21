import os
import tensorflow.compat.v1 as tf
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.disable_v2_behavior()

# print(5)

# #import matpoltlib.pyplot as plt
# num_points = 100
# vector_set =[]
# t1  = tf.constant(2)
# t2  = tf.constant(3)
# for i in range(num_points):
#     x1  = np.random.normal(0.0, 0.55)
#     y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
#     vector_set.append([x1,y1])
#
# x_data = [v[0] for v in vector_set]
# y_data = [v[1] for v in vector_set]
# print(7)
# listA = [2,4,3,1,5,7,6,8,9]
# print([x for x in listA])
#
# print('************************')
#
# W = tf.Variable(tf.random_uniform([1] , -1.0, -1.0))
# b = tf.Variable(tf.zeros([1]) )
# y = W * x_data+b
#
# #print(W.shape)
# #print (b.shape)
# #print(y.shape)
#
# loss = tf.reduce_mean(tf.square(y - y_data))
# sess = tf.Session()
# #print(sess.run(loss))
# #print(type(x_data[0]))
# #print(type(W))
#
# optimizer = tf.train.GradientDescentOptimizer(0.5)
# train = optimizer.minimize(loss)
#
# init = tf.initialize_all_variables()
#
# sess.run(init)
#
# for step in range(80):
#     sess.run(train)
#
#     print( step , sess.run(W) , sess.run(b) , "   === >   " , sess.run(loss))
#
#
#

# vector_list = []
# loop_num = 1000
# for step in range(loop_num):
#     x = np.random.normal(0.0,0.55)
#     y = 2 *x + 3 + np.random.normal(0.0 , 0.2)
#     vector_list.append([x,y])
#
# x_data = [v[0] for v in vector_list]
# y_data = [v[1] for v in vector_list]
#
# W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# b = tf.Variable(tf.zeros([1]))
# y = W*x_data + b
#
# loss = tf.reduce_mean(tf.square(y-y_data))
#
# sess= tf.Session()
# sess.run(tf.initialize_all_variables())
#
# optimizer = tf.train.GradientDescentOptimizer(0.5)
# train = optimizer.minimize(loss)
#
# for step in range(loop_num):
#     sess.run(train)
#     print(sess.run(loss))
#

class NearestNeighbor:
    def __init__(self):
        pass

    def train(self , x_train , y_train):
        self.xTr = x_train
        self.yTr = y_train

    def predict(self, X_test):
        num_test = X_test .shape[0]   # X 는 test_img  x length  개 임

        Ypred    = np.zeros(num_test , dtype = self.yTr.dtype)

        for i in range(num_test):  # xrange python 2 , range python 3
            distances = np.sum(np.abs(self.xTr - X_test[i,:]) , axis = 1)
            min_index = np.argmin(distances)
            Ypred[i]  = self.yTr[min_index]

        return Ypred
def unpickle(filename):
    import pickle
    with open(filename,'rb') as fo:
        dict = pickle.load(fo,encoding='bytes')
    return dict
def load_CIFAR_batch( filename ): # 배치 파일 1개를 읽어서 가지고온다
    with open(filename,'rb') as f:
        data_dict = unpickle(filename)
        X = data_dict[b'data']
        Y = data_dict[b'labels']
        return X,Y
def load_CIFAR10_All(batch_dir):
    xs = []
    ys = []
    for idx in range(1,6):
        f = os.path.join( batch_dir , 'data_batch_{idx}'.format(idx=idx) )
        data,label = load_CIFAR_batch(f)
        xs.append(data)
        ys.append(label)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del data,label
    Xte, Yte = load_CIFAR_batch(os.path.join(batch_dir , 'test_batch') )
    return Xtr,Ytr,Xte,Yte

def load_CIFAR10_data(batch_dir ,nTraing = 49000, nValidation = 1000, nTest=10000):
    X_train , Y_train, X_test, Y_text = load_CIFAR10_All(batch_dir)
    x_train = X_train[:nTraing]
    y_train = Y_train[:nTraing]
    x_val   = X_train[nTraing: nTraing + nValidation]
    y_val   = Y_train[nTraing: nTraing + nValidation]
    x_test  = X_test[:nTest]
    y_test  = Y_text[:nTest]
    return x_train,y_train,x_val,y_val,x_test,y_test
base_dir = os.getcwd()
batch_folder = "cifar-10-batches-py"
batch_filename ="data_batch_1"
batch_name = os.path.join(batch_folder, batch_filename)

#dictX,dictY = load_CIFAR_batch(os.path.join(base_dir, batch_name))
X_train,Y_train,X_val,Y_val,X_test,Y_test = load_CIFAR10_data(os.path.join(base_dir, batch_folder))

print(X_train.shape)
print(type(X_train))

t = NearestNeighbor()
print(1)
t.train(X_train,Y_train)
ans = t.predict(X_test)
print(2)
print(ans)
print(ans.shape)

print(3)


