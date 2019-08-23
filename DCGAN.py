import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ['CUDA_VISIBLE_DEVICES']='4'

mnist=input_data.read_data_sets('dataset/mnist',reshape=False)

epochs=100
batch_size=100
learning_rate=0.0002
image_size=28**2
latent_num=100

X_train=mnist.train.images
X_train=np.reshape(X_train,[len(X_train),28,28,1])
X_train=(X_train-0.5)/0.5
print(np.min(X_train))

class Model:
    def __init__(self):
        tf.reset_default_graph()

        self.is_training=True
        self.z=tf.placeholder(tf.float32,(None, latent_num))
        self.x_T=tf.placeholder(tf.float32,(None, 28,28,1))

        self.x_F=self.generator(self.z)

        F_out=self.discriminator(self.x_F)
        T_out=self.discriminator(self.x_T)

        self.g_loss=-tf.reduce_mean(tf.log(F_out+1e-8))
        self.d_loss=-(tf.reduce_mean(tf.log(T_out+1e-8))+tf.reduce_mean(tf.log(1e-8+1-F_out)))

        train_vars=tf.trainable_variables()

        gen=[var for var in train_vars if 'gen' in var.name]
        dis=[var for var in train_vars if 'dis' in var.name]

        u_ops_g=tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='gen')
        u_ops_d=tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='dis')

        with tf.control_dependencies(u_ops_g):
            self.g_optimzer=tf.train.AdamOptimizer(learning_rate).minimize(self.g_loss,var_list=gen)
        with tf.control_dependencies(u_ops_d):
            self.d_optimzer=tf.train.AdamOptimizer(learning_rate).minimize(self.d_loss,var_list=dis)


def generator(self,input_):
    with tf.variable_scope('gen', reuse=tf.AUTO_REUSE):
        w_init = tf.contrib.layers.xavier_initializer()
        b_init = tf.contrib.layers.xavier_initializer()

        fc1=tf.layers.dense(input_, 7*7*128 , activation=tf.nn.sigmoid, kernel_initializer=w_init,
                bias_initializer=b_init)
        fc1=tf.reshape(fc1,[-1,7,7,128])
        fc2=tf.layers.conv2d_transpose(fc1,filters=64, kernel_size=5, activation=None, strides=(2,2),
                             padding='SAME', kernel_initializer=w_init, bias_initializer=b_init)
        fc2=tf.layers.batch_normalization(fc2,training=self.is_training)
        fc2=tf.nn.relu(fc2)
        fc3=tf.layers.conv2d_transpose(fc2,filters=32,kernel_size=5, activation=None, strides=(2,2),
                             padding='SAME', kernel_initializer=w_init, bias_initializer=b_init)
        fc3=tf.layers.batch_normalization(fc3,training=self.is_training)
        fc3=tf.nn.relu(fc3)
        fc4=tf.layers.conv2d_transpose(fc3,filters=1,kernel_size=5, activation=tf.nn.tanh, strides=(1,1),
                             padding='SAME', kernel_initializer=w_init, bias_initializer=b_init)

        return fc4

def discriminator(self, input_):
    with tf.variable_scope('dis', reuse=tf.AUTO_REUSE):
        w_init = tf.contrib.layers.xavier_initializer()
        b_init = tf.contrib.layers.xavier_initializer()

        fc1=tf.layers.conv2d(input_,filters=32, kernel_size=5, activation=tf.nn.leaky_relu,
                    strides=(2,2),padding='SAME', kernel_initializer=w_init, bias_initializer=b_init)
        fc2=tf.layers.conv2d(fc1,filters=64,kernel_size=5, activation=None, strides=(2,2),
                             padding='SAME', kernel_initializer=w_init, bias_initializer=b_init)
        fc2=tf.layers.batch_normalization(fc2,training=self.is_training)
        fc2=tf.nn.leaky_relu(fc2)
        fc3=tf.layers.conv2d(fc2,filters=128,kernel_size=5, activation=None, strides=(2,2),
                             padding='SAME', kernel_initializer=w_init, bias_initializer=b_init)
        fc3=tf.layers.batch_normalization(fc3,training=self.is_training)
        fc3=tf.nn.leaky_relu(fc3)
        fc3=tf.contrib.layers.flatten(fc3)
        TF=tf.layers.dense(fc3, 1, activation=tf.nn.sigmoid, kernel_initializer=w_init,
                                bias_initializer=b_init)
        return TF


if __name__=='__main__':

    GAN=Model()

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))
    sess = tf.Session(config=config)
    Saver=tf.train.Saver()

    init=tf.global_variables_initializer()
    sess.run(init)

    len_data=len(X_train)

    for i in range(epochs):
        np.random.shuffle(X_train)

        for offset in range(0,len_data,batch_size):
            end=offset+batch_size
            batch_x=X_train[offset:end]
            batch_z=np.random.normal(size=[batch_size,latent_num])

            sess.run(GAN.g_optimzer, feed_dict={GAN.z: batch_z})
            sess.run(GAN.d_optimzer, feed_dict={GAN.z: batch_z , GAN.x_T: batch_x})


        zz=np.random.normal(size=[10,latent_num])
        xx=sess.run(GAN.x_F,feed_dict={GAN.z:zz})

        xx=(xx+1)/2
        fig, ax = plt.subplots(1, 10, figsize=(10, 1))

        for i in range(10) :
            ax[i].set_axis_off()
            ax[i].imshow(np.reshape( xx[i], (28, 28)) ,cmap='gray')
        plt.show()

        Saver.save(sess,"./DCGAN/data")

    with tf.Session(config=config) as sess:
    Saver.restore(sess, "./DCGAN/data")

    GAN.is_training=False
    f, ax = plt.subplots(10, 10, figsize=(20, 20))
    f.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)

    for i in range(10):

        for j in range(10):
            zz=np.random.normal(size=[10,latent_num])

            xx=sess.run(GAN.x_F,feed_dict={GAN.z:zz})

            ax[i][j].imshow(np.reshape(xx[i], (28, 28)) ,cmap='gray')
            ax[i][j].axis("off")

    plt.show()
