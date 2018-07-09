import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles

def random_gen():
    with tf.Session() as sess:
        ax = plt.subplot()
        ax.plot(tf.random_normal([100]).eval(),
                tf.random_normal([100]).eval(),'o')
        ax.set_title('Sample  random  plot for tensorflow')
        plt.show()

def sklearn_gen():
    X,y = make_blobs(n_samples=100,n_features=2,centers=3,cluster_std=1.0,center_box=(-10,10),shuffle=True,
               random_state=None)
    print X
    #X,y = make_circles(n_samples=100,shuffle=True,noise=None,random_state=None,factor=0.8)

    plt.plot(X,'o')
    plt.show()


def sklearn_gen2():
    centers = [(-2,-2),(-2,1.5),(1.5,-2),(2,1.5)]
    X,y = make_blobs(n_samples=200,centers=centers,n_features=2,cluster_std=0.8,shuffle=False,random_state=42)
    plt.plot(np.array(centers).transpose()[0],np.array(centers).transpose()[1],marker='o',s=250)
    plt.show()


if __name__ == "__main__":

    sklearn_gen2()
        #plt.savefig("result.png")