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

def sklearn_gen3():
    centers = [(-2,2),(-2,1.5),(1.5,-2),(2,1.5)]
    data,features  = make_blobs(n_samples=200,centers=centers,n_features=
                                2,cluster_std=0.8,shuffle=False,random_state=42)
    fig,ax = plt.subplots()
    #ax.scatter(np.asarray(centers).transpose()[0],
    #           np.asarray(centers).transpose()[1],marker='o',s=250)
    #plt.show()
    N = 200
    K = 2
    points = tf.Variable(data)
    cluster_assignments = tf.Variable(tf.zeros([N],dtype=tf.int64))
    centroids = tf.Variable(tf.slice(points.initialized_value(),[0,0],[K,2]))
    #ax.scatter(np.asarray(centers).transpose()[0],
    #           np.asarray(centers).transpose()[1],marker='o',s=25)
    #plt.show()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    n_centroids = tf.tile(centroids,[N,1])
    rep_centroids = tf.reshape(tf.tile(centroids,[N,1]),[N,K,2])

    rep_points = tf.reshape(tf.tile(points,[1,K]),[N,K,2])
    sum_squares = tf.reduce_sum(tf.square(rep_points-rep_centroids),
                                reduction_indices=2)
    best_centroids = tf.argmin(sum_squares,1)
    did_assignments_chane = tf.reduce_any(tf.not_equal(best_centroids,
                                                       cluster_assignments))

    def bucket_mean(data,bucket_ids,num_buckets):
        total = tf.unsorted_segment_sum(data, bucket_ids, num_buckets)
        count = tf.unsorted_segment_sum(tf.ones_like(data),
                                        bucket_ids,num_buckets)
        return total/count
    means = bucket_mean(points,best_centroids,K)
    with tf.control_dependencies([did_assignments_chane]):
        do_updates = tf.group(
            centroids.assign(means),
            cluster_assignments.assign(best_centroids)
        )



if __name__ == "__main__":

    sklearn_gen3()
        #plt.savefig("result.png")