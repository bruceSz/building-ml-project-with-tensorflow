import tensorflow as tf
sess = tf.Session()


file_name_q = tf.train.string_input_producer(
    tf.train.match_filenames_once(".")
)