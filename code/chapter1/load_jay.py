import tensorflow as tf
sess = tf.Session()
filename_q = tf.train.string_input_producer(
    tf.train.match_filenames_once("./data/blue_jay.jpeg")
)
reader = tf.WholeFileReader()
k,v = reader.read(filename_q)
image = tf.image.decode_jpeg(v)
flipImg = tf.image.encode_jpeg(tf.image.flip_up_down(image=image))
flipImgLeftRight = tf.image.encode_jpeg(tf.image.flip_left_right(image))
tf.local_variables_initializer().run(session=sess)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord,sess=sess)
example = sess.run(flipImgLeftRight)

file = open("./data/flipupdown.jpeg","w+")
file.write(flipImg.eval(session=sess))
file.close()

file = open("./data/flipleftright.jpeg","w+")
file.write(flipImgLeftRight.eval(session=sess))
file.close()