import tensorflow as tf
sess = tf.Session()

# combine multi filenames to one q.
file_name_q = tf.train.string_input_producer(
    tf.train.match_filenames_once("./data/iris.csv"),
    shuffle=True
)
reader = tf.TextLineReader(skip_header_lines=1)
k,v  = reader.read(file_name_q)
record_d = [[0.],[0.],[0.],[0.],[""]]
c1,c2,c3,c4,c5 = tf.decode_csv(v,
                               record_defaults=record_d)
fts = tf.stack([c1,c3,c3,c4])
#tf.global_variables_initializer().run(session=sess)
tf.local_variables_initializer().run(session=sess)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord,sess=sess)
for iter in range(0,5):
    example = sess.run([fts])
    print (example)
coord.request_stop()
coord.join(threads)

