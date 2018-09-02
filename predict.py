#!/bin/python3
import tensorflow as tf
import training
# import train_nn_TrainingAugmentation
import readData
import os

IMAGE_SIZE = 224
BATCH_SIZE = 794
with open('label.txt') as f:
    con = f.read()
class_ = con.splitlines()
label = {}
for i in range(len(class_)):
    a, b = class_[i].split('#')
    label[int(a)] = b
print(label)
class0 = os.listdir('G:/datasets/all/test/test/')
# keep_prob = 1
with tf.Session() as sess:
#     for i in class0:
#         path = 'G:/datasets/all/test/test/' + i
#         file = tf.read_file(path)
#         image = tf.image.decode_png(file, 3)
#         image = tf.image.convert_image_dtype(image, dtype = tf.float32)
#         image_test = tf.image.resize_images(image, [IMAGE_SIZE, IMAGE_SIZE])
#         image_test = tf.reshape(image_test, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])
#         img_name = i
    keep_prob = tf.placeholder(tf.float32)
    img_test, img_name = readData.Read_Test_TFRecords('InputData_Test_data*', IMAGE_SIZE)
    img_test_batch, img_name_batch = tf.train.batch([img_test, img_name], batch_size = BATCH_SIZE)
    keep_prob = tf.placeholder(tf.float32)
#         logits = train_nn_TrainingAugmentation.Model(img_test_batch, keep_prob)
    logits = training.Model(img_test_batch, keep_prob)
    pred = tf.argmax(tf.nn.softmax(logits), 1)
    saver = tf.train.Saver()
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)
    # results04
#     ckpt = tf.train.get_checkpoint_state('./Model_Saver04_TrainingDataAugmentation')
# #     saver.restore(sess, ckpt.model_checkpoint_path)
# #     saver.restore(sess, ''./Model_Saver04_TrainingDataAugmentation)
#     saver = tf.train.import_meta_graph('./Model_Saver04_TrainingDataAugmentation/model_save.ckpt-18.meta')
    saver = tf.train.import_meta_graph('./Model_Saver04_TrainingDataAugmentation/model_save.ckpt-18.meta')
    saver.restore(sess, './Model_Saver04_TrainingDataAugmentation/model_save.ckpt-18')
#     saver.restore(sess, ckpt.model_checkpoint_path)
    #     saver.restore(sess, './Model_Saver04_TrainingDataAugmentation')
        # results03
    #     saver = tf.train.import_meta_graph('./Model_Saver01/model_save.ckpt.meta')
    #     saver.restore(sess, './Model_Saver01/model_save.ckpt')
        # results02
        #saver = tf.train.import_meta_graph('./Model_Saver05_Final_Augmentation/model_save.ckpt-15.meta')
        #saver.restore(sess, './Model_Saver05_Final_Augmentation/model_save.ckpt-15')
        # results01
        #saver = tf.train.import_meta_graph('./Model_Saver06_Final_Augmentation/model_save.ckpt-13.meta')
        #saver.restore(sess, './Model_Saver06_Final_Augmentation/model_save.ckpt-13')
    with open('sample_submission.csv', 'w') as fw:
        fw.write('species\n')
        for i in range(1):
#             print('----------------------------------------')
            a = sess.run(img_name_batch)
            print(a)
            ans = sess.run(pred, feed_dict = {keep_prob: 1.0})
#             print("panchenglong pred is %s" % ans)
        #print(img_name_batch.eval())
            for _ in range(len(ans)):
                fw.write(',' + label[ans[_]] + '\n')
            #print(a)
            #print(img_name_batch.eval())
            #print(img_name_batch.eval())
    coord.request_stop()
    coord.join(threads)