import os
import tensorflow as tf
import numpy as np
from CustomDoc2Vec.doc2vec import Doc2VecTrainer
from CustomDoc2Vec.dataset import Doc2VecDataset
from CustomDoc2Vec.doc2vec import Doc2VecInferencer

class Doc2VecAdapter():

    def __init__(self, arch='PV-DBOW', algm='negative_sampling', epochs=1, batch_size=64, max_vocab_size=0, min_count=10, sample=0.001, window_size=5, 
                 dbow_train_words=False, dm_concat=True, embed_size=300, negatives=5, power=0.75, alpha=0.025, min_alpha=0.0001, add_bias=True):

        self.Dataset = Doc2VecDataset(arch=arch,
                                algm=algm,
                                epochs=epochs,
                                batch_size=batch_size,
                                max_vocab_size=max_vocab_size,
                                min_count=min_count,
                                sample=sample,
                                window_size=window_size,
                                dbow_train_words=dbow_train_words,
                                dm_concat=dm_concat)


        self.Doc2vec = Doc2VecTrainer(arch=arch,
                           algm=algm,
                           embed_size=embed_size,
                           batch_size=batch_size,
                           negatives=negatives,
                           power=power,
                           alpha=alpha,
                           min_alpha=min_alpha,
                           add_bias=add_bias,
                           random_seed=0,
                           dm_concat=dm_concat,
                           window_size=window_size)

        self.InferDoc2vec = Doc2VecInferencer(arch=arch,
            algm=algm,
            embed_size=embed_size,
            batch_size=batch_size,
            negatives=negatives,
            power=power,
            alpha=alpha,
            min_alpha=min_alpha,
            add_bias=add_bias,
            random_seed=0,
            dm_concat=dm_concat,
            window_size=window_size)

        self.Vocab = None 
        self.Word_embeddings = None
        self.Train_doc_embed = None 
        self.Saveables = None 

    def Fit(self, filenames):

        self.Dataset.build_vocab(filenames)
        to_be_run_dict = self.Doc2vec.train(self.Dataset, filenames)
        save_list = self.Doc2vec.get_save_list() 

        sess = tf.Session()
        sess.run(self.Dataset.iterator_initializer)
        sess.run(tf.tables_initializer())
        sess.run(tf.global_variables_initializer())

        average_loss = 0.
        step = 0
        while True:
            try:
                result_dict = sess.run(to_be_run_dict)
            except tf.errors.OutOfRangeError:
                break
            average_loss += result_dict['loss'].mean()
            if step % 10000 == 0:
                if step > 0:
                    average_loss /= 10000
                print('step', step, 'average_loss', average_loss, 'learning_rate', 
                    result_dict['learning_rate'])
                average_loss = 0.
            step += 1

        syn0_w_final = sess.run(self.Doc2vec.syn0_w)
        syn0_d_final = sess.run(self.Doc2vec.syn0_d)

        self.Saveables = save_list
        self.Word_embeddings = syn0_w_final
        self.Train_doc_embed = syn0_d_final 
        self.Vocab = self.Dataset.table_words

    def Infer(self, filenames):

        to_be_run_dict = self.InferDoc2vec.infer(self.Dataset, filenames)
        save_list = self.InferDoc2vec.get_save_list()

        sess = tf.Session()
        sess.run(self.Dataset.iterator_initializer)
        sess.run(tf.tables_initializer())

        saver = tf.train.Saver(var_list=save_list)
        saver.restore(sess, self.Saveables)
        sess.run(self.InferDoc2vec.syn0_d.initializer)

        average_loss = 0.
        step = 0
        while True:
            try:
                result_dict = sess.run(to_be_run_dict)
            except tf.errors.OutOfRangeError:
                break
            average_loss += result_dict['loss'].mean()
            if step % 10000 == 0:
                if step > 0:
                    average_loss /= 10000
                print('step', step, 'average_loss', average_loss, 'learning_rate', 
                    result_dict['learning_rate'])
                average_loss = 0.
            step += 1

        syn0_d_final = sess.run(self.InferDoc2vec.syn0_d)

        return syn0_d_final