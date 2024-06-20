import csv
import os
import tensorflow as tf
import config
import generator
import discriminator
import utils
import time
import numpy as np
# from dblp_evaluation import DBLP_evaluation
# from yelp_evaluation import Yelp_evaluation
# from aminer_evaluation import Aminer_evaluation

# print(tf.__version__)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Model():
    def __init__(self):

        t = time.time()
        print("reading graph...")
        self.node2id, self.total_node, self.need_node, self.n_node, self.n_relation, self.graph = utils.read_dblp_graph(config.graph_filename)
        self.node_list = list(self.graph.keys())#range(0, self.n_node)
        print('[%.2f] reading graph finished. #node = %d #relation = %d' % (time.time() - t, self.n_node, self.n_relation))

        t = time.time()
        print("read initial embeddings...")
        self.node_embed_init_d = np.random.uniform(0.1, 1.0, size=(self.n_node, config.n_emb))
            # utils.read_embeddings(filename=config.pretrain_node_emb_filename_d,
            #                                            n_node=self.n_node,
            #                                            n_embed=config.n_emb)
        self.node_embed_init_g = np.random.uniform(0.1, 1.0, size=(self.n_node, config.n_emb))
        # utils.read_embeddings(filename=config.pretrain_node_emb_filename_g,
        #                                                n_node=self.n_node,
        #                                                n_embed=config.n_emb)

        #self.rel_embed_init_d = utils.read_embeddings(filename=config.pretrain_rel_emb_filename_d,
        #                                              n_node=self.n_node,
        #                                              n_embed=config.n_emb)
        #self.rel_embed_init_g = utils.read_embeddings(filename=config.pretrain_rel_emb_filename_g,
        #                                              n_node=self.n_node,
        #                                              n_embed=config.n_emb)
        print("[%.2f] read initial embeddings finished." % (time.time() - t))

        print("build GAN model...")

        self.discriminator = None
        self.generator = None
        self.build_generator()
        self.build_discriminator()

        # self.latest_checkpoint = tf.train.latest_checkpoint(config.model_log)
        # self.saver = tf.train.Saver()

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        with tf.device('/device:GPU:0'):
            self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            self.sess = tf.Session(config=self.config)
            self.sess.run(self.init_op)


        self.show_config()

    def show_config(self):
        print('--------------------')
        print('Model config : ')
        print('dataset = ', config.dataset)
        print('batch_size = ', config.batch_size)
        print('lambda_gen = ', config.lambda_gen)
        print('lambda_dis = ', config.lambda_dis)
        print('n_sample = ', config.n_sample)
        print('lr_gen = ', config.lr_gen)
        print('lr_dis = ', config.lr_dis)
        print('n_epoch = ', config.n_epoch)
        print('d_epoch = ', config.d_epoch)
        print('g_epoch = ', config.g_epoch)
        print('n_emb = ', config.n_emb)
        print('sig = ', config.sig)
        print('label smooth = ', config.label_smooth)
        print('--------------------')

    def build_generator(self):
        # with tf.variable_scope("generator"):
        with tf.device('/device:GPU:0'):
            self.generator = generator.Generator(n_node = self.n_node,
                                                 n_relation = self.n_relation,
                                                 node_emd_init = self.node_embed_init_g,
                                                 relation_emd_init = None)
    def build_discriminator(self):
        # with tf.variable_scope("discriminator"):
        with tf.device('/device:GPU:0'):
            self.discriminator = discriminator.Discriminator(n_node = self.n_node,
                                                             n_relation = self.n_relation,
                                                             node_emd_init = self.node_embed_init_d,
                                                             relation_emd_init = None)

    def train(self):
        with tf.device('/device:GPU:0'):
            print('start training...')
            for epoch in range(config.n_epoch):
                print('epoch %d' % epoch)
                t = time.time()

                one_epoch_gen_loss = 0.0
                one_epoch_dis_loss = 0.0
                # one_epoch_batch_num = 0.0

                #D-step
                #t1 = time.time()
                for d_epoch in range(config.d_epoch):
                    np.random.shuffle(self.node_list)
                    one_epoch_dis_loss = 0.0
                    one_epoch_pos_loss = 0.0
                    one_epoch_neg_loss_1 = 0.0
                    one_epoch_neg_loss_2 = 0.0

                    for index in range(len(self.node_list) // config.batch_size):
                        #t1 = time.time()
                        pos_node_ids, pos_relation_ids, pos_node_neighbor_ids, neg_node_ids_1, neg_relation_ids_1, neg_node_neighbor_ids_1, neg_node_ids_2, neg_relation_ids_2, node_fake_neighbor_embedding = self.prepare_data_for_d(index)
                        #t2 = time.time()
                        #print t2 - t1
                        _, dis_loss, pos_loss, neg_loss_1, neg_loss_2 = self.sess.run([self.discriminator.d_updates, self.discriminator.loss, self.discriminator.pos_loss, self.discriminator.neg_loss_1, self.discriminator.neg_loss_2],
                                                     feed_dict = {self.discriminator.pos_node_id : np.array(pos_node_ids),
                                                                  self.discriminator.pos_relation_id : np.array(pos_relation_ids),
                                                                  self.discriminator.pos_node_neighbor_id : np.array(pos_node_neighbor_ids),
                                                                  self.discriminator.neg_node_id_1 : np.array(neg_node_ids_1),
                                                                  self.discriminator.neg_relation_id_1 : np.array(neg_relation_ids_1),
                                                                  self.discriminator.neg_node_neighbor_id_1 : np.array(neg_node_neighbor_ids_1),
                                                                  self.discriminator.neg_node_id_2 : np.array(neg_node_ids_2),
                                                                  self.discriminator.neg_relation_id_2 : np.array(neg_relation_ids_2),
                                                                  self.discriminator.node_fake_neighbor_embedding : np.array(node_fake_neighbor_embedding)})

                        one_epoch_dis_loss += dis_loss
                        one_epoch_pos_loss += pos_loss
                        one_epoch_neg_loss_1 += neg_loss_1
                        one_epoch_neg_loss_2 += neg_loss_2

                #G-step

                for g_epoch in range(config.g_epoch):
                    np.random.shuffle(self.node_list)
                    one_epoch_gen_loss = 0.0

                    for index in range(len(self.node_list) // config.batch_size):

                        gen_node_ids, gen_relation_ids, gen_noise_embedding, gen_dis_node_embedding, gen_dis_relation_embedding = self.prepare_data_for_g(index)
                        t2 = time.time()

                        _, gen_loss = self.sess.run([self.generator.g_updates, self.generator.loss],
                                                     feed_dict = {self.generator.node_id :  np.array(gen_node_ids),
                                                                  self.generator.relation_id :  np.array(gen_relation_ids),
                                                                  self.generator.noise_embedding : np.array(gen_noise_embedding),
                                                                  self.generator.dis_node_embedding : np.array(gen_dis_node_embedding),
                                                                  self.generator.dis_relation_embedding : np.array(gen_dis_relation_embedding)})

                        one_epoch_gen_loss += gen_loss

                one_epoch_batch_num = len(self.node_list) // config.batch_size

                #print t2 - t1
                #exit()
                # print('[%.2f] gen loss = %.4f, dis loss = %.4f pos loss = %.4f neg loss-1 = %.4f neg loss-2 = %.4f' % \
                #         (time.time() - t, one_epoch_gen_loss / one_epoch_batch_num, one_epoch_dis_loss / one_epoch_batch_num,
                #         one_epoch_pos_loss / one_epoch_batch_num, one_epoch_neg_loss_1 / one_epoch_batch_num, one_epoch_neg_loss_2 / one_epoch_batch_num))


            print("training completes")

    def prepare_data_for_d(self, index):
        # with tf.device('/device:GPU:0'):
        pos_node_ids = []
        pos_relation_ids = []
        pos_node_neighbor_ids = []

        #real node and wrong relation
        neg_node_ids_1 = []
        neg_relation_ids_1 = []
        neg_node_neighbor_ids_1 = []

        #fake node and true relation
        neg_node_ids_2 = []
        neg_relation_ids_2 = []
        node_fake_neighbor_embedding = None


        for node_id in self.node_list[index * config.batch_size : (index + 1) * config.batch_size]:
            for i in range(config.n_sample):

                # sample real node and true relation
                relations = list(self.graph[node_id].keys())
                relation_id = relations[np.random.randint(0, len(relations))]
                neighbors = self.graph[node_id][relation_id]
                node_neighbor_id = neighbors[np.random.randint(0, len(neighbors))]

                pos_node_ids.append(node_id)
                pos_relation_ids.append(relation_id)
                pos_node_neighbor_ids.append(node_neighbor_id)

                #sample real node and wrong relation
                neg_node_ids_1.append(node_id)
                neg_node_neighbor_ids_1.append(node_neighbor_id)
                neg_relation_id_1 = np.random.randint(0, self.n_relation)
                while neg_relation_id_1 == relation_id:
                    neg_relation_id_1 = np.random.randint(0, self.n_relation)
                neg_relation_ids_1.append(neg_relation_id_1)

                #sample fake node and true relation
                neg_node_ids_2.append(node_id)
                neg_relation_ids_2.append(relation_id)

        # generate fake node
        noise_embedding = np.random.normal(0.0, config.sig, (len(neg_node_ids_2), config.n_emb))

        node_fake_neighbor_embedding = self.sess.run(self.generator.node_neighbor_embedding,
                                                     feed_dict = {self.generator.node_id : np.array(neg_node_ids_2),
                                                                  self.generator.relation_id : np.array(neg_relation_ids_2),
                                                                  self.generator.noise_embedding : np.array(noise_embedding)})

        return pos_node_ids, pos_relation_ids, pos_node_neighbor_ids, neg_node_ids_1, neg_relation_ids_1, \
               neg_node_neighbor_ids_1, neg_node_ids_2, neg_relation_ids_2, node_fake_neighbor_embedding

    def prepare_data_for_g(self, index):
        # with tf.device('/device:GPU:0'):
        node_ids = []
        relation_ids = []

        for node_id in self.node_list[index * config.batch_size : (index + 1) * config.batch_size]:
            for i in range(config.n_sample):
                relations = self.graph[node_id].keys()
                relation_id = relations[np.random.randint(0, len(relations))]

                node_ids.append(node_id)
                relation_ids.append(relation_id)

        noise_embedding = np.random.normal(0.0, config.sig, (len(node_ids), config.n_emb))

        dis_node_embedding, dis_relation_embedding = self.sess.run([self.discriminator.pos_node_embedding, self.discriminator.pos_relation_embedding],
                                                                    feed_dict = {self.discriminator.pos_node_id : np.array(node_ids),
                                                                                 self.discriminator.pos_relation_id : np.array(relation_ids)})
        return node_ids, relation_ids, noise_embedding, dis_node_embedding, dis_relation_embedding


    # def evaluate_dblp_link_prediction(self):
    #     modes = [self.generator, self.discriminator]
    #
    #     for i in range(2):
    #         embedding_matrix = self.sess.run(modes[i].node_embedding_matrix)
    #         #relation_matrix = self.sess.run(modes[i].relation_embedding_matrix)
    #
    #         auc, f1, acc = self.dblp_evaluation.evaluation_link_prediction(embedding_matrix)
    #
    #         print('auc = %.4f f1 = %.4f acc = %.4f' % (auc, f1, acc))



    def write_embeddings_to_file(self, epoch=20):
        # with tf.device('/device:GPU:0'):
        modes = [self.generator, self.discriminator]
        for i in range(2):
            embedding_matrix = self.sess.run(modes[i].node_embedding_matrix)
            index = np.array(self.total_node).reshape(-1, 1)
            embedding_matrix = np.hstack([index, embedding_matrix])
            embedding_list = embedding_matrix.tolist()
            # embedding_str = [str(emb[0]) + ' ' + ' '.join([str(x) for x in emb[1:]]) + '\n' for emb in embedding_list]
            print(self.need_node)
            with open(config.emb_filenames[i], 'w') as csv_file:
                # lines = [str(self.n_node) + ' ' + str(config.n_emb) + '\n'] + embedding_str
                # f.writelines(embedding_str)
                writer = csv.writer(csv_file)
                for emb in embedding_list:
                    if int(emb[0]) in self.need_node:
                        sub_name = utils.get_dict_key(self.node2id, int(emb[0]))
                        patent = sub_name.replace('/m/01 ', '')
                        embedding = [float(item) for item in emb[1:]]
                        writer.writerow([patent, embedding])


if __name__ == '__main__':
    # init_op = tf.global_variables_initializer()
    #
    # with tf.Session() as sess:
    #     with tf.device('/gpu:0'):
    #         sess.run(init_op)
    with tf.device('/gpu:0'):
        model = Model()
        model.train()
        model.write_embeddings_to_file()


