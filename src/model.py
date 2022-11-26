import math
import timeit
import numpy as np
# import tensorflow as tf
import tensorflow._api.v2.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import multiprocessing as mp
from dataset import KnowledgeGraph
import torch
import torch.nn as nn

class TransE:
    def __init__(self, kg: KnowledgeGraph,
                 embedding_dim, margin_value, score_func,
                 batch_size, lr, n_generator, n_rank_calculator):
        self.kg = kg #就是KnowledgeGraph知识图谱
        self.embedding_dim = embedding_dim #就是embedding 的 维度，也就是头实体，关系，尾实体的维度
        self.margin_value = margin_value #
        self.score_func = score_func #打分函数
        self.batch_size = batch_size
        self.lr = lr
        self.n_generator = n_generator
        self.n_rank_calculator = n_rank_calculator
        '''ops for training'''
        self.triple_pos = tf.placeholder(dtype=tf.int32, shape=[None, 3])#占位符 placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存
        self.triple_neg = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        self.margin = tf.placeholder(dtype=tf.float32, shape=[None])
        self.train_op = None
        self.loss = None
        self.global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        self.merge = None
        '''ops for evaluation'''
        self.eval_triple = tf.placeholder(dtype=tf.int32, shape=[3])
        self.idx_head_prediction = None
        self.idx_tail_prediction = None
        '''embeddings'''
        bound = 6 / math.sqrt(self.embedding_dim) # 这里就对应着论文里面的 l = uniform(-6/sqrt(k),6/sqrt(k))for each entity l 属于 L
        with tf.variable_scope('embedding'):
            self.entity_embedding = tf.get_variable(name='entity',
                                                    shape=[kg.n_entity, self.embedding_dim],
                                                    initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                              maxval=bound))
            # entity就是那个实体变量，relation就是关系。
            tf.summary.histogram(name=self.entity_embedding.op.name, values=self.entity_embedding)
            self.relation_embedding = tf.get_variable(name='relation',
                                                      shape=[kg.n_relation, self.embedding_dim],
                                                      initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                maxval=bound))
            tf.summary.histogram(name=self.relation_embedding.op.name, values=self.relation_embedding)#显示直方图信息，用来显示训练过程中变量的分布情况
        self.build_graph()
        self.build_eval_graph()

    def build_graph(self):
        with tf.name_scope('normalization'):
            '''文章中强调的norm'''
            self.entity_embedding = tf.nn.l2_normalize(self.entity_embedding, dim=1)
            self.relation_embedding = tf.nn.l2_normalize(self.relation_embedding, dim=1)
        with tf.name_scope('training'):
            '''这一步初始化（调用）一些基本的步骤包括dis，loss，优化器等'''
            distance_pos, distance_neg = self.infer(self.triple_pos, self.triple_neg)
            self.loss = self.calculate_loss(distance_pos, distance_neg, self.margin)
            tf.summary.scalar(name=self.loss.op.name, tensor=self.loss)# 用来显示标量信息 画loss,accuary时会用到这个函数。
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr) #经典adam优化器，2015年提出，也就是最优化里面提出的找寻最优解
            #optimizer = tf.train.sgd(lr = self.lr) #这个是当年2013年最强的优化器，也是论文中用的
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
            self.merge = tf.summary.merge_all()

    def build_eval_graph(self):
        with tf.name_scope('evaluation'):
            self.idx_head_prediction, self.idx_tail_prediction = self.evaluate(self.eval_triple)

    def infer(self, triple_pos, triple_neg):
        with tf.name_scope('lookup'):
            head_pos = tf.nn.embedding_lookup(self.entity_embedding, triple_pos[:, 0])
            tail_pos = tf.nn.embedding_lookup(self.entity_embedding, triple_pos[:, 1])
            relation_pos = tf.nn.embedding_lookup(self.relation_embedding, triple_pos[:, 2])
            head_neg = tf.nn.embedding_lookup(self.entity_embedding, triple_neg[:, 0])
            tail_neg = tf.nn.embedding_lookup(self.entity_embedding, triple_neg[:, 1])
            relation_neg = tf.nn.embedding_lookup(self.relation_embedding, triple_neg[:, 2])
        with tf.name_scope('link'):
            distance_pos = head_pos + relation_pos - tail_pos
            distance_neg = head_neg + relation_neg - tail_neg
        return distance_pos, distance_neg

    def calculate_loss(self, distance_pos, distance_neg, margin):
        '''
        loss function
        论文里面的损失函数是Hinge Loss Function 具体公式就如笔记所示，使用的原因是为了讲正和负尽可能的分开。一般的话 margin = 1
        分别使用L1或者L2loss进行衡量预测和真值之间的差别，
        L1：绝对值 L2：平方 这两个应该就是表示的是论文loss function中的d那一项。
        pos字面意思就是正，neg就是负。
        然后论文中的[]+通过relu函数实现，具体的relu函数就是一个小于0的时候是0，大于0的时候是因变量等于自变量的函数
        最后就sum一下，loss函数就完成了！'''
        with tf.name_scope('loss'):
            if self.score_func == 'L1':  # L1 score
                score_pos = tf.reduce_sum(tf.abs(distance_pos), axis=1)
                score_neg = tf.reduce_sum(tf.abs(distance_neg), axis=1)
            else:  # L2 score
                score_pos = tf.reduce_sum(tf.square(distance_pos), axis=1)
                score_neg = tf.reduce_sum(tf.square(distance_neg), axis=1)
            loss = tf.reduce_sum(tf.nn.relu(margin + score_pos - score_neg), name='max_margin_loss')
        return loss

    def evaluate(self, eval_triple):
        '''评价函数：
        '''
        with tf.name_scope('lookup'):
            head = tf.nn.embedding_lookup(self.entity_embedding, eval_triple[0])#选取一个张量里面索引对应的元素，params可以是张量也可以是数组等，id就是对应的索引
            tail = tf.nn.embedding_lookup(self.entity_embedding, eval_triple[1])
            relation = tf.nn.embedding_lookup(self.relation_embedding, eval_triple[2])
        with tf.name_scope('link'):
            # head +relation = tail   dis_head_pre = head_pre - head
            distance_head_prediction = self.entity_embedding + relation - tail
            distance_tail_prediction = head + relation - self.entity_embedding
        with tf.name_scope('rank'):
            if self.score_func == 'L1':  # L1 score
                _, idx_head_prediction = tf.nn.top_k(tf.reduce_sum(tf.abs(distance_head_prediction), axis=1),
                                                     k=self.kg.n_entity)#找到输入的张量的最后的一个维度的最大的k个值和它的下标
                _, idx_tail_prediction = tf.nn.top_k(tf.reduce_sum(tf.abs(distance_tail_prediction), axis=1),
                                                     k=self.kg.n_entity)
            else:  # L2 score
                _, idx_head_prediction = tf.nn.top_k(tf.reduce_sum(tf.square(distance_head_prediction), axis=1),
                                                     k=self.kg.n_entity)
                _, idx_tail_prediction = tf.nn.top_k(tf.reduce_sum(tf.square(distance_tail_prediction), axis=1),
                                                     k=self.kg.n_entity)
        return idx_head_prediction, idx_tail_prediction

    def launch_training(self, session, summary_writer):
        raw_batch_queue = mp.Queue()
        training_batch_queue = mp.Queue()
        for _ in range(self.n_generator):
            mp.Process(target=self.kg.generate_training_batch, kwargs={'in_queue': raw_batch_queue,
                                                                       'out_queue': training_batch_queue}).start()
        print('-----Start training-----')
        start = timeit.default_timer()
        n_batch = 0
        for raw_batch in self.kg.next_raw_batch(self.batch_size):
            raw_batch_queue.put(raw_batch)
            n_batch += 1
        for _ in range(self.n_generator):
            raw_batch_queue.put(None)
        print('-----Constructing training batches-----')
        epoch_loss = 0
        n_used_triple = 0
        for i in range(n_batch):
            batch_pos, batch_neg = training_batch_queue.get()
            batch_loss, _, summary = session.run(fetches=[self.loss, self.train_op, self.merge],
                                                 feed_dict={self.triple_pos: batch_pos,
                                                            self.triple_neg: batch_neg,
                                                            self.margin: [self.margin_value] * len(batch_pos)})
            summary_writer.add_summary(summary, global_step=self.global_step.eval(session=session))
            epoch_loss += batch_loss
            n_used_triple += len(batch_pos)
            print('[{:.3f}s] #triple: {}/{} triple_avg_loss: {:.6f}'.format(timeit.default_timer() - start,
                                                                            n_used_triple,
                                                                            self.kg.n_training_triple,
                                                                            batch_loss / len(batch_pos)), end='\r')
        print()
        print('epoch loss: {:.3f}'.format(epoch_loss))
        print('cost time: {:.3f}s'.format(timeit.default_timer() - start))
        print('-----Finish training-----')
        self.check_norm(session=session)

    def launch_test(self, session):
        '''评估：
        正确的实体评分函数的平均排名（mean rank），
        正确的实体排名在前10的比例，即十命中率（hit@10）
        实现：在这个排好序的n-1元素中，我们从第一个开始遍历，看从第一个到第十个是否能够遇到真实的实体，如果遇到了就将（hit@10 +1），这就是hit@10的意义，表示我们的算法能够正确表示三元组关系的能力（在hit@10里  不要求第一个才是对的，能做到前十的能力就可以了）
        而对于mean rank是计算在测试集里，平均到第多少个才能匹配到正确的结果。
        '''
        eval_result_queue = mp.JoinableQueue()
        rank_result_queue = mp.Queue()
        print('-----Start evaluation-----')
        start = timeit.default_timer()
        for _ in range(self.n_rank_calculator):
            mp.Process(target=self.calculate_rank, kwargs={'in_queue': eval_result_queue,
                                                           'out_queue': rank_result_queue}).start()
        n_used_eval_triple = 0
        for eval_triple in self.kg.test_triples:
            # 调用test的3元组数据
            idx_head_prediction, idx_tail_prediction = session.run(fetches=[self.idx_head_prediction,
                                                                            self.idx_tail_prediction],
                                                                   feed_dict={self.eval_triple: eval_triple})
            eval_result_queue.put((eval_triple, idx_head_prediction, idx_tail_prediction))
            n_used_eval_triple += 1
            print('[{:.3f}s] #evaluation triple: {}/{}'.format(timeit.default_timer() - start,
                                                               n_used_eval_triple,
                                                               self.kg.n_test_triple), end='\r')
            #打印：花费多少时间才得到结果，用了多少个命中的，一共有几个test的3元组
        print()
        for _ in range(self.n_rank_calculator):
            eval_result_queue.put(None)
        print('-----Joining all rank calculator-----')
        eval_result_queue.join()
        print('-----All rank calculation accomplished-----')
        print('-----Obtaining evaluation results-----')
        '''Raw'''
        head_meanrank_raw = 0
        head_hits10_raw = 0
        tail_meanrank_raw = 0
        tail_hits10_raw = 0
        '''Filter'''
        head_meanrank_filter = 0
        head_hits10_filter = 0
        tail_meanrank_filter = 0
        tail_hits10_filter = 0
        for _ in range(n_used_eval_triple):
            head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter = rank_result_queue.get()
            head_meanrank_raw += head_rank_raw
            if head_rank_raw < 10:
                head_hits10_raw += 1
            tail_meanrank_raw += tail_rank_raw
            if tail_rank_raw < 10:
                tail_hits10_raw += 1
            head_meanrank_filter += head_rank_filter
            if head_rank_filter < 10:
                head_hits10_filter += 1
            tail_meanrank_filter += tail_rank_filter
            if tail_rank_filter < 10:
                tail_hits10_filter += 1
        print('-----Raw-----')
        head_meanrank_raw /= n_used_eval_triple
        head_hits10_raw /= n_used_eval_triple
        tail_meanrank_raw /= n_used_eval_triple
        tail_hits10_raw /= n_used_eval_triple
        print('-----Head prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(head_meanrank_raw, head_hits10_raw))
        print('-----Tail prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(tail_meanrank_raw, tail_hits10_raw))
        print('------Average------')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format((head_meanrank_raw + tail_meanrank_raw) / 2,
                                                         (head_hits10_raw + tail_hits10_raw) / 2))
        print('-----Filter-----')
        head_meanrank_filter /= n_used_eval_triple
        head_hits10_filter /= n_used_eval_triple
        tail_meanrank_filter /= n_used_eval_triple
        tail_hits10_filter /= n_used_eval_triple
        print('-----Head prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(head_meanrank_filter, head_hits10_filter))
        print('-----Tail prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(tail_meanrank_filter, tail_hits10_filter))
        print('-----Average-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format((head_meanrank_filter + tail_meanrank_filter) / 2,
                                                         (head_hits10_filter + tail_hits10_filter) / 2))
        print('cost time: {:.3f}s'.format(timeit.default_timer() - start))
        print('-----Finish evaluation-----')

    def calculate_rank(self, in_queue, out_queue):
        while True:
            idx_predictions = in_queue.get()
            if idx_predictions is None:
                in_queue.task_done()
                return
            else:
                eval_triple, idx_head_prediction, idx_tail_prediction = idx_predictions
                head, tail, relation = eval_triple
                head_rank_raw = 0
                tail_rank_raw = 0
                head_rank_filter = 0
                tail_rank_filter = 0
                for candidate in idx_head_prediction[::-1]:
                    if candidate == head:
                        break
                    else:
                        head_rank_raw += 1
                        if (candidate, tail, relation) in self.kg.golden_triple_pool:
                            continue
                        else:
                            head_rank_filter += 1
                for candidate in idx_tail_prediction[::-1]:
                    if candidate == tail:
                        break
                    else:
                        tail_rank_raw += 1
                        if (head, candidate, relation) in self.kg.golden_triple_pool:
                            continue
                        else:
                            tail_rank_filter += 1
                out_queue.put((head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter))
                in_queue.task_done()

    def check_norm(self, session):
        print('-----Check norm-----')
        entity_embedding = self.entity_embedding.eval(session=session)
        relation_embedding = self.relation_embedding.eval(session=session)
        entity_norm = np.linalg.norm(entity_embedding, ord=2, axis=1) #求范数
        relation_norm = np.linalg.norm(relation_embedding, ord=2, axis=1)
        print('entity norm: {} relation norm: {}'.format(entity_norm, relation_norm))
