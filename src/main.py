from dataset import KnowledgeGraph
from model import TransE

import tensorflow as tf
import argparse


def main():
    parser = argparse.ArgumentParser(description='TransE')
    parser.add_argument('--data_dir', type=str, default='../data/FB15k/')
    parser.add_argument('--embedding_dim', type=int, default=200)
    parser.add_argument('--margin_value', type=float, default=1.0)
    parser.add_argument('--score_func', type=str, default='L2')#果然还是l2吧
    parser.add_argument('--batch_size', type=int, default=4800)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_generator', type=int, default=24)
    parser.add_argument('--n_rank_calculator', type=int, default=24)
    parser.add_argument('--ckpt_dir', type=str, default='../ckpt/')# 定义模型存储的位置
    parser.add_argument('--summary_dir', type=str, default='../summary/')
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--eval_freq', type=int, default=10) #评估
    args = parser.parse_args()
    print(args)
    kg = KnowledgeGraph(data_dir=args.data_dir)
    kge_model = TransE(kg=kg, embedding_dim=args.embedding_dim, margin_value=args.margin_value,
                       score_func=args.score_func, batch_size=args.batch_size, lr=args.lr,
                       n_generator=args.n_generator, n_rank_calculator=args.n_rank_calculator)
    gpu_config = tf.GPUOptions(allow_growth=True)
    #tf.GPUOptions（）主要用于tensorflow的训练。tensorflow训练时默认占用所有GPU的显存。深度学习代码运行时往往出现多个GPU显存被占满清理。在构造tf.Session()时可通过tf.GPUOptions作为可选配置参数的一部分来显示地指定需要分配的显存比例。
    sess_config = tf.ConfigProto(gpu_options=gpu_config)#tf.ConfigProto一般用在创建session的时候，用来对session进行参数配置。
    with tf.Session(config=sess_config) as sess:
        print('-----Initializing tf graph-----')
        tf.global_variables_initializer().run()
        print('-----Initialization accomplished-----')
        kge_model.check_norm(session=sess)# 先norm一下
        summary_writer = tf.summary.FileWriter(logdir=args.summary_dir, graph=sess.graph) #准备好写入日志
        for epoch in range(args.max_epoch):
            print('=' * 30 + '[EPOCH {}]'.format(epoch) + '=' * 30)
            kge_model.launch_training(session=sess, summary_writer=summary_writer)# 丢进去train
            if (epoch + 1) % args.eval_freq == 0: #每epoch经过eval_freq次后test一下，来验证目前train的成果水平
                kge_model.launch_test(session=sess)


if __name__ == '__main__':
    main()
