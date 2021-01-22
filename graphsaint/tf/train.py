#from graphsaint.globals import *
#from graphsaint.tf.inits import *
from graphsaint.tf.input import *
#from graphsaint.tf.model import GraphSAINT
from graphsaint.tf.model import model_fn
#from graphsaint.tf.minibatch import Minibatch
#from graphsaint.utils import *
#from graphsaint.metric import *
#from tensorflow.python.client import timeline

#import argparse
import sys, os, random
import tensorflow as tf
import numpy as np
import time
import pdb
import json

#class TimeLiner:
#    _timeline_dict = None
#
#    def update_timeline(self, chrome_trace):
#        # convert crome trace to python dict
#        chrome_trace_dict = json.loads(chrome_trace)
#        # for first run store full trace
#        if self._timeline_dict is None:
#            self._timeline_dict = chrome_trace_dict
#        # for other - update only time consumption, not definitions
#        else:
#            for event in chrome_trace_dict['traceEvents']:
#                # events time consumption started with 'ts' prefix
#                if 'ts' in event:
#                    self._timeline_dict['traceEvents'].append(event)
#
#    def save(self, f_name):
#        with open(f_name, 'w') as f:
#            json.dump(self._timeline_dict, f)

#def evaluate_full_batch(sess,model,minibatch_iter,many_runs_timeline,mode):
#    """
#    Full batch evaluation
#    NOTE: HERE GCN RUNS THROUGH THE FULL GRAPH. HOWEVER, WE CALCULATE F1 SCORE
#        FOR VALIDATION / TEST NODES ONLY.
#    """
#    options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
#    run_metadata = tf.compat.v1.RunMetadata()
#    t1 = time.time()
#    num_cls = minibatch_iter.class_arr.shape[-1]
#    feed_dict, labels = minibatch_iter.feed_dict(mode)
#    if args_global.timeline:
#        preds,loss = sess.run([model.preds, model.loss], feed_dict=feed_dict, options=options, run_metadata=run_metadata)
#        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
#        chrome_trace = fetched_timeline.generate_chrome_trace_format()
#        many_runs_timeline.append(chrome_trace)
#    else:
#        preds,loss = sess.run([model.preds, model.loss], feed_dict=feed_dict)
#    node_val_test = minibatch_iter.node_val if mode=='val' else minibatch_iter.node_test
#    t2 = time.time()
#    f1_scores = calc_f1(labels[node_val_test],preds[node_val_test],model.sigmoid_loss)
#    return loss, f1_scores[0], f1_scores[1], (t2-t1)


#########
# TRAIN #
#########
def train(train_phases,model,minibatch,\
            sess,train_stat,ph_misc_stat,summary_writer):
    import time

    # saver = tf.train.Saver(var_list=tf.trainable_variables())
    saver=tf.compat.v1.train.Saver()

    epoch_ph_start = 0
    f1mic_best, e_best = 0, 0
    time_calc_f1, time_train, time_prepare = 0, 0, 0
    options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE,report_tensor_allocations_upon_oom=True)
    run_metadata = tf.compat.v1.RunMetadata()
    many_runs_timeline=[]       # only used when TF timeline is enabled
    for ip,phase in enumerate(train_phases):
        # We normally only have a single phase of training (see README for defn of 'phase').
        # On the other hand, our implementation does support multi-phase training.
        # e.g., you can use smaller subgraphs during initial epochs and larger subgraphs
        #       when closer to convergence. -- This might speed up convergence.
        minibatch.set_sampler(phase)
        num_batches = minibatch.num_training_batches()
        printf('START PHASE {:4d}'.format(ip),style='underline')
        for e in range(epoch_ph_start,int(phase['end'])):
            printf('Epoch {:4d}'.format(e),style='bold')
            minibatch.shuffle()
            l_loss_tr, l_f1mic_tr, l_f1mac_tr, l_size_subg = [], [], [], []
            time_train_ep, time_prepare_ep = 0, 0
            while not minibatch.end():
                t0 = time.time()
                feed_dict, labels = minibatch.feed_dict(mode='train')
                t1 = time.time()
                if args_global.timeline:      # profile the code with Tensorflow Timeline
                    _,__,loss_train,pred_train = sess.run([train_stat[0], \
                            model.opt_op, model.loss, model.preds], feed_dict=feed_dict, \
                            options=options, run_metadata=run_metadata)
                    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    many_runs_timeline.append(chrome_trace)
                else:
                    _,__,loss_train,pred_train = sess.run([train_stat[0], \
                            model.opt_op, model.loss, model.preds], feed_dict=feed_dict, \
                            options=tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom=True))
                t2 = time.time()
                time_train_ep += t2-t1
                time_prepare_ep += t1-t0
                if not minibatch.batch_num % args_global.eval_train_every:
                    f1_mic,f1_mac = calc_f1(labels,pred_train,model.sigmoid_loss)
                    l_loss_tr.append(loss_train)
                    l_f1mic_tr.append(f1_mic)
                    l_f1mac_tr.append(f1_mac)
                    l_size_subg.append(minibatch.size_subgraph)
            time_train += time_train_ep
            time_prepare += time_prepare_ep
            if args_global.cpu_eval:      # Full batch evaluation using CPU
                # we have to start a new session so that CPU can perform full-batch eval.
                # current model params are communicated to the new session via tmp.chkpt
                saver.save(sess,'./tmp.chkpt')
                with tf.device('/cpu:0'):
                    sess_cpu = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(device_count={'GPU': 0}))
                    sess_cpu.run(tf.compat.v1.global_variables_initializer())
                    saver = tf.compat.v1.train.Saver()
                    saver.restore(sess_cpu, './tmp.chkpt')
                    sess_eval=sess_cpu
            else:
                sess_eval=sess
            loss_val,f1mic_val,f1mac_val,time_eval = \
                evaluate_full_batch(sess_eval,model,minibatch,many_runs_timeline,mode='val')
            printf(' TRAIN (Ep avg): loss = {:.4f}\tmic = {:.4f}\tmac = {:.4f}\ttrain time = {:.4f} sec'.format(f_mean(l_loss_tr),f_mean(l_f1mic_tr),f_mean(l_f1mac_tr),time_train_ep))
            printf(' VALIDATION:     loss = {:.4f}\tmic = {:.4f}\tmac = {:.4f}'.format(loss_val,f1mic_val,f1mac_val),style='yellow')
            if f1mic_val > f1mic_best:
                f1mic_best, e_best = f1mic_val, e
                if not os.path.exists(args_global.dir_log+'/models'):
                    os.makedirs(args_global.dir_log+'/models')
                print('  Saving models ...')
                savepath = saver.save(sess, '{}/models/saved_model_{}.chkpt'.format(args_global.dir_log,timestamp),write_meta_graph=False,write_state=False)

            if args_global.tensorboard:
                misc_stat = sess.run([train_stat[1]],feed_dict={\
                                        ph_misc_stat['val_f1_micro']: f1mic_val,
                                        ph_misc_stat['val_f1_macro']: f1mac_val,
                                        ph_misc_stat['train_f1_micro']: f_mean(l_f1mic_tr),
                                        ph_misc_stat['train_f1_macro']: f_mean(l_f1mac_tr),
                                        ph_misc_stat['time_per_epoch']: time_train_ep+time_prepare_ep,
                                        ph_misc_stat['size_subgraph']: f_mean(l_size_subg)})
                # tensorboard visualization
                summary_writer.add_summary(_, e)
                summary_writer.add_summary(misc_stat[0], e)
        epoch_ph_start = int(phase['end'])
    printf("Optimization Finished!",style='yellow')
    timelines = TimeLiner()
    for tl in many_runs_timeline:
        timelines.update_timeline(tl)
    timelines.save('timeline.json')
    saver.restore(sess_eval, '{}/models/saved_model_{}.chkpt'.format(args_global.dir_log,timestamp))
    loss_val, f1mic_val, f1mac_val, duration = evaluate_full_batch(sess_eval,model,minibatch,many_runs_timeline,mode='val')
    printf("Full validation (Epoch {:4d}): \n  F1_Micro = {:.4f}\tF1_Macro = {:.4f}".format(e_best,f1mic_val,f1mac_val),style='red')
    loss_test, f1mic_test, f1mac_test, duration = evaluate_full_batch(sess_eval,model,minibatch,many_runs_timeline,mode='test')
    printf("Full test stats: \n  F1_Micro = {:.4f}\tF1_Macro = {:.4f}".format(f1mic_test,f1mac_test),style='red')
    printf('Total training time: {:6.2f} sec'.format(time_train),style='red')
    #ret = {'loss_val_opt':loss_val,'f1mic_val_opt':f1mic_val,'f1mac_val_opt':f1mac_val,\
    #        'loss_test_opt':loss_test,'f1mic_test_opt':f1mic_test,'f1mac_test_opt':f1mac_test,\
    #        'epoch_best':e_best,
    #        'time_train': time_train}
    return      # everything is logged by TF. no need to return anything

########
# MAIN #
########

def train_main(argv=None):
    train_params,train_phases,train_data,arch_gcn = parse_n_prepare(args_global)
    model,minibatch,sess,train_stat,ph_misc_stat,summary_writer = prepare(train_data,train_params,arch_gcn)
    ret = train(train_phases,model,minibatch,sess,train_stat,ph_misc_stat,summary_writer)
    return ret

def train_main_2(argv=None):
    print("train_main_2")

    args = parse_args()
    params = get_params(args.params, args)

    config = tf.estimator.RunConfig(
        model_dir=params['model_dir'],
        save_checkpoints_steps=1000,
    )
    est = tf.estimator.Estimator(
        model_fn,
        model_dir=params['model_dir'],
        params=params,
        config=config,
    )

    if params['mode'] == 'train':
        est.train(
            input_fn=lambda: input_fn(params, tf.estimator.ModeKeys.TRAIN),
            steps=params['training']['steps']
        )
    elif params['mode'] == 'eval':
        est.evaluate(
            input_fn=lambda: input_fn(params, tf.estimator.ModeKeys.EVAL),
        )
    elif params['mode'] == 'eval_all':
        ckpt_list = tf.train.get_checkpoint_state(
            params['model_dir']
        ).all_model_checkpoint_paths
        for ckpt in ckpt_list:
            est.evaluate(
                input_fn=lambda: input_fn(params, tf.estimator.ModeKeys.EVAL),
                checkpoint_path=ckpt
            )
    else:
        est.compile(
            input_fn,
            validate_only=(params['mode'] == 'validate_only'),
        )

if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    #tf.compat.v1.app.run(main=train_main)
    tf.compat.v1.app.run(main=train_main_2)
