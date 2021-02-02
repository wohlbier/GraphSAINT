import numpy as np
import tensorflow as tf
import graphsaint.tf.layers as layers
from graphsaint.tf.minibatch import *
#from graphsaint.globals import *
#from graphsaint.tf.inits import *
#from graphsaint.utils import *
#import pdb

class GraphSAINT:

    def __init__(self, num_classes, placeholders, features,
            arch_gcn, train_params, adj_full_norm, **kwargs):
        '''
        Args:
            - placeholders: TensorFlow placeholder object.
            - features: Numpy array with node features.
            - adj: Numpy array with adjacency lists (padded with random re-samples)
            - degrees: Numpy array with node degrees.
            - sigmoid_loss: Set to true if nodes can belong to multiple classes
        '''
        if "attention" in arch_gcn:
            print("using attention aggregator")
            self.aggregator_cls = layers.AttentionAggregator
            self.mulhead = int(arch_gcn['attention'])
        else:
            print("using high order aggregator")
            self.aggregator_cls = layers.HighOrderAggregator
            self.mulhead = 1
        self.lr = train_params['lr']
        self.node_subgraph = placeholders['node_subgraph']
        self.num_layers = len(arch_gcn['arch'].split('-'))
        self.weight_decay = train_params['weight_decay']
        self.jk = None if 'jk' not in arch_gcn else arch_gcn['jk']
        self.arch_gcn = arch_gcn
        self.adj_subgraph = placeholders['adj_subgraph']
        # adj_subgraph_* are to store row-wise partitioned full graph adj tiles.
        self.adj_subgraph_0=placeholders['adj_subgraph_0']
        self.adj_subgraph_1=placeholders['adj_subgraph_1']
        self.adj_subgraph_2=placeholders['adj_subgraph_2']
        self.adj_subgraph_3=placeholders['adj_subgraph_3']
        self.adj_subgraph_4=placeholders['adj_subgraph_4']
        self.adj_subgraph_5=placeholders['adj_subgraph_5']
        self.adj_subgraph_6=placeholders['adj_subgraph_6']
        self.adj_subgraph_7=placeholders['adj_subgraph_7']
        self.dim0_adj_sub = placeholders['dim0_adj_sub'] #adj_full_norm.shape[0]/8
        self.features = tf.Variable(tf.constant(features, dtype=DTYPE),
                                    trainable=False)
        #self.dualGPU=args_global.dualGPU
        _indices = np.column_stack(adj_full_norm.nonzero())
        _data = adj_full_norm.data
        _shape = adj_full_norm.shape
        with tf.device('/cpu:0'):
            self.adj_full_norm = tf.compat.v1.SparseTensor(_indices,_data,_shape)
        self.num_classes = num_classes
        self.sigmoid_loss = (arch_gcn['loss']=='sigmoid')
        _dims, self.order_layer, self.act_layer, self.bias_layer, \
            self.aggr_layer = parse_layer_yml(arch_gcn,features.shape[1])
        print(_dims)
        print(self.order_layer)
        print(self.act_layer)
        print(self.bias_layer)
        print(self.aggr_layer)
        # get layer index for each conv layer, useful for jk net last layer
        # aggregation
        self.set_idx_conv()
        self.set_dims(_dims)
        self.placeholders = placeholders

        self.optimizer = \
            tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr)

        self.loss = 0
        self.opt_op = None
        self.norm_loss = placeholders['norm_loss']
        self.is_train = placeholders['is_train']

        self.build()

    def set_idx_conv(self):
        idx_conv = np.where(np.array(self.order_layer)>=1)[0]
        idx_conv = list(idx_conv[1:] - 1)
        idx_conv.append(len(self.order_layer)-1)
        _o_arr = np.array(self.order_layer)[idx_conv]
        if np.prod(np.ediff1d(_o_arr)) == 0:
            self.idx_conv = idx_conv
        else:
            self.idx_conv = list(np.where(np.array(self.order_layer)==1)[0])

    def set_dims(self,dims):
        self.dims_feat = [dims[0]] + \
            [((self.aggr_layer[l]=='concat')*self.order_layer[l]+1)*dims[l+1] \
             for l in range(len(dims)-1)]
        self.dims_weight = [(self.dims_feat[l],dims[l+1]) \
                            for l in range(len(dims)-1)]
        print(self.dims_feat)
        print(self.dims_weight)

    def build(self):
        """
        Build the sample graph with adj info in self.sample()
        directly feed the sampled support vectors to tf placeholder
        """
        self.aggregators = self.get_aggregators()
        _outputs_l = self.aggregate_subgraph()
        if self.jk == 'concat':
            _dim_input_jk = \
                np.array(self.dims_feat)[np.array(self.idx_conv)+1].sum()
        else:
            _dim_input_jk = self.dims_feat[-1]
        self.layer_jk = layers.JumpingKnowledge(self.arch_gcn,
                                                _dim_input_jk,mode=self.jk)
        self.outputs = self.layer_jk([_outputs_l, self.idx_conv])
        # OUPTUT LAYER
        self.outputs = tf.nn.l2_normalize(self.outputs, 1)
        _dim_final = self.arch_gcn['dim'] if self.jk else self.dims_feat[-1]
        self.layer_pred = \
            layers.HighOrderAggregator(_dim_final,
                                       self.num_classes,act="I",
                                       order=0,
                                       dropout=self.placeholders["dropout"],
                                       bias="bias")
        self.node_preds = self.layer_pred((self.outputs,None,None,None,None))

        # BACK PROP
        self._loss()
        update_ops = \
            tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            grads_and_vars = self.optimizer.compute_gradients(self.loss)
            clipped_grads_and_vars = \
                [(tf.clip_by_value(grad, -5.0, 5.0) \
                  if grad is not None else None, var)
                 for grad, var in grads_and_vars]
            self.grad, _ = clipped_grads_and_vars[0]
            self.opt_op = \
                self.optimizer.apply_gradients(clipped_grads_and_vars)
        self.preds = self.predict()

    def _loss(self):
        # these are all the trainable var
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += self.weight_decay * tf.nn.l2_loss(var)
        for var in self.layer_pred.vars.values():
            self.loss += self.weight_decay * tf.nn.l2_loss(var)

        # classification loss
        f_loss = tf.nn.sigmoid_cross_entropy_with_logits if self.sigmoid_loss\
            else tf.nn.softmax_cross_entropy_with_logits
        # weighted loss due to bias in appearance of vertices
        self.loss_terms = f_loss(logits=self.node_preds,
                                 labels=self.placeholders['labels'])
        loss_terms_ndims = self.loss_terms.shape.ndims \
            if self.loss_terms.shape is not None else None
        #if loss_terms_ndims == 1: # jgw
        self.loss_terms = tf.reshape(self.loss_terms,(-1,1))
        self._weight_loss_batch = \
            tf.nn.embedding_lookup(params=self.norm_loss,
                                   ids=self.node_subgraph)
        _loss_terms_weight = \
            tf.linalg.matmul(tf.transpose(a=self.loss_terms),
                             tf.reshape(self._weight_loss_batch,(-1,1)))
        self.loss += tf.reduce_sum(input_tensor=_loss_terms_weight)
        tf.compat.v1.summary.scalar('loss', self.loss)

    def predict(self):
        return tf.nn.sigmoid(self.node_preds) if self.sigmoid_loss \
            else tf.nn.softmax(self.node_preds)

    def get_aggregators(self,name=None):
        aggregators = []
        print(self.num_layers)
        for layer in range(self.num_layers):
            aggregator = \
                self.aggregator_cls(self.dims_weight[layer][0],
                                    self.dims_weight[layer][1],
                                    dropout=self.placeholders['dropout'],
                                    name=name,
                                    act=self.act_layer[layer],
                                    order=self.order_layer[layer],
                                    aggr=self.aggr_layer[layer],
                                    is_train=self.is_train,
                                    bias=self.bias_layer[layer],
                                    mulhead=self.mulhead)
            aggregators.append(aggregator)
        return aggregators

    def aggregate_subgraph(self, batch_size=None, name=None, mode='train'):
        if mode == 'train':
            hidden = tf.nn.embedding_lookup(params=self.features,
                                            ids=self.node_subgraph)
            adj = self.adj_subgraph
        else:
            hidden = self.features
            adj = self.adj_full_norm
        ret_l = list()
        _adj_partition_list = [self.adj_subgraph_0,self.adj_subgraph_1,
                               self.adj_subgraph_2,self.adj_subgraph_3,
                               self.adj_subgraph_4,self.adj_subgraph_5,
                               self.adj_subgraph_6,self.adj_subgraph_7]
        #if not args_global.dualGPU:
        for layer in range(self.num_layers):
            hidden = \
                self.aggregators[layer]((hidden,adj,
                                         self.dims_feat[layer],
                                         _adj_partition_list,
                                         self.dim0_adj_sub))
            ret_l.append(hidden)
#        else:
#            split=int(self.num_layers/2)
#            with tf.device('/gpu:0'):
#                for layer in range(split):
#                    hidden = self.aggregators[layer]((hidden,adj,self.dims_feat[layer],_adj_partition_list,self.dim0_adj_sub))
#                    ret_l.append(hidden)
#            with tf.device('/gpu:1'):
#                for layer in range(split,self.num_layers):
#                    hidden = self.aggregators[layer]((hidden,adj,self.dims_feat[layer],_adj_partition_list,self.dim0_adj_sub))
#                    ret_l.append(hidden)
        return ret_l

    def build_train_ops(self):
        self.loss_op = tf.compat.v1.reduce_mean(self.loss_terms)
        return \
            self.optimizer.minimize(self.loss_op,
                                    global_step=tf.compat.v1.train.get_global_step())

    def build_eval_metric_ops(self):
        return 0

def construct_placeholders(num_classes):
    placeholders = {
        'labels': tf.compat.v1.placeholder(DTYPE, shape=(None, num_classes),
                                           name='labels'),
        'node_subgraph': tf.compat.v1.placeholder(tf.int32, shape=(None),
                                                  name='node_subgraph'),
        'dropout': tf.compat.v1.placeholder(DTYPE, shape=(None),
                                            name='dropout'),
        'adj_subgraph': tf.compat.v1.sparse_placeholder(DTYPE,
                                                        name='adj_subgraph',
                                                        shape=(None,None)),
        'adj_subgraph_0': tf.compat.v1.sparse_placeholder(DTYPE,
                                                          name='adj_subgraph_0'),
        'adj_subgraph_1': tf.compat.v1.sparse_placeholder(DTYPE,
                                                          name='adj_subgraph_1'),
        'adj_subgraph_2': tf.compat.v1.sparse_placeholder(DTYPE,
                                                          name='adj_subgraph_2'),
        'adj_subgraph_3': tf.compat.v1.sparse_placeholder(DTYPE,
                                                          name='adj_subgraph_3'),
        'adj_subgraph_4': tf.compat.v1.sparse_placeholder(DTYPE,
                                                          name='adj_subgraph_4'),
        'adj_subgraph_5': tf.compat.v1.sparse_placeholder(DTYPE,
                                                          name='adj_subgraph_5'),
        'adj_subgraph_6': tf.compat.v1.sparse_placeholder(DTYPE,
                                                          name='adj_subgraph_6'),
        'adj_subgraph_7': tf.compat.v1.sparse_placeholder(DTYPE,
                                                          name='adj_subgraph_7'),
        'dim0_adj_sub': tf.compat.v1.placeholder(tf.int64, shape=(None),
                                                 name='dim0_adj_sub'),
        'norm_loss': tf.compat.v1.placeholder(DTYPE, shape=(None),
                                              name='norm_loss'),
        'is_train': tf.compat.v1.placeholder(tf.bool, shape=(None),
                                             name='is_train')
    }
    return placeholders

def adj_norm(adj, deg=None, sort_indices=True):
    """
    Normalize adj according to the method of rw normalization.
    Note that sym norm is used in the original GCN paper (kipf),
    while rw norm is used in GraphSAGE and some other variants.
    Here we don't perform sym norm since it doesn't seem to
    help with accuracy improvement.

    # Procedure:
    #       1. adj add self-connection --> adj'
    #       2. D' deg matrix from adj'
    #       3. norm by D^{-1} x adj'
    if sort_indices is True, we re-sort the indices of the returned adj
    Note that after 'dot' the indices of a node would be in descending order
    rather than ascending order
    """
    diag_shape = (adj.shape[0],adj.shape[1])
    D = adj.sum(1).flatten() if deg is None else deg
    norm_diag = scipy.sparse.dia_matrix((1/D,0),shape=diag_shape)
    adj_norm = norm_diag.dot(adj)
    if sort_indices:
        adj_norm.sort_indices()
    return adj_norm

def parse_layer_yml(arch_gcn,dim_input):
    """
    Parse the *.yml config file to retrieve the GNN structure.
    """
    num_layers = len(arch_gcn['arch'].split('-'))
    # set default values, then update by arch_gcn
    bias_layer = [arch_gcn['bias']]*num_layers
    act_layer = [arch_gcn['act']]*num_layers
    aggr_layer = [arch_gcn['aggr']]*num_layers
    dims_layer = [arch_gcn['dim']]*num_layers
    order_layer = [int(o) for o in arch_gcn['arch'].split('-')]
    return [dim_input]+dims_layer,order_layer,act_layer,bias_layer,aggr_layer

def log_dir(f_train_config, prefix, git_branch, git_rev, timestamp):
    #import getpass
    #log_dir = args_global.dir_log+"/log_train/" + prefix.split("/")[-1]
    #log_dir += "/{ts}-{model}-{gitrev:s}/".format(
    #        model='graphsaint',
    #        gitrev=git_rev.strip(),
    #        ts=timestamp)
    log_dir = "./graphsaint/tf/log_train"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if f_train_config != '':
        from shutil import copyfile
        copyfile(f_train_config,'{}/{}'.format(log_dir,f_train_config.split('/')[-1]))
    return log_dir

def prepare(features, labels, params):
    train_params = params["train_params"]
    arch_gcn = params["arch_gcn"]

    #adj_full,adj_train,feats,class_arr,role = train_data
    adj_full = features["adj_full"]
    adj_train = features["adj_train"]
    feats = features["feats"]
    class_arr = labels["class_arr"]
    role = features["role"]

    adj_full = adj_full.astype(np.int32)
    adj_train = adj_train.astype(np.int32)
    adj_full_norm = adj_norm(adj_full)
    num_classes = class_arr.shape[1]

    placeholders = construct_placeholders(num_classes)
    minibatch = Minibatch(adj_full_norm, adj_train, role, class_arr,
                          placeholders, train_params)
    model = GraphSAINT(num_classes, placeholders, feats, arch_gcn,
                       train_params, adj_full_norm, logging=True)

    # Initialize session
    sess = \
        tf.compat.v1.Session(config=\
                             tf.compat.v1.ConfigProto(device_count={"CPU":40},
                                                      inter_op_parallelism_threads=44,
                                                      intra_op_parallelism_threads=44,
                                                      log_device_placement=False))
    ph_misc_stat = {'val_f1_micro': tf.compat.v1.placeholder(DTYPE, shape=()),
                    'val_f1_macro': tf.compat.v1.placeholder(DTYPE, shape=()),
                    'train_f1_micro': tf.compat.v1.placeholder(DTYPE,
                                                               shape=()),
                    'train_f1_macro': tf.compat.v1.placeholder(DTYPE,
                                                               shape=()),
                    'time_per_epoch': tf.compat.v1.placeholder(DTYPE,
                                                               shape=()),
                    'size_subgraph': tf.compat.v1.placeholder(tf.int32,
                                                              shape=())}
    merged = tf.compat.v1.summary.merge_all()

    with tf.compat.v1.name_scope('summary'):
        _misc_val_f1_micro = tf.compat.v1.summary.scalar('val_f1_micro',
                                                         ph_misc_stat['val_f1_micro'])
        _misc_val_f1_macro = tf.compat.v1.summary.scalar('val_f1_macro',
                                                         ph_misc_stat['val_f1_macro'])
        _misc_train_f1_micro = tf.compat.v1.summary.scalar('train_f1_micro',
                                                           ph_misc_stat['train_f1_micro'])
        _misc_train_f1_macro = tf.compat.v1.summary.scalar('train_f1_macro',
                                                           ph_misc_stat['train_f1_macro'])
        _misc_time_per_epoch = tf.compat.v1.summary.scalar('time_per_epoch',
                                                           ph_misc_stat['time_per_epoch'])
        _misc_size_subgraph = tf.compat.v1.summary.scalar('size_subgraph',
                                                          ph_misc_stat['size_subgraph'])

    misc_stats = tf.compat.v1.summary.merge([_misc_val_f1_micro,
                                             _misc_val_f1_macro,
                                             _misc_train_f1_micro,
                                             _misc_train_f1_macro,
                                             _misc_time_per_epoch,
                                             _misc_size_subgraph])
    #op = sess.graph.get_operations()
    #[m.values() for m in op][1]
    summary_writer = \
        tf.compat.v1.summary.FileWriter(log_dir(params["params_file"],
                                                params["data_prefix"],
                                                git_branch, git_rev,
                                                timestamp),
                                        sess.graph)
    #summary_writer = \
    #    tf.summary.create_file_writer('./graphsaint/tf/log')
    #tf.compat.v1.summary_writer.add_summary(merged, misc_stats)

    # Init variables
    sess.run(tf.compat.v1.global_variables_initializer())

    return model, minibatch, sess, [merged,misc_stats], \
        ph_misc_stat, summary_writer

def model_fn(features, labels, mode, params):
    """
    The model function to be used with TF estimator API
    """
    #gnn = getattr(sys.modules[__name__], params["model"]["model"])(params)

#    model, minibatch, sess, train_stat, ph_misc_stat, summary_writer = \
#        prepare(features, labels, params)
#
#    #outputs = gnn(features, labels, mode)
#    #loss = gnn.build_total_loss(outputs, features, labels, mode)
#    loss = model.loss
#
#    if mode == tf.estimator.ModeKeys.TRAIN:
#        #train_op = gnn.build_train_ops(loss)
#        train_op = model.build_train_ops()
#        eval_metric_ops = None
#    elif mode == tf.estimator.ModeKeys.EVAL:
#        train_op = None
#        #eval_metric_ops = gnn.build_eval_metric_ops(outputs, features, labels)
#        eval_metric_ops = model.build_eval_metric_ops()
#    else:
#        raise ValueError(f"Mode {mode} not supported.")

    loss = None
    train_op = None
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, train_op=train_op)#,
    #    eval_metric_ops=eval_metric_ops
    #)
