import numpy as np
import tensorflow as tf
import monolith.layers as layers
import tensorflow.keras.initializers as initializers

def lhuc_pp_net(name, nn_dims, nn_initializers, nn_activations, nn_inputs, lhuc_dims, lhuc_inputs,
            scale_last, lhuc_use_nn_input=False):
    '''
    nn_dims: 每个任务对应的隐藏层的单元
    '''
    if not isinstance(nn_initializers, list):
        nn_initializers = [nn_initializers] * len(nn_dims)
    if not isinstance(nn_activations, list):
        nn_activations = [nn_activations] * (len(nn_dims) - 1) + [None]
    if lhuc_use_nn_input:
        lhuc_final_input = tf.concat([tf.stop_gradient(nn_inputs), lhuc_inputs], axis=1)
    else:
        lhuc_final_input = lhuc_inputs
    tf.summary.histogram('lhuc_final_input', lhuc_final_input) # 便于后续调整参数
    cur_layer = nn_inputs
    for idx, nn_dim in enumerate(nn_dims):
        lhuc_output = layers.MLP(name='{}_lhuc_{}'.format(name, idx),
                        output_dims=lhuc_dims+[int(cur_layer.shape[1])],
                        activations=['relu'] * len(lhuc_dims) + [None],
                        initializers=tf.keras.initializers.GlorotNormal(),
                        )(lhuc_final_input)
        tf.summary.histogram('lhuc_output_%d'%idx, lhuc_output)
        lhuc_scale = tf.nn.tanh(lhuc_output * 0.2) * (5.0 + idx) + 1.0
        tf.summary.histogram('lhuc_scale_%d'%idx, lhuc_scale)
        tf.summary.histogram('lhuc_cur_layer_%d'%idx, cur_layer)
        cur_layer = layers.MLP(name='{}_layer_{}'.format(name, idx),
                        output_dims=[nn_dim],
                        initializers=nn_initializers[idx],
                        activations=[nn_activations[idx]], # fatal
                        )(cur_layer * lhuc_scale)
    if scale_last:
        lhuc_scale = layers.MLP(name='{}_lhuc_{}'.format(name, len(nn_dims)),
                        output_dims=lhuc_dims+[nn_dims[-1]],
                        activations=['relu'] * len(lhuc_dims) + ['sigmoid'],
                        initializers=tf.keras.initializers.GlorotNormal(),
                        )(lhuc_final_input)
        tf.compat.v1.summary.histogram('{}_lhuc_{}'.format(name, len(nn_dims)) + '_scale', lhuc_scale)
        cur_layer = cur_layer * lhuc_scale * 2.0
        tf.compat.v1.summary.histogram('{}_layer_{}'.format(name, len(nn_dims))+'_output', cur_layer)

    return cur_layer

# lhuc epnet
def lhuc_ep_net(name, output_dim, lhuc_inputs, lhuc_dims):
    
    lhuc_final_input = lhuc_inputs
    lhuc_output = layers.MLP(name='{}'.format(name),
                    output_dims=lhuc_dims+[output_dim],
                    activations=['relu'] * len(lhuc_dims) + [None],
                    initializers=initializers.HeNormal(),
                    )(lhuc_final_input)
    tf.summary.histogram('lhuc_ep_output', lhuc_output)
    lhuc_scale = tf.nn.tanh(lhuc_output * 0.2) * 5.0 + 1.0
    tf.summary.histogram('lhuc_ep_scale', lhuc_scale)
    return lhuc_scale