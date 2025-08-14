from model.network import SpaceTime

input_dim = 1
output_dim = 1

model_dim = 128
kernel_dim = 64
n_layers_encoder = 2
n_layers_decoder = 2
norm_order=1
dropout=0.25
closed_loop_decoder= False

lag=900
horizon=100
inference_only = False

#embedding
embedding_config = {'method': 'repeat', 'kwargs': {'input_dim': input_dim, 'embedding_dim': model_dim, 'n_heads': 4, 'n_kernels': 32}}

#encoder
encoder_block_start = {'input_dim': model_dim, 'pre_config': {'method': 'residual', 'kwargs': {'max_diff_order': 4, 'min_avg_window': 4, 'max_avg_window': 64, 'model_dim': model_dim, 'n_kernels': 8, 'kernel_dim': 2, 'kernel_repeat': 16, 'n_heads': 1, 'head_dim': 1, 'kernel_weights': None, 'kernel_init': None, 'kernel_train': False, 'skip_connection': False, 'seed': 0}}, 'ssm_config': {'method': 'companion', 'kwargs': {'model_dim': model_dim, 'n_kernels': 16, 'kernel_dim': kernel_dim, 'kernel_repeat': 1, 'n_heads': 8, 'head_dim': 1, 'kernel_weights': None, 'kernel_init': 'normal', 'kernel_train': True, 'skip_connection': True, 'norm_order': norm_order}}, 'mlp_config': {'method': 'mlp', 'kwargs': {'input_dim': model_dim, 'output_dim': model_dim, 'activation': 'gelu', 'dropout': dropout, 'layernorm': False, 'n_layers': 1, 'n_activations': 1, 'pre_activation': True, 'input_shape': 'bld', 'skip_connection': True, 'average_pool': None}}, 'skip_connection': True, 'skip_preprocess': False}
encoder_block_standard = {'input_dim': model_dim, 'pre_config': {'method': 'identity', 'kwargs': None}, 'ssm_config': {'method': 'companion', 'kwargs': {'model_dim': model_dim, 'n_kernels': 128, 'kernel_dim': kernel_dim, 'kernel_repeat': 1, 'n_heads': 1, 'head_dim': 1, 'kernel_weights': None, 'kernel_init': 'normal', 'kernel_train': True, 'skip_connection': True, 'norm_order': norm_order}}, 'mlp_config': {'method': 'mlp', 'kwargs': {'input_dim': model_dim, 'output_dim': model_dim, 'activation': 'gelu', 'dropout': dropout, 'layernorm': False, 'n_layers': 1, 'n_activations': 1, 'pre_activation': True, 'input_shape': 'bld', 'skip_connection': True, 'average_pool': None}}, 'skip_connection': True, 'skip_preprocess': True}
encoder_config= {'blocks': [encoder_block_start]+ [encoder_block_standard]*(n_layers_encoder-1)}
if(n_layers_decoder==0):
    encoder_config["blocks"][-1]["mlp_config"]= {'method': 'identity', 'kwargs': None} #activation is taken care of in output layer

#decoder
decoder_block_closed = {'input_dim': model_dim, 'pre_config': {'method': 'identity', 'kwargs': None}, 'ssm_config': {'method': 'closed_loop_companion', 'kwargs': {'lag':lag, 'horizon':horizon, 'model_dim': model_dim, 'n_kernels': 128, 'kernel_dim': kernel_dim, 'kernel_repeat': 1, 'n_heads': 1, 'head_dim': 1, 'kernel_weights': None, 'kernel_init': 'normal', 'kernel_train': True, 'skip_connection': False, 'norm_order': norm_order, 'use_initial':False}}, 'mlp_config': {'method': 'mlp', 'kwargs': {'input_dim': model_dim, 'output_dim': model_dim, 'activation': 'gelu', 'dropout': dropout, 'layernorm': False, 'n_layers': 1, 'n_activations': 1, 'pre_activation': True, 'input_shape': 'bld', 'skip_connection': True, 'average_pool': None}}, 'skip_connection': False, 'skip_preprocess': False}
decoder_config={'blocks': [] if n_layers_decoder==0 else [decoder_block_closed]+[encoder_block_standard]*(n_layers_decoder-1)}
if(n_layers_decoder>0):
    decoder_config["blocks"][-1]["mlp_config"]= {'method': 'identity', 'kwargs': None} #activation is taken care of in output layer

#output
output_config={'input_dim': model_dim, 'output_dim': output_dim, 'method': 'mlp', 'kwargs': {'input_dim': model_dim, 'output_dim': output_dim, 'activation': 'gelu', 'dropout': dropout, 'layernorm': False, 'n_layers': 1, 'n_activations': 1, 'pre_activation': True, 'input_shape': 'bld', 'skip_connection': False, 'average_pool': None}}


spacetime_config={'embedding_config': embedding_config, 'encoder_config': encoder_config, 'decoder_config': decoder_config, 'output_config': output_config, "inference_only": inference_only, "lag": lag, "horizon": horizon}

model = SpaceTime(**spacetime_config)
#input shape B L E
#output shape B L E
print(model)
    
