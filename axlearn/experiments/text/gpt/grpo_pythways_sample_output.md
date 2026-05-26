## Job Launch Command
```shell
export BASTION_TIER=disabled
export NAME=eshen-v6e-rl-grpo-pathways
export RUNNER_NAME=gke_tpu_single
export CONFIG=grpo-fuji-8B-v1
export INSTANCE_TYPE=tpu-v6e-16
export CLUSTER=ericshen-axlearn
export OUTPUT_DIR="gs://cloud-tpu-multipod-dev-axlearn/users/ericshen/${NAME}/$(date +%s)"

➜  ~ nohup axlearn gcp launch run \
  --cluster=$CLUSTER \
  --instance_type=$INSTANCE_TYPE \
  --name=$NAME \
  --num_replicas=1 \
  --runner_name=$RUNNER_NAME \
  --bundler_spec=allow_dirty=True \
  --bundler_type=artifactregistry \
  --bundler_spec=image=tpu \
  --bundler_spec=dockerfile=Dockerfile \
  --bundler_spec=target=tpu --pathways_head_mem=128 --pathways_head_cpu=8 --env=GRPO_REWARD_TYPE:gsm8k \
  -- \
  python3 -m axlearn.common.launch_trainer_main \
  --module=text.gpt.grpo_native_example \
  --config=$CONFIG \
  --trainer_dir=$OUTPUT_DIR \
  --data_dir=gs://ericshen-axlearn/tensorflow-datasets \
  --jax_backend=proxy \
   --trainer_log_every_n_steps=10 > axlearn_launcher.log 2>&1 &

```
## output directory gs://cloud-tpu-multipod-dev-axlearn/users/ericshen/eshen-v6e-rl-grpo-pathways/1779314896
## trainer_config
```shell
➜  ~ gcloud storage cat gs://cloud-tpu-multipod-dev-axlearn/users/ericshen/eshen-v6e-rl-grpo-pathways/1779314896/trainer_config
batch_axis_names: 'data'
checkpointer.gc_loop_interval_seconds: 60
checkpointer.keep_last_n: 1
checkpointer.klass: 'axlearn.common.checkpointer.Checkpointer'
checkpointer.save_policy.fn: 'axlearn.common.checkpointer.every_n_steps_policy'
checkpointer.save_policy.min_step: 1
checkpointer.save_policy.n: 1
checkpointer.storage.klass: 'axlearn.common.checkpointer.TensorStoreStateStorage'
checkpointer.storage.timeout_secs: 3600
crash_on_hang_timeout_seconds: 7200
dir: 'gs://cloud-tpu-multipod-dev-axlearn/users/ericshen/eshen-v6e-rl-grpo-pathways/1779314896'
init_state_builder.klass: 'axlearn.experiments.text.gpt.grpo_native_example.GRPOStateBuilder'
init_state_builder.storage_builder.concurrent_gb: 4
init_state_builder.storage_builder.dir: 'gs://ericshen-axlearn/checkpoints/llama-3-1-8B-instruct/step_00000000'
init_state_builder.storage_builder.klass: 'axlearn.common.state_builder.TensorStoreStateStorageBuilder'
init_state_builder.storage_builder.storage.klass: 'axlearn.common.checkpointer.TensorStoreStateStorage'
init_state_builder.storage_builder.storage.timeout_secs: 3600
init_state_builder.storage_builder.validation: 'CONTAINS_STATE_UP_TO_DTYPE'
input.batcher.fn: 'axlearn.common.input_tf_data.per_feed_batch'
input.batcher.pad_example_fn: 'axlearn.common.input_tf_data.default_pad_example_fn'
input.batcher.prefetch_buffer_size: -1
input.input_dispatcher.global_logical_batch_size: 128
input.input_dispatcher.klass: 'axlearn.common.input_dispatch.SpmdInputDispatcher'
input.input_dispatcher.partition_spec: PartitionSpec(('data', 'expert', 'fsdp'),)
input.input_partitioner.fn: 'axlearn.common.input_base.partition_by_path_rank'
input.input_partitioner.path_rank_to_partition[(None, 1)]: PartitionSpec(('data', 'expert', 'fsdp'),)
input.input_partitioner.path_rank_to_partition[(None, 2)]: PartitionSpec(('data', 'expert', 'fsdp'), 'seq')
input.is_training: True
input.klass: 'axlearn.common.input_tf_data.Input'
input.processor.fn: 'axlearn.common.input_tf_data.identity'
input.source.dataset_name: 'gsm8k'
input.source.fn: 'axlearn.experiments.text.gpt.grpo_native_example.gsm8k_tfds_input'
input.source.is_training: True
input.source.max_sequence_length: 512
input.source.split: 'train'
input.source.train_shuffle_buffer_size: 16384
input.source.vocab_cfg.filename: 'Llama-3-tokenizer.json'
input.source.vocab_cfg.klass: 'axlearn.experiments.text.gpt.grpo_native_example.GRPOV3Vocabulary'
klass: 'axlearn.common.grpo_trainer.GrpoSpmdTrainer'
learner.beta: 0.04
learner.ema.fn: 'axlearn.common.optimizers.param_ema'
learner.enable_per_variable_summaries: False
learner.epsilon: 0.2
learner.klass: 'axlearn.common.grpo_learner.AxlearnGrpoLearner'
learner.name: 'learner'
learner.num_generations: 2
learner.optimizer.b1: 0.9
learner.optimizer.b2: 0.95
learner.optimizer.eps: 1e-08
learner.optimizer.fn: 'axlearn.common.optimizers.adamw_optimizer'
learner.optimizer.learning_rate: 1e-05
learner.optimizer.weight_decay: 0
learner.update_rules[0][0]: 'reference/.*'
learner.update_rules[0][1]: <UpdateType.NO_UPDATE: 'no_update'>
learner.update_rules[1][0]: 'sampler/.*'
learner.update_rules[1][1]: <UpdateType.NO_UPDATE: 'no_update'>
log_every_n_steps: 10
max_step: 1000
mesh_axis_names[0]: 'pipeline'
mesh_axis_names[1]: 'data'
mesh_axis_names[2]: 'expert'
mesh_axis_names[3]: 'fsdp'
mesh_axis_names[4]: 'seq'
mesh_axis_names[5]: 'model'
mesh_shape[0]: 1
mesh_shape[1]: 1
mesh_shape[2]: 1
mesh_shape[3]: 16
mesh_shape[4]: 1
mesh_shape[5]: 1
model.actor.batch_axis_names: None
model.actor.decoder.attention_mask: None
model.actor.decoder.decoding.klass: 'axlearn.common.decoder.DecodingLayer'
model.actor.decoder.dim: 4096
model.actor.decoder.dropout_rate: 0.0
model.actor.decoder.dtype: 'jax.numpy.bfloat16'
model.actor.decoder.emb.dropout.klass: 'axlearn.common.layers.Dropout'
model.actor.decoder.emb.klass: 'axlearn.common.embedding.TransformerTextEmbeddings'
model.actor.decoder.emb.token_emb.klass: 'axlearn.common.layers.Embedding'
model.actor.decoder.emb.token_emb.param_init.init_by_param_name['.*weight$'].distribution: 'normal'
model.actor.decoder.emb.token_emb.param_init.init_by_param_name['.*weight$'].fan: 'fan_out'
model.actor.decoder.emb.token_emb.param_init.init_by_param_name['.*weight$'].klass: 'axlearn.common.param_init.WeightInitializer'
model.actor.decoder.emb.token_emb.param_init.init_by_param_name['.*weight$'].scale: 1.0
model.actor.decoder.emb.token_emb.param_init.klass: 'axlearn.common.param_init.DefaultInitializer'
model.actor.decoder.emb.token_emb.param_partition_spec[0]: 'model'
model.actor.decoder.emb.token_emb.param_partition_spec[1][0]: 'expert'
model.actor.decoder.emb.token_emb.param_partition_spec[1][1]: 'fsdp'
model.actor.decoder.emb.token_emb.param_partition_spec[1][2]: 'seq'
model.actor.decoder.eos_token_id: 128001
model.actor.decoder.klass: 'axlearn.common.decoder.Decoder'
model.actor.decoder.lm_head.klass: 'axlearn.common.decoder.LmHead'
model.actor.decoder.lm_head.param_partition_spec[0]: 'model'
model.actor.decoder.lm_head.param_partition_spec[1][0]: 'expert'
model.actor.decoder.lm_head.param_partition_spec[1][1]: 'fsdp'
model.actor.decoder.lm_head.param_partition_spec[1][2]: 'seq'
model.actor.decoder.logits_partition_spec[0][0]: 'data'
model.actor.decoder.logits_partition_spec[0][1]: 'expert'
model.actor.decoder.logits_partition_spec[0][2]: 'fsdp'
model.actor.decoder.logits_partition_spec[1]: 'seq'
model.actor.decoder.logits_partition_spec[2]: 'model'
model.actor.decoder.output_dropout.klass: 'axlearn.common.layers.Dropout'
model.actor.decoder.output_norm.eps: 1e-05
model.actor.decoder.output_norm.forward_dtype: None
model.actor.decoder.output_norm.klass: 'axlearn.common.layers.RMSNorm'
model.actor.decoder.pad_token_id: 128004
model.actor.decoder.transformer.klass: 'axlearn.common.attention.RepeatedTransformerLayer'
model.actor.decoder.transformer.layer.feed_forward.activation[0]: 'nn.silu'
model.actor.decoder.transformer.layer.feed_forward.activation[1]: 'linear'
model.actor.decoder.transformer.layer.feed_forward.dropout.klass: 'axlearn.common.layers.Dropout'
model.actor.decoder.transformer.layer.feed_forward.hidden_dim.fn: 'axlearn.experiments.text.gpt.common.scale_fn'
model.actor.decoder.transformer.layer.feed_forward.hidden_dim.round_up_to_multiples_of: 256
model.actor.decoder.transformer.layer.feed_forward.hidden_dim.scale: 3.5
model.actor.decoder.transformer.layer.feed_forward.klass: 'axlearn.common.attention.TransformerFeedForwardLayer'
model.actor.decoder.transformer.layer.feed_forward.linear1.bias: False
model.actor.decoder.transformer.layer.feed_forward.linear1.klass: 'axlearn.common.layers.Linear'
model.actor.decoder.transformer.layer.feed_forward.linear1.output_partition_spec[0][0]: 'data'
model.actor.decoder.transformer.layer.feed_forward.linear1.output_partition_spec[0][1]: 'expert'
model.actor.decoder.transformer.layer.feed_forward.linear1.output_partition_spec[0][2]: 'fsdp'
model.actor.decoder.transformer.layer.feed_forward.linear1.output_partition_spec[1]: 'seq'
model.actor.decoder.transformer.layer.feed_forward.linear1.output_partition_spec[2]: 'model'
model.actor.decoder.transformer.layer.feed_forward.linear1.param_partition_spec[0][0]: 'expert'
model.actor.decoder.transformer.layer.feed_forward.linear1.param_partition_spec[0][1]: 'fsdp'
model.actor.decoder.transformer.layer.feed_forward.linear1.param_partition_spec[0][2]: 'seq'
model.actor.decoder.transformer.layer.feed_forward.linear1.param_partition_spec[1]: 'model'
model.actor.decoder.transformer.layer.feed_forward.linear2.bias: False
model.actor.decoder.transformer.layer.feed_forward.linear2.klass: 'axlearn.common.layers.Linear'
model.actor.decoder.transformer.layer.feed_forward.linear2.output_partition_spec[0][0]: 'data'
model.actor.decoder.transformer.layer.feed_forward.linear2.output_partition_spec[0][1]: 'expert'
model.actor.decoder.transformer.layer.feed_forward.linear2.output_partition_spec[0][2]: 'fsdp'
model.actor.decoder.transformer.layer.feed_forward.linear2.output_partition_spec[1]: 'seq'
model.actor.decoder.transformer.layer.feed_forward.linear2.output_partition_spec[2]: 'model'
model.actor.decoder.transformer.layer.feed_forward.linear2.param_partition_spec[0]: 'model'
model.actor.decoder.transformer.layer.feed_forward.linear2.param_partition_spec[1][0]: 'expert'
model.actor.decoder.transformer.layer.feed_forward.linear2.param_partition_spec[1][1]: 'fsdp'
model.actor.decoder.transformer.layer.feed_forward.linear2.param_partition_spec[1][2]: 'seq'
model.actor.decoder.transformer.layer.feed_forward.norm.eps: 1e-05
model.actor.decoder.transformer.layer.feed_forward.norm.forward_dtype: None
model.actor.decoder.transformer.layer.feed_forward.norm.klass: 'axlearn.common.layers.RMSNorm'
model.actor.decoder.transformer.layer.feed_forward.residual_weight: 1.0
model.actor.decoder.transformer.layer.feed_forward.stochastic_depth.klass: 'axlearn.common.layers.StochasticDepth'
model.actor.decoder.transformer.layer.feed_forward.stochastic_depth.mode: 'row'
model.actor.decoder.transformer.layer.feed_forward.structure: 'prenorm'
model.actor.decoder.transformer.layer.klass: 'axlearn.common.attention.TransformerLayer'
model.actor.decoder.transformer.layer.remat_spec['prevent_cse']: False
model.actor.decoder.transformer.layer.remat_spec['policy'].fn: 'axlearn.common.attention._save_and_offload_only_these_names_regex'
model.actor.decoder.transformer.layer.remat_spec['policy'].names_which_can_be_offloaded: None
model.actor.decoder.transformer.layer.remat_spec['policy'].names_which_can_be_saved: '.*([qkvo]_proj|context)'
model.actor.decoder.transformer.layer.remat_spec['policy'].offload_dst: 'pinned_host'
model.actor.decoder.transformer.layer.remat_spec['policy'].offload_src: 'device'
model.actor.decoder.transformer.layer.self_attention.attention.causal: True
model.actor.decoder.transformer.layer.self_attention.attention.dropout.klass: 'axlearn.common.layers.Dropout'
model.actor.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.klass: 'axlearn.common.attention.FusedGroupedQKVLinear'
model.actor.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.bias: False
model.actor.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.klass: 'axlearn.common.attention.MultiheadInputLinear'
model.actor.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.param_partition_spec[0][0]: 'expert'
model.actor.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.param_partition_spec[0][1]: 'fsdp'
model.actor.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.param_partition_spec[0][2]: 'seq'
model.actor.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.param_partition_spec[1]: 'model'
model.actor.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.param_partition_spec[2]: None
model.actor.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.num_kv_heads: 8
model.actor.decoder.transformer.layer.self_attention.attention.input_linear.klass: 'axlearn.common.attention.RoFormerQKVLinear'
model.actor.decoder.transformer.layer.self_attention.attention.input_linear.rope_pos_emb_layer.klass: 'axlearn.common.attention.RoFormerSinusoidalPositionalEmbedding'
model.actor.decoder.transformer.layer.self_attention.attention.input_linear.rope_pos_emb_layer.theta: 500000.0
model.actor.decoder.transformer.layer.self_attention.attention.input_linear.rotary_value: False
model.actor.decoder.transformer.layer.self_attention.attention.key_scale.klass: 'axlearn.common.attention.ScaleKey'
model.actor.decoder.transformer.layer.self_attention.attention.klass: 'axlearn.common.attention.GroupedQueryAttention'
model.actor.decoder.transformer.layer.self_attention.attention.kv_cache.cache_dtype: 'jax.numpy.bfloat16'
model.actor.decoder.transformer.layer.self_attention.attention.kv_cache.klass: 'axlearn.common.kv_cache.kv_cache.KVCache'
model.actor.decoder.transformer.layer.self_attention.attention.num_heads: 32
model.actor.decoder.transformer.layer.self_attention.attention.output_linear.bias: False
model.actor.decoder.transformer.layer.self_attention.attention.output_linear.klass: 'axlearn.common.attention.MultiheadOutputLinear'
model.actor.decoder.transformer.layer.self_attention.attention.output_linear.param_partition_spec[0][0]: 'expert'
model.actor.decoder.transformer.layer.self_attention.attention.output_linear.param_partition_spec[0][1]: 'fsdp'
model.actor.decoder.transformer.layer.self_attention.attention.output_linear.param_partition_spec[0][2]: 'seq'
model.actor.decoder.transformer.layer.self_attention.attention.output_linear.param_partition_spec[1]: 'model'
model.actor.decoder.transformer.layer.self_attention.attention.output_linear.param_partition_spec[2]: None
model.actor.decoder.transformer.layer.self_attention.attention.query_scale.klass: 'axlearn.common.attention.ScaleQuery'
model.actor.decoder.transformer.layer.self_attention.dropout.klass: 'axlearn.common.layers.Dropout'
model.actor.decoder.transformer.layer.self_attention.klass: 'axlearn.common.attention.TransformerAttentionLayer'
model.actor.decoder.transformer.layer.self_attention.norm.eps: 1e-05
model.actor.decoder.transformer.layer.self_attention.norm.forward_dtype: None
model.actor.decoder.transformer.layer.self_attention.norm.klass: 'axlearn.common.layers.RMSNorm'
model.actor.decoder.transformer.layer.self_attention.stochastic_depth.klass: 'axlearn.common.layers.StochasticDepth'
model.actor.decoder.transformer.layer.self_attention.stochastic_depth.mode: 'row'
model.actor.decoder.transformer.layer.self_attention.structure: 'prenorm'
model.actor.decoder.transformer.num_layers: 32
model.actor.decoder.transformer.repeat.drop_output.fn: 'axlearn.common.repeat._drop_by_regex'
model.actor.decoder.transformer.repeat.drop_output.rules[0]: 'module_outputs.*'
model.actor.decoder.transformer.repeat.klass: 'axlearn.common.attention._TransformerRepeat'
model.actor.decoder.vocab_size: 128256
model.actor.dtype: 'jax.numpy.bfloat16'
model.actor.klass: 'axlearn.common.causal_lm.Model'
model.actor.metrics.klass: 'axlearn.common.causal_lm.CompositeLossMetrics'
model.actor.metrics.metrics['lm'].klass: 'axlearn.common.causal_lm.CrossEntropyLossMetrics'
model.actor.metrics.metrics['aux'].klass: 'axlearn.common.causal_lm.AuxLossMetrics'
model.actor.param_init.init_by_param_name['.*weight$'].distribution: 'normal'
model.actor.param_init.init_by_param_name['.*weight$'].fan: 'fan_in'
model.actor.param_init.init_by_param_name['.*weight$'].klass: 'axlearn.common.param_init.WeightInitializer'
model.actor.param_init.init_by_param_name['.*weight$'].scale: 1.0
model.actor.param_init.klass: 'axlearn.common.param_init.DefaultInitializer'
model.dtype: 'jax.numpy.bfloat16'
model.klass: 'axlearn.common.grpo_model.GrpoModel'
model.name: 'model'
model.reference.batch_axis_names: None
model.reference.decoder.attention_mask: None
model.reference.decoder.decoding.klass: 'axlearn.common.decoder.DecodingLayer'
model.reference.decoder.dim: 4096
model.reference.decoder.dropout_rate: 0.0
model.reference.decoder.dtype: 'jax.numpy.bfloat16'
model.reference.decoder.emb.dropout.klass: 'axlearn.common.layers.Dropout'
model.reference.decoder.emb.klass: 'axlearn.common.embedding.TransformerTextEmbeddings'
model.reference.decoder.emb.token_emb.klass: 'axlearn.common.layers.Embedding'
model.reference.decoder.emb.token_emb.param_init.init_by_param_name['.*weight$'].distribution: 'normal'
model.reference.decoder.emb.token_emb.param_init.init_by_param_name['.*weight$'].fan: 'fan_out'
model.reference.decoder.emb.token_emb.param_init.init_by_param_name['.*weight$'].klass: 'axlearn.common.param_init.WeightInitializer'
model.reference.decoder.emb.token_emb.param_init.init_by_param_name['.*weight$'].scale: 1.0
model.reference.decoder.emb.token_emb.param_init.klass: 'axlearn.common.param_init.DefaultInitializer'
model.reference.decoder.emb.token_emb.param_partition_spec[0]: 'model'
model.reference.decoder.emb.token_emb.param_partition_spec[1][0]: 'expert'
model.reference.decoder.emb.token_emb.param_partition_spec[1][1]: 'fsdp'
model.reference.decoder.emb.token_emb.param_partition_spec[1][2]: 'seq'
model.reference.decoder.eos_token_id: 128001
model.reference.decoder.klass: 'axlearn.common.decoder.Decoder'
model.reference.decoder.lm_head.klass: 'axlearn.common.decoder.LmHead'
model.reference.decoder.lm_head.param_partition_spec[0]: 'model'
model.reference.decoder.lm_head.param_partition_spec[1][0]: 'expert'
model.reference.decoder.lm_head.param_partition_spec[1][1]: 'fsdp'
model.reference.decoder.lm_head.param_partition_spec[1][2]: 'seq'
model.reference.decoder.logits_partition_spec[0][0]: 'data'
model.reference.decoder.logits_partition_spec[0][1]: 'expert'
model.reference.decoder.logits_partition_spec[0][2]: 'fsdp'
model.reference.decoder.logits_partition_spec[1]: 'seq'
model.reference.decoder.logits_partition_spec[2]: 'model'
model.reference.decoder.output_dropout.klass: 'axlearn.common.layers.Dropout'
model.reference.decoder.output_norm.eps: 1e-05
model.reference.decoder.output_norm.forward_dtype: None
model.reference.decoder.output_norm.klass: 'axlearn.common.layers.RMSNorm'
model.reference.decoder.pad_token_id: 128004
model.reference.decoder.transformer.klass: 'axlearn.common.attention.RepeatedTransformerLayer'
model.reference.decoder.transformer.layer.feed_forward.activation[0]: 'nn.silu'
model.reference.decoder.transformer.layer.feed_forward.activation[1]: 'linear'
model.reference.decoder.transformer.layer.feed_forward.dropout.klass: 'axlearn.common.layers.Dropout'
model.reference.decoder.transformer.layer.feed_forward.hidden_dim.fn: 'axlearn.experiments.text.gpt.common.scale_fn'
model.reference.decoder.transformer.layer.feed_forward.hidden_dim.round_up_to_multiples_of: 256
model.reference.decoder.transformer.layer.feed_forward.hidden_dim.scale: 3.5
model.reference.decoder.transformer.layer.feed_forward.klass: 'axlearn.common.attention.TransformerFeedForwardLayer'
model.reference.decoder.transformer.layer.feed_forward.linear1.bias: False
model.reference.decoder.transformer.layer.feed_forward.linear1.klass: 'axlearn.common.layers.Linear'
model.reference.decoder.transformer.layer.feed_forward.linear1.output_partition_spec[0][0]: 'data'
model.reference.decoder.transformer.layer.feed_forward.linear1.output_partition_spec[0][1]: 'expert'
model.reference.decoder.transformer.layer.feed_forward.linear1.output_partition_spec[0][2]: 'fsdp'
model.reference.decoder.transformer.layer.feed_forward.linear1.output_partition_spec[1]: 'seq'
model.reference.decoder.transformer.layer.feed_forward.linear1.output_partition_spec[2]: 'model'
model.reference.decoder.transformer.layer.feed_forward.linear1.param_partition_spec[0][0]: 'expert'
model.reference.decoder.transformer.layer.feed_forward.linear1.param_partition_spec[0][1]: 'fsdp'
model.reference.decoder.transformer.layer.feed_forward.linear1.param_partition_spec[0][2]: 'seq'
model.reference.decoder.transformer.layer.feed_forward.linear1.param_partition_spec[1]: 'model'
model.reference.decoder.transformer.layer.feed_forward.linear2.bias: False
model.reference.decoder.transformer.layer.feed_forward.linear2.klass: 'axlearn.common.layers.Linear'
model.reference.decoder.transformer.layer.feed_forward.linear2.output_partition_spec[0][0]: 'data'
model.reference.decoder.transformer.layer.feed_forward.linear2.output_partition_spec[0][1]: 'expert'
model.reference.decoder.transformer.layer.feed_forward.linear2.output_partition_spec[0][2]: 'fsdp'
model.reference.decoder.transformer.layer.feed_forward.linear2.output_partition_spec[1]: 'seq'
model.reference.decoder.transformer.layer.feed_forward.linear2.output_partition_spec[2]: 'model'
model.reference.decoder.transformer.layer.feed_forward.linear2.param_partition_spec[0]: 'model'
model.reference.decoder.transformer.layer.feed_forward.linear2.param_partition_spec[1][0]: 'expert'
model.reference.decoder.transformer.layer.feed_forward.linear2.param_partition_spec[1][1]: 'fsdp'
model.reference.decoder.transformer.layer.feed_forward.linear2.param_partition_spec[1][2]: 'seq'
model.reference.decoder.transformer.layer.feed_forward.norm.eps: 1e-05
model.reference.decoder.transformer.layer.feed_forward.norm.forward_dtype: None
model.reference.decoder.transformer.layer.feed_forward.norm.klass: 'axlearn.common.layers.RMSNorm'
model.reference.decoder.transformer.layer.feed_forward.residual_weight: 1.0
model.reference.decoder.transformer.layer.feed_forward.stochastic_depth.klass: 'axlearn.common.layers.StochasticDepth'
model.reference.decoder.transformer.layer.feed_forward.stochastic_depth.mode: 'row'
model.reference.decoder.transformer.layer.feed_forward.structure: 'prenorm'
model.reference.decoder.transformer.layer.klass: 'axlearn.common.attention.TransformerLayer'
model.reference.decoder.transformer.layer.remat_spec['prevent_cse']: False
model.reference.decoder.transformer.layer.remat_spec['policy'].fn: 'axlearn.common.attention._save_and_offload_only_these_names_regex'
model.reference.decoder.transformer.layer.remat_spec['policy'].names_which_can_be_offloaded: None
model.reference.decoder.transformer.layer.remat_spec['policy'].names_which_can_be_saved: '.*([qkvo]_proj|context)'
model.reference.decoder.transformer.layer.remat_spec['policy'].offload_dst: 'pinned_host'
model.reference.decoder.transformer.layer.remat_spec['policy'].offload_src: 'device'
model.reference.decoder.transformer.layer.self_attention.attention.causal: True
model.reference.decoder.transformer.layer.self_attention.attention.dropout.klass: 'axlearn.common.layers.Dropout'
model.reference.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.klass: 'axlearn.common.attention.FusedGroupedQKVLinear'
model.reference.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.bias: False
model.reference.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.klass: 'axlearn.common.attention.MultiheadInputLinear'
model.reference.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.param_partition_spec[0][0]: 'expert'
model.reference.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.param_partition_spec[0][1]: 'fsdp'
model.reference.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.param_partition_spec[0][2]: 'seq'
model.reference.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.param_partition_spec[1]: 'model'
model.reference.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.param_partition_spec[2]: None
model.reference.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.num_kv_heads: 8
model.reference.decoder.transformer.layer.self_attention.attention.input_linear.klass: 'axlearn.common.attention.RoFormerQKVLinear'
model.reference.decoder.transformer.layer.self_attention.attention.input_linear.rope_pos_emb_layer.klass: 'axlearn.common.attention.RoFormerSinusoidalPositionalEmbedding'
model.reference.decoder.transformer.layer.self_attention.attention.input_linear.rope_pos_emb_layer.theta: 500000.0
model.reference.decoder.transformer.layer.self_attention.attention.input_linear.rotary_value: False
model.reference.decoder.transformer.layer.self_attention.attention.key_scale.klass: 'axlearn.common.attention.ScaleKey'
model.reference.decoder.transformer.layer.self_attention.attention.klass: 'axlearn.common.attention.GroupedQueryAttention'
model.reference.decoder.transformer.layer.self_attention.attention.kv_cache.cache_dtype: 'jax.numpy.bfloat16'
model.reference.decoder.transformer.layer.self_attention.attention.kv_cache.klass: 'axlearn.common.kv_cache.kv_cache.KVCache'
model.reference.decoder.transformer.layer.self_attention.attention.num_heads: 32
model.reference.decoder.transformer.layer.self_attention.attention.output_linear.bias: False
model.reference.decoder.transformer.layer.self_attention.attention.output_linear.klass: 'axlearn.common.attention.MultiheadOutputLinear'
model.reference.decoder.transformer.layer.self_attention.attention.output_linear.param_partition_spec[0][0]: 'expert'
model.reference.decoder.transformer.layer.self_attention.attention.output_linear.param_partition_spec[0][1]: 'fsdp'
model.reference.decoder.transformer.layer.self_attention.attention.output_linear.param_partition_spec[0][2]: 'seq'
model.reference.decoder.transformer.layer.self_attention.attention.output_linear.param_partition_spec[1]: 'model'
model.reference.decoder.transformer.layer.self_attention.attention.output_linear.param_partition_spec[2]: None
model.reference.decoder.transformer.layer.self_attention.attention.query_scale.klass: 'axlearn.common.attention.ScaleQuery'
model.reference.decoder.transformer.layer.self_attention.dropout.klass: 'axlearn.common.layers.Dropout'
model.reference.decoder.transformer.layer.self_attention.klass: 'axlearn.common.attention.TransformerAttentionLayer'
model.reference.decoder.transformer.layer.self_attention.norm.eps: 1e-05
model.reference.decoder.transformer.layer.self_attention.norm.forward_dtype: None
model.reference.decoder.transformer.layer.self_attention.norm.klass: 'axlearn.common.layers.RMSNorm'
model.reference.decoder.transformer.layer.self_attention.stochastic_depth.klass: 'axlearn.common.layers.StochasticDepth'
model.reference.decoder.transformer.layer.self_attention.stochastic_depth.mode: 'row'
model.reference.decoder.transformer.layer.self_attention.structure: 'prenorm'
model.reference.decoder.transformer.num_layers: 32
model.reference.decoder.transformer.repeat.drop_output.fn: 'axlearn.common.repeat._drop_by_regex'
model.reference.decoder.transformer.repeat.drop_output.rules[0]: 'module_outputs.*'
model.reference.decoder.transformer.repeat.klass: 'axlearn.common.attention._TransformerRepeat'
model.reference.decoder.vocab_size: 128256
model.reference.dtype: 'jax.numpy.bfloat16'
model.reference.klass: 'axlearn.common.causal_lm.Model'
model.reference.metrics.klass: 'axlearn.common.causal_lm.CompositeLossMetrics'
model.reference.metrics.metrics['lm'].klass: 'axlearn.common.causal_lm.CrossEntropyLossMetrics'
model.reference.metrics.metrics['aux'].klass: 'axlearn.common.causal_lm.AuxLossMetrics'
model.reference.param_init.init_by_param_name['.*weight$'].distribution: 'normal'
model.reference.param_init.init_by_param_name['.*weight$'].fan: 'fan_in'
model.reference.param_init.init_by_param_name['.*weight$'].klass: 'axlearn.common.param_init.WeightInitializer'
model.reference.param_init.init_by_param_name['.*weight$'].scale: 1.0
model.reference.param_init.klass: 'axlearn.common.param_init.DefaultInitializer'
model.sampler.batch_axis_names: None
model.sampler.decoder.attention_mask: None
model.sampler.decoder.decoding.klass: 'axlearn.common.decoder.DecodingLayer'
model.sampler.decoder.dim: 4096
model.sampler.decoder.dropout_rate: 0.0
model.sampler.decoder.dtype: 'jax.numpy.bfloat16'
model.sampler.decoder.emb.dropout.klass: 'axlearn.common.layers.Dropout'
model.sampler.decoder.emb.klass: 'axlearn.common.embedding.TransformerTextEmbeddings'
model.sampler.decoder.emb.token_emb.klass: 'axlearn.common.layers.Embedding'
model.sampler.decoder.emb.token_emb.param_init.init_by_param_name['.*weight$'].distribution: 'normal'
model.sampler.decoder.emb.token_emb.param_init.init_by_param_name['.*weight$'].fan: 'fan_out'
model.sampler.decoder.emb.token_emb.param_init.init_by_param_name['.*weight$'].klass: 'axlearn.common.param_init.WeightInitializer'
model.sampler.decoder.emb.token_emb.param_init.init_by_param_name['.*weight$'].scale: 1.0
model.sampler.decoder.emb.token_emb.param_init.klass: 'axlearn.common.param_init.DefaultInitializer'
model.sampler.decoder.emb.token_emb.param_partition_spec[0]: 'model'
model.sampler.decoder.emb.token_emb.param_partition_spec[1][0]: 'expert'
model.sampler.decoder.emb.token_emb.param_partition_spec[1][1]: 'fsdp'
model.sampler.decoder.emb.token_emb.param_partition_spec[1][2]: 'seq'
model.sampler.decoder.eos_token_id: 128001
model.sampler.decoder.klass: 'axlearn.common.decoder.Decoder'
model.sampler.decoder.lm_head.klass: 'axlearn.common.decoder.LmHead'
model.sampler.decoder.lm_head.param_partition_spec[0]: 'model'
model.sampler.decoder.lm_head.param_partition_spec[1][0]: 'expert'
model.sampler.decoder.lm_head.param_partition_spec[1][1]: 'fsdp'
model.sampler.decoder.lm_head.param_partition_spec[1][2]: 'seq'
model.sampler.decoder.logits_partition_spec[0][0]: 'data'
model.sampler.decoder.logits_partition_spec[0][1]: 'expert'
model.sampler.decoder.logits_partition_spec[0][2]: 'fsdp'
model.sampler.decoder.logits_partition_spec[1]: 'seq'
model.sampler.decoder.logits_partition_spec[2]: 'model'
model.sampler.decoder.output_dropout.klass: 'axlearn.common.layers.Dropout'
model.sampler.decoder.output_norm.eps: 1e-05
model.sampler.decoder.output_norm.forward_dtype: None
model.sampler.decoder.output_norm.klass: 'axlearn.common.layers.RMSNorm'
model.sampler.decoder.pad_token_id: 128004
model.sampler.decoder.transformer.klass: 'axlearn.common.attention.RepeatedTransformerLayer'
model.sampler.decoder.transformer.layer.feed_forward.activation[0]: 'nn.silu'
model.sampler.decoder.transformer.layer.feed_forward.activation[1]: 'linear'
model.sampler.decoder.transformer.layer.feed_forward.dropout.klass: 'axlearn.common.layers.Dropout'
model.sampler.decoder.transformer.layer.feed_forward.hidden_dim.fn: 'axlearn.experiments.text.gpt.common.scale_fn'
model.sampler.decoder.transformer.layer.feed_forward.hidden_dim.round_up_to_multiples_of: 256
model.sampler.decoder.transformer.layer.feed_forward.hidden_dim.scale: 3.5
model.sampler.decoder.transformer.layer.feed_forward.klass: 'axlearn.common.attention.TransformerFeedForwardLayer'
model.sampler.decoder.transformer.layer.feed_forward.linear1.bias: False
model.sampler.decoder.transformer.layer.feed_forward.linear1.klass: 'axlearn.common.layers.Linear'
model.sampler.decoder.transformer.layer.feed_forward.linear1.output_partition_spec[0][0]: 'data'
model.sampler.decoder.transformer.layer.feed_forward.linear1.output_partition_spec[0][1]: 'expert'
model.sampler.decoder.transformer.layer.feed_forward.linear1.output_partition_spec[0][2]: 'fsdp'
model.sampler.decoder.transformer.layer.feed_forward.linear1.output_partition_spec[1]: 'seq'
model.sampler.decoder.transformer.layer.feed_forward.linear1.output_partition_spec[2]: 'model'
model.sampler.decoder.transformer.layer.feed_forward.linear1.param_partition_spec[0][0]: 'expert'
model.sampler.decoder.transformer.layer.feed_forward.linear1.param_partition_spec[0][1]: 'fsdp'
model.sampler.decoder.transformer.layer.feed_forward.linear1.param_partition_spec[0][2]: 'seq'
model.sampler.decoder.transformer.layer.feed_forward.linear1.param_partition_spec[1]: 'model'
model.sampler.decoder.transformer.layer.feed_forward.linear2.bias: False
model.sampler.decoder.transformer.layer.feed_forward.linear2.klass: 'axlearn.common.layers.Linear'
model.sampler.decoder.transformer.layer.feed_forward.linear2.output_partition_spec[0][0]: 'data'
model.sampler.decoder.transformer.layer.feed_forward.linear2.output_partition_spec[0][1]: 'expert'
model.sampler.decoder.transformer.layer.feed_forward.linear2.output_partition_spec[0][2]: 'fsdp'
model.sampler.decoder.transformer.layer.feed_forward.linear2.output_partition_spec[1]: 'seq'
model.sampler.decoder.transformer.layer.feed_forward.linear2.output_partition_spec[2]: 'model'
model.sampler.decoder.transformer.layer.feed_forward.linear2.param_partition_spec[0]: 'model'
model.sampler.decoder.transformer.layer.feed_forward.linear2.param_partition_spec[1][0]: 'expert'
model.sampler.decoder.transformer.layer.feed_forward.linear2.param_partition_spec[1][1]: 'fsdp'
model.sampler.decoder.transformer.layer.feed_forward.linear2.param_partition_spec[1][2]: 'seq'
model.sampler.decoder.transformer.layer.feed_forward.norm.eps: 1e-05
model.sampler.decoder.transformer.layer.feed_forward.norm.forward_dtype: None
model.sampler.decoder.transformer.layer.feed_forward.norm.klass: 'axlearn.common.layers.RMSNorm'
model.sampler.decoder.transformer.layer.feed_forward.residual_weight: 1.0
model.sampler.decoder.transformer.layer.feed_forward.stochastic_depth.klass: 'axlearn.common.layers.StochasticDepth'
model.sampler.decoder.transformer.layer.feed_forward.stochastic_depth.mode: 'row'
model.sampler.decoder.transformer.layer.feed_forward.structure: 'prenorm'
model.sampler.decoder.transformer.layer.klass: 'axlearn.common.attention.TransformerLayer'
model.sampler.decoder.transformer.layer.remat_spec['prevent_cse']: False
model.sampler.decoder.transformer.layer.remat_spec['policy'].fn: 'axlearn.common.attention._save_and_offload_only_these_names_regex'
model.sampler.decoder.transformer.layer.remat_spec['policy'].names_which_can_be_offloaded: None
model.sampler.decoder.transformer.layer.remat_spec['policy'].names_which_can_be_saved: '.*([qkvo]_proj|context)'
model.sampler.decoder.transformer.layer.remat_spec['policy'].offload_dst: 'pinned_host'
model.sampler.decoder.transformer.layer.remat_spec['policy'].offload_src: 'device'
model.sampler.decoder.transformer.layer.self_attention.attention.causal: True
model.sampler.decoder.transformer.layer.self_attention.attention.dropout.klass: 'axlearn.common.layers.Dropout'
model.sampler.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.klass: 'axlearn.common.attention.FusedGroupedQKVLinear'
model.sampler.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.bias: False
model.sampler.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.klass: 'axlearn.common.attention.MultiheadInputLinear'
model.sampler.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.param_partition_spec[0][0]: 'expert'
model.sampler.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.param_partition_spec[0][1]: 'fsdp'
model.sampler.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.param_partition_spec[0][2]: 'seq'
model.sampler.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.param_partition_spec[1]: 'model'
model.sampler.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.layer.param_partition_spec[2]: None
model.sampler.decoder.transformer.layer.self_attention.attention.input_linear.input_linear.num_kv_heads: 8
model.sampler.decoder.transformer.layer.self_attention.attention.input_linear.klass: 'axlearn.common.attention.RoFormerQKVLinear'
model.sampler.decoder.transformer.layer.self_attention.attention.input_linear.rope_pos_emb_layer.klass: 'axlearn.common.attention.RoFormerSinusoidalPositionalEmbedding'
model.sampler.decoder.transformer.layer.self_attention.attention.input_linear.rope_pos_emb_layer.theta: 500000.0
model.sampler.decoder.transformer.layer.self_attention.attention.input_linear.rotary_value: False
model.sampler.decoder.transformer.layer.self_attention.attention.key_scale.klass: 'axlearn.common.attention.ScaleKey'
model.sampler.decoder.transformer.layer.self_attention.attention.klass: 'axlearn.common.attention.GroupedQueryAttention'
model.sampler.decoder.transformer.layer.self_attention.attention.kv_cache.cache_dtype: 'jax.numpy.bfloat16'
model.sampler.decoder.transformer.layer.self_attention.attention.kv_cache.klass: 'axlearn.common.kv_cache.kv_cache.KVCache'
model.sampler.decoder.transformer.layer.self_attention.attention.num_heads: 32
model.sampler.decoder.transformer.layer.self_attention.attention.output_linear.bias: False
model.sampler.decoder.transformer.layer.self_attention.attention.output_linear.klass: 'axlearn.common.attention.MultiheadOutputLinear'
model.sampler.decoder.transformer.layer.self_attention.attention.output_linear.param_partition_spec[0][0]: 'expert'
model.sampler.decoder.transformer.layer.self_attention.attention.output_linear.param_partition_spec[0][1]: 'fsdp'
model.sampler.decoder.transformer.layer.self_attention.attention.output_linear.param_partition_spec[0][2]: 'seq'
model.sampler.decoder.transformer.layer.self_attention.attention.output_linear.param_partition_spec[1]: 'model'
model.sampler.decoder.transformer.layer.self_attention.attention.output_linear.param_partition_spec[2]: None
model.sampler.decoder.transformer.layer.self_attention.attention.query_scale.klass: 'axlearn.common.attention.ScaleQuery'
model.sampler.decoder.transformer.layer.self_attention.dropout.klass: 'axlearn.common.layers.Dropout'
model.sampler.decoder.transformer.layer.self_attention.klass: 'axlearn.common.attention.TransformerAttentionLayer'
model.sampler.decoder.transformer.layer.self_attention.norm.eps: 1e-05
model.sampler.decoder.transformer.layer.self_attention.norm.forward_dtype: None
model.sampler.decoder.transformer.layer.self_attention.norm.klass: 'axlearn.common.layers.RMSNorm'
model.sampler.decoder.transformer.layer.self_attention.stochastic_depth.klass: 'axlearn.common.layers.StochasticDepth'
model.sampler.decoder.transformer.layer.self_attention.stochastic_depth.mode: 'row'
model.sampler.decoder.transformer.layer.self_attention.structure: 'prenorm'
model.sampler.decoder.transformer.num_layers: 32
model.sampler.decoder.transformer.repeat.drop_output.fn: 'axlearn.common.repeat._drop_by_regex'
model.sampler.decoder.transformer.repeat.drop_output.rules[0]: 'module_outputs.*'
model.sampler.decoder.transformer.repeat.klass: 'axlearn.common.attention._TransformerRepeat'
model.sampler.decoder.vocab_size: 128256
model.sampler.dtype: 'jax.numpy.bfloat16'
model.sampler.klass: 'axlearn.common.causal_lm.Model'
model.sampler.metrics.klass: 'axlearn.common.causal_lm.CompositeLossMetrics'
model.sampler.metrics.metrics['lm'].klass: 'axlearn.common.causal_lm.CrossEntropyLossMetrics'
model.sampler.metrics.metrics['aux'].klass: 'axlearn.common.causal_lm.AuxLossMetrics'
model.sampler.param_init.init_by_param_name['.*weight$'].distribution: 'normal'
model.sampler.param_init.init_by_param_name['.*weight$'].fan: 'fan_in'
model.sampler.param_init.init_by_param_name['.*weight$'].klass: 'axlearn.common.param_init.WeightInitializer'
model.sampler.param_init.init_by_param_name['.*weight$'].scale: 1.0
model.sampler.param_init.klass: 'axlearn.common.param_init.DefaultInitializer'
name: 'grpo_trainer'
num_generations: 2
prune_empty_state_updates: True
recorder.fn: '__main__.<lambda>'
reward_type: 'gsm8k'
save_input_iterator: False
start_trace_process_indices[0]: 0
summary_writer.klass: 'axlearn.common.summary_writer.SummaryWriter'
summary_writer.max_queue: 1000
summary_writer.write_every_n_steps: 1
vocab.filename: 'Llama-3-tokenizer.json'
vocab.klass: 'axlearn.experiments.text.gpt.grpo_native_example.GRPOV3Vocabulary'
watchdog_timeout_seconds: 3600%
```
## model_analysis
```shell
➜  ~ gcloud storage cat gs://cloud-tpu-multipod-dev-axlearn/users/ericshen/eshen-v6e-rl-grpo-pathways/1779314896/model_analysis.txt | tr '\t' '\n'
##################### Model analysis #####################
## Parameters:
 525336576 [128256, 4096]       actor/decoder/emb/token_emb/weight
 525336576 (128256, 4096)       actor/decoder/lm_head/weight
      4096 [4096]               actor/decoder/output_norm/scale
1879048192 (32, 4096, 14336)    actor/decoder/transformer/repeat/layer/feed_forward/linear1_0/weight
1879048192 (32, 4096, 14336)    actor/decoder/transformer/repeat/layer/feed_forward/linear1_1/weight
1879048192 (32, 14336, 4096)    actor/decoder/transformer/repeat/layer/feed_forward/linear2/weight
    131072 (32, 4096)           actor/decoder/transformer/repeat/layer/feed_forward/norm/scale
 805306368 (32, 4096, 48, 128)  actor/decoder/transformer/repeat/layer/self_attention/attention/i_proj/i_proj/qkv_proj/weight
 536870912 (32, 4096, 32, 128)  actor/decoder/transformer/repeat/layer/self_attention/attention/o_proj/weight
    131072 (32, 4096)           actor/decoder/transformer/repeat/layer/self_attention/norm/scale
 525336576 [128256, 4096]       reference/decoder/emb/token_emb/weight
 525336576 (128256, 4096)       reference/decoder/lm_head/weight
      4096 [4096]               reference/decoder/output_norm/scale
1879048192 (32, 4096, 14336)    reference/decoder/transformer/repeat/layer/feed_forward/linear1_0/weight
1879048192 (32, 4096, 14336)    reference/decoder/transformer/repeat/layer/feed_forward/linear1_1/weight
1879048192 (32, 14336, 4096)    reference/decoder/transformer/repeat/layer/feed_forward/linear2/weight
    131072 (32, 4096)           reference/decoder/transformer/repeat/layer/feed_forward/norm/scale
 805306368 (32, 4096, 48, 128)  reference/decoder/transformer/repeat/layer/self_attention/attention/i_proj/i_proj/qkv_proj/weight
 536870912 (32, 4096, 32, 128)  reference/decoder/transformer/repeat/layer/self_attention/attention/o_proj/weight
    131072 (32, 4096)           reference/decoder/transformer/repeat/layer/self_attention/norm/scale
 525336576 [128256, 4096]       sampler/decoder/emb/token_emb/weight
 525336576 (128256, 4096)       sampler/decoder/lm_head/weight
      4096 [4096]               sampler/decoder/output_norm/scale
1879048192 (32, 4096, 14336)    sampler/decoder/transformer/repeat/layer/feed_forward/linear1_0/weight
1879048192 (32, 4096, 14336)    sampler/decoder/transformer/repeat/layer/feed_forward/linear1_1/weight
1879048192 (32, 14336, 4096)    sampler/decoder/transformer/repeat/layer/feed_forward/linear2/weight
    131072 (32, 4096)           sampler/decoder/transformer/repeat/layer/feed_forward/norm/scale
 805306368 (32, 4096, 48, 128)  sampler/decoder/transformer/repeat/layer/self_attention/attention/i_proj/i_proj/qkv_proj/weight
 536870912 (32, 4096, 32, 128)  sampler/decoder/transformer/repeat/layer/self_attention/attention/o_proj/weight
    131072 (32, 4096)           sampler/decoder/transformer/repeat/layer/self_attention/norm/scale
Total number of model params: 24,090,783,744
## Trainer States:
State: prng_key=uint32((4,)) mesh_axes=ParameterSpec(shape=[4], dtype=<class 'jax.numpy.uint32'>, mesh_axes=PartitionSpec(None,), memory_kind=None, initializer=None, factorization=None, fan_axes=None, weight_decay_scale=None)
State: model/actor/decoder/emb/token_emb/weight=bfloat16((128256, 4096)) mesh_axes=ParameterSpec(shape=[128256, 4096], dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec('model', ('expert', 'fsdp', 'seq')), memory_kind=None, initializer=None, factorization=None, fan_axes=FanAxes(in_axis=-2, out_axis=-1, batch_axis=()), weight_decay_scale=None)
State: model/actor/decoder/lm_head/weight=bfloat16((128256, 4096)) mesh_axes=ParameterSpec(shape=(128256, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec('model', ('expert', 'fsdp', 'seq')), memory_kind=None, initializer=None, factorization=None, fan_axes=FanAxes(in_axis=-2, out_axis=-1, batch_axis=()), weight_decay_scale=None)
State: model/actor/decoder/output_norm/scale=bfloat16((4096,)) mesh_axes=ParameterSpec(shape=[4096], dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None,), memory_kind=None, initializer=None, factorization=None, fan_axes=None, weight_decay_scale=None)
State: model/actor/decoder/transformer/repeat/layer/feed_forward/linear1_0/weight=bfloat16((32, 4096, 14336)) mesh_axes=ParameterSpec(shape=(32, 4096, 14336), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model'), memory_kind=None, initializer=None, factorization=FactorizationSpec(axes=[None, 'row', 'col']), fan_axes=FanAxes(in_axis=(-2,), out_axis=(-1,), batch_axis=(0,)), weight_decay_scale=None)
State: model/actor/decoder/transformer/repeat/layer/feed_forward/linear1_1/weight=bfloat16((32, 4096, 14336)) mesh_axes=ParameterSpec(shape=(32, 4096, 14336), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model'), memory_kind=None, initializer=None, factorization=FactorizationSpec(axes=[None, 'row', 'col']), fan_axes=FanAxes(in_axis=(-2,), out_axis=(-1,), batch_axis=(0,)), weight_decay_scale=None)
State: model/actor/decoder/transformer/repeat/layer/feed_forward/linear2/weight=bfloat16((32, 14336, 4096)) mesh_axes=ParameterSpec(shape=(32, 14336, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, 'model', ('expert', 'fsdp', 'seq')), memory_kind=None, initializer=None, factorization=FactorizationSpec(axes=[None, 'row', 'col']), fan_axes=FanAxes(in_axis=(-2,), out_axis=(-1,), batch_axis=(0,)), weight_decay_scale=None)
State: model/actor/decoder/transformer/repeat/layer/feed_forward/norm/scale=bfloat16((32, 4096)) mesh_axes=ParameterSpec(shape=(32, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, None), memory_kind=None, initializer=None, factorization=None, fan_axes=None, weight_decay_scale=None)
State: model/actor/decoder/transformer/repeat/layer/self_attention/attention/i_proj/i_proj/qkv_proj/weight=bfloat16((32, 4096, 48, 128)) mesh_axes=ParameterSpec(shape=(32, 4096, 48, 128), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model', None), memory_kind=None, initializer=None, factorization=FactorizationSpec(axes=[None, 'row', None, 'col']), fan_axes=FanAxes(in_axis=(1,), out_axis=(2, 3), batch_axis=(0,)), weight_decay_scale=None)
State: model/actor/decoder/transformer/repeat/layer/self_attention/attention/o_proj/weight=bfloat16((32, 4096, 32, 128)) mesh_axes=ParameterSpec(shape=(32, 4096, 32, 128), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model', None), memory_kind=None, initializer=None, factorization=FactorizationSpec(axes=[None, 'row', None, 'col']), fan_axes=FanAxes(in_axis=(2, 3), out_axis=(1,), batch_axis=(0,)), weight_decay_scale=None)
State: model/actor/decoder/transformer/repeat/layer/self_attention/norm/scale=bfloat16((32, 4096)) mesh_axes=ParameterSpec(shape=(32, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, None), memory_kind=None, initializer=None, factorization=None, fan_axes=None, weight_decay_scale=None)
State: model/reference/decoder/emb/token_emb/weight=bfloat16((128256, 4096)) mesh_axes=ParameterSpec(shape=[128256, 4096], dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec('model', ('expert', 'fsdp', 'seq')), memory_kind=None, initializer=None, factorization=None, fan_axes=FanAxes(in_axis=-2, out_axis=-1, batch_axis=()), weight_decay_scale=None)
State: model/reference/decoder/lm_head/weight=bfloat16((128256, 4096)) mesh_axes=ParameterSpec(shape=(128256, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec('model', ('expert', 'fsdp', 'seq')), memory_kind=None, initializer=None, factorization=None, fan_axes=FanAxes(in_axis=-2, out_axis=-1, batch_axis=()), weight_decay_scale=None)
State: model/reference/decoder/output_norm/scale=bfloat16((4096,)) mesh_axes=ParameterSpec(shape=[4096], dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None,), memory_kind=None, initializer=None, factorization=None, fan_axes=None, weight_decay_scale=None)
State: model/reference/decoder/transformer/repeat/layer/feed_forward/linear1_0/weight=bfloat16((32, 4096, 14336)) mesh_axes=ParameterSpec(shape=(32, 4096, 14336), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model'), memory_kind=None, initializer=None, factorization=FactorizationSpec(axes=[None, 'row', 'col']), fan_axes=FanAxes(in_axis=(-2,), out_axis=(-1,), batch_axis=(0,)), weight_decay_scale=None)
State: model/reference/decoder/transformer/repeat/layer/feed_forward/linear1_1/weight=bfloat16((32, 4096, 14336)) mesh_axes=ParameterSpec(shape=(32, 4096, 14336), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model'), memory_kind=None, initializer=None, factorization=FactorizationSpec(axes=[None, 'row', 'col']), fan_axes=FanAxes(in_axis=(-2,), out_axis=(-1,), batch_axis=(0,)), weight_decay_scale=None)
State: model/reference/decoder/transformer/repeat/layer/feed_forward/linear2/weight=bfloat16((32, 14336, 4096)) mesh_axes=ParameterSpec(shape=(32, 14336, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, 'model', ('expert', 'fsdp', 'seq')), memory_kind=None, initializer=None, factorization=FactorizationSpec(axes=[None, 'row', 'col']), fan_axes=FanAxes(in_axis=(-2,), out_axis=(-1,), batch_axis=(0,)), weight_decay_scale=None)
State: model/reference/decoder/transformer/repeat/layer/feed_forward/norm/scale=bfloat16((32, 4096)) mesh_axes=ParameterSpec(shape=(32, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, None), memory_kind=None, initializer=None, factorization=None, fan_axes=None, weight_decay_scale=None)
State: model/reference/decoder/transformer/repeat/layer/self_attention/attention/i_proj/i_proj/qkv_proj/weight=bfloat16((32, 4096, 48, 128)) mesh_axes=ParameterSpec(shape=(32, 4096, 48, 128), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model', None), memory_kind=None, initializer=None, factorization=FactorizationSpec(axes=[None, 'row', None, 'col']), fan_axes=FanAxes(in_axis=(1,), out_axis=(2, 3), batch_axis=(0,)), weight_decay_scale=None)
State: model/reference/decoder/transformer/repeat/layer/self_attention/attention/o_proj/weight=bfloat16((32, 4096, 32, 128)) mesh_axes=ParameterSpec(shape=(32, 4096, 32, 128), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model', None), memory_kind=None, initializer=None, factorization=FactorizationSpec(axes=[None, 'row', None, 'col']), fan_axes=FanAxes(in_axis=(2, 3), out_axis=(1,), batch_axis=(0,)), weight_decay_scale=None)
State: model/reference/decoder/transformer/repeat/layer/self_attention/norm/scale=bfloat16((32, 4096)) mesh_axes=ParameterSpec(shape=(32, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, None), memory_kind=None, initializer=None, factorization=None, fan_axes=None, weight_decay_scale=None)
State: model/sampler/decoder/emb/token_emb/weight=bfloat16((128256, 4096)) mesh_axes=ParameterSpec(shape=[128256, 4096], dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec('model', ('expert', 'fsdp', 'seq')), memory_kind=None, initializer=None, factorization=None, fan_axes=FanAxes(in_axis=-2, out_axis=-1, batch_axis=()), weight_decay_scale=None)
State: model/sampler/decoder/lm_head/weight=bfloat16((128256, 4096)) mesh_axes=ParameterSpec(shape=(128256, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec('model', ('expert', 'fsdp', 'seq')), memory_kind=None, initializer=None, factorization=None, fan_axes=FanAxes(in_axis=-2, out_axis=-1, batch_axis=()), weight_decay_scale=None)
State: model/sampler/decoder/output_norm/scale=bfloat16((4096,)) mesh_axes=ParameterSpec(shape=[4096], dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None,), memory_kind=None, initializer=None, factorization=None, fan_axes=None, weight_decay_scale=None)
State: model/sampler/decoder/transformer/repeat/layer/feed_forward/linear1_0/weight=bfloat16((32, 4096, 14336)) mesh_axes=ParameterSpec(shape=(32, 4096, 14336), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model'), memory_kind=None, initializer=None, factorization=FactorizationSpec(axes=[None, 'row', 'col']), fan_axes=FanAxes(in_axis=(-2,), out_axis=(-1,), batch_axis=(0,)), weight_decay_scale=None)
State: model/sampler/decoder/transformer/repeat/layer/feed_forward/linear1_1/weight=bfloat16((32, 4096, 14336)) mesh_axes=ParameterSpec(shape=(32, 4096, 14336), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model'), memory_kind=None, initializer=None, factorization=FactorizationSpec(axes=[None, 'row', 'col']), fan_axes=FanAxes(in_axis=(-2,), out_axis=(-1,), batch_axis=(0,)), weight_decay_scale=None)
State: model/sampler/decoder/transformer/repeat/layer/feed_forward/linear2/weight=bfloat16((32, 14336, 4096)) mesh_axes=ParameterSpec(shape=(32, 14336, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, 'model', ('expert', 'fsdp', 'seq')), memory_kind=None, initializer=None, factorization=FactorizationSpec(axes=[None, 'row', 'col']), fan_axes=FanAxes(in_axis=(-2,), out_axis=(-1,), batch_axis=(0,)), weight_decay_scale=None)
State: model/sampler/decoder/transformer/repeat/layer/feed_forward/norm/scale=bfloat16((32, 4096)) mesh_axes=ParameterSpec(shape=(32, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, None), memory_kind=None, initializer=None, factorization=None, fan_axes=None, weight_decay_scale=None)
State: model/sampler/decoder/transformer/repeat/layer/self_attention/attention/i_proj/i_proj/qkv_proj/weight=bfloat16((32, 4096, 48, 128)) mesh_axes=ParameterSpec(shape=(32, 4096, 48, 128), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model', None), memory_kind=None, initializer=None, factorization=FactorizationSpec(axes=[None, 'row', None, 'col']), fan_axes=FanAxes(in_axis=(1,), out_axis=(2, 3), batch_axis=(0,)), weight_decay_scale=None)
State: model/sampler/decoder/transformer/repeat/layer/self_attention/attention/o_proj/weight=bfloat16((32, 4096, 32, 128)) mesh_axes=ParameterSpec(shape=(32, 4096, 32, 128), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model', None), memory_kind=None, initializer=None, factorization=FactorizationSpec(axes=[None, 'row', None, 'col']), fan_axes=FanAxes(in_axis=(2, 3), out_axis=(1,), batch_axis=(0,)), weight_decay_scale=None)
State: model/sampler/decoder/transformer/repeat/layer/self_attention/norm/scale=bfloat16((32, 4096)) mesh_axes=ParameterSpec(shape=(32, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, None), memory_kind=None, initializer=None, factorization=None, fan_axes=None, weight_decay_scale=None)
State: learner/optimizer/0/count=int32(()) mesh_axes=TensorSpec(shape=(), dtype=dtype('int32'), mesh_axes=PartitionSpec(), memory_kind=None)
State: learner/optimizer/0/mu/actor/decoder/emb/token_emb/weight=bfloat16((128256, 4096)) mesh_axes=TensorSpec(shape=[128256, 4096], dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec('model', ('expert', 'fsdp', 'seq')), memory_kind=None)
State: learner/optimizer/0/mu/actor/decoder/lm_head/weight=bfloat16((128256, 4096)) mesh_axes=TensorSpec(shape=(128256, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec('model', ('expert', 'fsdp', 'seq')), memory_kind=None)
State: learner/optimizer/0/mu/actor/decoder/output_norm/scale=bfloat16((4096,)) mesh_axes=TensorSpec(shape=[4096], dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None,), memory_kind=None)
State: learner/optimizer/0/mu/actor/decoder/transformer/repeat/layer/feed_forward/linear1_0/weight=bfloat16((32, 4096, 14336)) mesh_axes=TensorSpec(shape=(32, 4096, 14336), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model'), memory_kind=None)
State: learner/optimizer/0/mu/actor/decoder/transformer/repeat/layer/feed_forward/linear1_1/weight=bfloat16((32, 4096, 14336)) mesh_axes=TensorSpec(shape=(32, 4096, 14336), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model'), memory_kind=None)
State: learner/optimizer/0/mu/actor/decoder/transformer/repeat/layer/feed_forward/linear2/weight=bfloat16((32, 14336, 4096)) mesh_axes=TensorSpec(shape=(32, 14336, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, 'model', ('expert', 'fsdp', 'seq')), memory_kind=None)
State: learner/optimizer/0/mu/actor/decoder/transformer/repeat/layer/feed_forward/norm/scale=bfloat16((32, 4096)) mesh_axes=TensorSpec(shape=(32, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, None), memory_kind=None)
State: learner/optimizer/0/mu/actor/decoder/transformer/repeat/layer/self_attention/attention/i_proj/i_proj/qkv_proj/weight=bfloat16((32, 4096, 48, 128)) mesh_axes=TensorSpec(shape=(32, 4096, 48, 128), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model', None), memory_kind=None)
State: learner/optimizer/0/mu/actor/decoder/transformer/repeat/layer/self_attention/attention/o_proj/weight=bfloat16((32, 4096, 32, 128)) mesh_axes=TensorSpec(shape=(32, 4096, 32, 128), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model', None), memory_kind=None)
State: learner/optimizer/0/mu/actor/decoder/transformer/repeat/layer/self_attention/norm/scale=bfloat16((32, 4096)) mesh_axes=TensorSpec(shape=(32, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, None), memory_kind=None)
State: learner/optimizer/0/nu/actor/decoder/emb/token_emb/weight=bfloat16((128256, 4096)) mesh_axes=TensorSpec(shape=[128256, 4096], dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec('model', ('expert', 'fsdp', 'seq')), memory_kind=None)
State: learner/optimizer/0/nu/actor/decoder/lm_head/weight=bfloat16((128256, 4096)) mesh_axes=TensorSpec(shape=(128256, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec('model', ('expert', 'fsdp', 'seq')), memory_kind=None)
State: learner/optimizer/0/nu/actor/decoder/output_norm/scale=bfloat16((4096,)) mesh_axes=TensorSpec(shape=[4096], dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None,), memory_kind=None)
State: learner/optimizer/0/nu/actor/decoder/transformer/repeat/layer/feed_forward/linear1_0/weight=bfloat16((32, 4096, 14336)) mesh_axes=TensorSpec(shape=(32, 4096, 14336), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model'), memory_kind=None)
State: learner/optimizer/0/nu/actor/decoder/transformer/repeat/layer/feed_forward/linear1_1/weight=bfloat16((32, 4096, 14336)) mesh_axes=TensorSpec(shape=(32, 4096, 14336), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model'), memory_kind=None)
State: learner/optimizer/0/nu/actor/decoder/transformer/repeat/layer/feed_forward/linear2/weight=bfloat16((32, 14336, 4096)) mesh_axes=TensorSpec(shape=(32, 14336, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, 'model', ('expert', 'fsdp', 'seq')), memory_kind=None)
State: learner/optimizer/0/nu/actor/decoder/transformer/repeat/layer/feed_forward/norm/scale=bfloat16((32, 4096)) mesh_axes=TensorSpec(shape=(32, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, None), memory_kind=None)
State: learner/optimizer/0/nu/actor/decoder/transformer/repeat/layer/self_attention/attention/i_proj/i_proj/qkv_proj/weight=bfloat16((32, 4096, 48, 128)) mesh_axes=TensorSpec(shape=(32, 4096, 48, 128), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model', None), memory_kind=None)
State: learner/optimizer/0/nu/actor/decoder/transformer/repeat/layer/self_attention/attention/o_proj/weight=bfloat16((32, 4096, 32, 128)) mesh_axes=TensorSpec(shape=(32, 4096, 32, 128), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, ('expert', 'fsdp', 'seq'), 'model', None), memory_kind=None)
State: learner/optimizer/0/nu/actor/decoder/transformer/repeat/layer/self_attention/norm/scale=bfloat16((32, 4096)) mesh_axes=TensorSpec(shape=(32, 4096), dtype=<class 'jax.numpy.bfloat16'>, mesh_axes=PartitionSpec(None, None), memory_kind=None)
State: learner/optimizer/2/count=int32(()) mesh_axes=TensorSpec(shape=[], dtype=<class 'jax.numpy.int32'>, mesh_axes=PartitionSpec(), memory_kind=None)
Training state size: 74.79 GiB
Training state size (partitioned): 4.68 GiB
Max training state size (partitioned): 4.68 GiB
##########################################################%

```

## Training Metrics

### Definition
---

#### 1. **`loss`** (Policy Gradient Objective)
*   **Definition**: The overall composite objective minimized by the optimizer. It combines the negative GRPO surrogate policy loss (which pushes the Actor to produce higher-advantage tokens) and the KL divergence penalty.
*   **Plain-Text Formula**:
    `Loss = -(Average of [ Minimum( Ratio * Advantage, Clipped_Ratio * Advantage ) ]) + beta * KL_Divergence`
*   **Intuition**: This is the "steering wheel" of your training. A smaller/more negative loss indicates the Actor model is successfully optimizing its parameters to maximize mathematical rewards while remaining safely constrained.

---

#### 2. **`kl_divergence`** (Kullback-Leibler Divergence)
*   **Definition**: A statistical measure of how much the active Actor model's token probability distribution has drifted/diverged from the frozen, pre-trained Reference model's token probability distribution.
*   **Plain-Text Formula**:
    `KL_Divergence = Average of [ (Reference_Probability / Actor_Probability) - log(Reference_Probability / Actor_Probability) - 1 ]`
*   **Intuition**: This measures the Actor's "deviation penalty." We want this to remain stable and small (e.g. `0.002 - 0.005`). If it surges too high, the model is "forgetting" its pre-trained knowledge or collapsing into gibberish to exploit the rewards.

---

#### 3. **`mean_advantage`**
*   **Definition**: The average relative advantage score across all generations in the batch.
*   **Plain-Text Formula**:
    `Mean_Advantage = (Sum of all Advantages in Batch) / (Total Number of Rollouts in Batch) = 0.0`
*   **Intuition**: Because GRPO computes relative advantages by Z-score normalizing the rewards *inside* each sibling group, the above-average advantages perfectly cancel out the below-average advantages. Thus, this metric is **mathematically, strictly always exactly `0.0`** by design.

---

#### 4. **`mean_reward`** (Absolute Scoreboard Accuracy)
*   **Definition**: The percentage of generated rollouts in the current training batch that successfully solved the math problem correctly and matched the ground-truth final answer (received a reward of `1.0`).
*   **Plain-Text Formula**:
    `Mean_Reward = (Number of Correct Rollouts in Batch) / (Total Number of Rollouts in Batch)`
*   **Intuition**: This is the **absolute scoreboard** showing how smart your model is in the real world. A value of `0.714` means exactly **71.4% of all math questions solved in this batch were completely correct.**

---

#### 5. **`std_advantage`** (The Learning Signal Contrast)
*   **Definition**: The standard deviation (the spread/variance) of the relative advantages. It measures the **strength and presence of contrast** inside your training groups.
*   **Plain-Text Formula**:
    `Std_Advantage = Standard_Deviation(Advantages_in_Batch)`
*   **Intuition**: Think of this as the **learning signal strength**.
    *   If it is positive ($>0.0$): It confirms that in some groups, some siblings were correct and others were incorrect. This **contrast** provides a strong active gradient signal to the Actor model.
    *   If it were $0.0$: All siblings got the exact same reward (no contrast), and the model would learn nothing.



### Step Time Calculation
To calculate the average step time across the entire range from Step 1 to Step 1000, I used the total wall-clock time elapsed.

#### Overall Average Calculation

The training run started at **2026-05-20 22:18:12** ([apple rl grpo decouple...](https://paste.googleplex.com/5465377663483904?content_ref=2026+05+20t22+18+12+092271991z)) and reached Step 1000 at **2026-05-23 09:13:19** ([apple rl grpo decouple...](https://paste.googleplex.com/5465377663483904?content_ref=2026+05+23t09+13+19+974985949z)).

*   **Total Elapsed Time:** 2 days, 10 hours, 55 minutes, and 7 seconds ($212,107.88$ seconds).
*   **Total Steps Completed:** 999 steps (from Step 1 to Step 1000).
*   **Average Step Time (Wall-Clock):** **3.54 minutes per step** (or $212.32$ seconds).

#### Comparison to Active Training Time
This "raw" average includes all system overhead and downtime (such as the ~1.5-hour gaps between steps 100-110 and 130-140).

| Metric | Duration (Minutes) | Duration (Seconds) |
| :--- | :---: | :---: |
| **Wall-Clock Average (incl. gaps)** | **3.54** | **212.32** |
| Active Training Average (excl. gaps) | 3.45 | 206.94 |

The small difference (approx. 5 seconds) between the wall-clock average and the active average indicates that while there were some process restarts, the training run was largely consistent throughout the 1000 steps. For example, the gap between Step 790 (20:02:28) and Step 800 (21:50:22) lasted approximately 107 minutes for 10 steps, averaging 10.7 minutes/step. This segment was excluded form the calculation.

### Metrics Log
| timestamp | message |
| :--- | :--- |
| 2026-05-20T22:18:12.092271991Z | grpo_trainer process   0 step        1] loss=-3.457978e-05 aux={'kl_divergence': 0.00066375732421875, 'loss': -3.457978164078668e-05, 'mean_advantage': 0.0, 'mean_reward': 0.68359375, 'std_advantage': 0.36970269680023193} |
| 2026-05-20T22:21:14.885783794Z | grpo_trainer process   0 step        2] loss=0.00028741112 aux={'kl_divergence': 0.007476806640625, 'loss': 0.00028741112328134477, 'mean_advantage': 0.0, 'mean_reward': 0.70703125, 'std_advantage': 0.4001386761665344} |
| 2026-05-20T22:24:01.066044014Z | grpo_trainer process   0 step        3] loss=0.00115081 aux={'kl_divergence': 0.028076171875, 'loss': 0.0011508100433275104, 'mean_advantage': 0.0, 'mean_reward': 0.7734375, 'std_advantage': 0.4049890339374542} |
| 2026-05-20T22:26:45.383707212Z | grpo_trainer process   0 step        4] loss=0.0013424202 aux={'kl_divergence': 0.03369140625, 'loss': 0.001342420233413577, 'mean_advantage': 0.0, 'mean_reward': 0.81640625, 'std_advantage': 0.3247136175632477} |
| 2026-05-20T22:29:27.062902737Z | grpo_trainer process   0 step        5] loss=0.0015679549 aux={'kl_divergence': 0.03955078125, 'loss': 0.001567954896017909, 'mean_advantage': 0.0, 'mean_reward': 0.79296875, 'std_advantage': 0.36970269680023193} |
| 2026-05-20T22:43:12.758484698Z | grpo_trainer process   0 step       10] loss=0.0018605773 aux={'kl_divergence': 0.046875, 'loss': 0.0018605772638693452, 'mean_advantage': 0.0, 'mean_reward': 0.78125, 'std_advantage': 0.34227824211120605} |
| 2026-05-20T23:11:27.154809090Z | grpo_trainer process   0 step       20] loss=0.0017447168 aux={'kl_divergence': 0.04345703125, 'loss': 0.0017447167774662375, 'mean_advantage': 0.0, 'mean_reward': 0.85546875, 'std_advantage': 0.3124558627605438} |
| 2026-05-20T23:40:07.707870475Z | grpo_trainer process   0 step       30] loss=0.0014684689 aux={'kl_divergence': 0.038818359375, 'loss': 0.0014684689231216908, 'mean_advantage': 0.0, 'mean_reward': 0.80078125, 'std_advantage': 0.34793609380722046} |
| 2026-05-21T00:09:13.667881221Z | grpo_trainer process   0 step       40] loss=0.002723298 aux={'kl_divergence': 0.06787109375, 'loss': 0.0027232980355620384, 'mean_advantage': 0.0, 'mean_reward': 0.703125, 'std_advantage': 0.34227821230888367} |
| 2026-05-21T00:38:07.962246864Z | grpo_trainer process   0 step       50] loss=0.002641279 aux={'kl_divergence': 0.06640625, 'loss': 0.0026412790175527334, 'mean_advantage': 0.0, 'mean_reward': 0.74609375, 'std_advantage': 0.4001387059688568} |
| 2026-05-21T01:06:50.863663753Z | grpo_trainer process   0 step       60] loss=0.0026504158 aux={'kl_divergence': 0.06640625, 'loss': 0.0026504157576709986, 'mean_advantage': 0.0, 'mean_reward': 0.7734375, 'std_advantage': 0.34227821230888367} |
| 2026-05-21T01:34:48.522336833Z | grpo_trainer process   0 step       70] loss=0.0018068241 aux={'kl_divergence': 0.044921875, 'loss': 0.0018068241188302636, 'mean_advantage': 0.0, 'mean_reward': 0.81640625, 'std_advantage': 0.3124558627605438} |
| 2026-05-21T02:02:55.233805252Z | grpo_trainer process   0 step       80] loss=0.0013492543 aux={'kl_divergence': 0.03369140625, 'loss': 0.0013492542784661055, 'mean_advantage': 0.0, 'mean_reward': 0.8515625, 'std_advantage': 0.2931095361709595} |
| 2026-05-21T02:31:21.249166640Z | grpo_trainer process   0 step       90] loss=0.0015166044 aux={'kl_divergence': 0.037353515625, 'loss': 0.0015166044468060136, 'mean_advantage': 0.0, 'mean_reward': 0.84765625, 'std_advantage': 0.2996971309185028} |
| 2026-05-21T02:59:34.445144393Z | grpo_trainer process   0 step      100] loss=0.0020650243 aux={'kl_divergence': 0.0517578125, 'loss': 0.0020650243386626244, 'mean_advantage': 0.0, 'mean_reward': 0.80859375, 'std_advantage': 0.3247136175632477} |
| 2026-05-21T04:29:48.562039133Z | grpo_trainer process   0 step      110] loss=0.0010812202 aux={'kl_divergence': 0.0274658203125, 'loss': 0.0010812202235683799, 'mean_advantage': 0.0, 'mean_reward': 0.76171875, 'std_advantage': 0.34793609380722046} |
| 2026-05-21T04:56:44.706656632Z | grpo_trainer process   0 step      120] loss=0.0010059193 aux={'kl_divergence': 0.0255126953125, 'loss': 0.0010059193009510636, 'mean_advantage': 0.0, 'mean_reward': 0.7890625, 'std_advantage': 0.3186436891555786} |
| 2026-05-21T05:23:43.210760675Z | grpo_trainer process   0 step      130] loss=0.0020528017 aux={'kl_divergence': 0.051025390625, 'loss': 0.002052801661193371, 'mean_advantage': 0.0, 'mean_reward': 0.73828125, 'std_advantage': 0.33652523159980774} |
| 2026-05-21T06:53:57.668558349Z | grpo_trainer process   0 step      140] loss=0.001375706 aux={'kl_divergence': 0.03466796875, 'loss': 0.0013757060514762998, 'mean_advantage': 0.0, 'mean_reward': 0.80859375, 'std_advantage': 0.35898441076278687} |
| 2026-05-21T07:25:51.180543689Z | grpo_trainer process   0 step      150] loss=0.0012500313 aux={'kl_divergence': 0.031005859375, 'loss': 0.0012500312877818942, 'mean_advantage': 0.0, 'mean_reward': 0.8359375, 'std_advantage': 0.34227821230888367} |
| 2026-05-21T07:57:43.633816818Z | grpo_trainer process   0 step      160] loss=0.0011262903 aux={'kl_divergence': 0.0281982421875, 'loss': 0.0011262902989983559, 'mean_advantage': 0.0, 'mean_reward': 0.80078125, 'std_advantage': 0.3124558627605438} |
| 2026-05-21T08:51:38.209021221Z | grpo_trainer process   0 step      170] loss=0.0018288587 aux={'kl_divergence': 0.04638671875, 'loss': 0.0018288587452843785, 'mean_advantage': 0.0, 'mean_reward': 0.8515625, 'std_advantage': 0.3061429262161255} |
| 2026-05-21T09:24:02.841561292Z | grpo_trainer process   0 step      180] loss=0.0016368012 aux={'kl_divergence': 0.04150390625, 'loss': 0.001636801171116531, 'mean_advantage': 0.0, 'mean_reward': 0.86328125, 'std_advantage': 0.32471364736557007} |
| 2026-05-21T09:56:26.578846432Z | grpo_trainer process   0 step      190] loss=0.00083456 aux={'kl_divergence': 0.021240234375, 'loss': 0.0008345600217580795, 'mean_advantage': 0.0, 'mean_reward': 0.8125, 'std_advantage': 0.35350343585014343} |
| 2026-05-21T10:28:49.752072669Z | grpo_trainer process   0 step      200] loss=0.0008566636 aux={'kl_divergence': 0.021484375, 'loss': 0.0008566636242903769, 'mean_advantage': 0.0, 'mean_reward': 0.859375, 'std_advantage': 0.3061429262161255} |
| 2026-05-21T11:00:41.725593516Z | grpo_trainer process   0 step      210] loss=0.0011084096 aux={'kl_divergence': 0.027587890625, 'loss': 0.0011084096040576696, 'mean_advantage': 0.0, 'mean_reward': 0.7890625, 'std_advantage': 0.33067217469215393} |
| 2026-05-21T11:33:03.217026791Z | grpo_trainer process   0 step      220] loss=0.0014873646 aux={'kl_divergence': 0.037109375, 'loss': 0.001487364643253386, 'mean_advantage': 0.0, 'mean_reward': 0.83984375, 'std_advantage': 0.31245583295822144} |
| 2026-05-21T12:05:03.160119936Z | grpo_trainer process   0 step      230] loss=0.0015340311 aux={'kl_divergence': 0.038818359375, 'loss': 0.0015340311219915748, 'mean_advantage': 0.0, 'mean_reward': 0.80078125, 'std_advantage': 0.35898441076278687} |
| 2026-05-21T12:37:30.442265416Z | grpo_trainer process   0 step      240] loss=0.0011927191 aux={'kl_divergence': 0.0294189453125, 'loss': 0.0011927190935239196, 'mean_advantage': 0.0, 'mean_reward': 0.89453125, 'std_advantage': 0.2723926603794098} |
| 2026-05-21T13:09:53.947645781Z | grpo_trainer process   0 step      250] loss=0.0010791621 aux={'kl_divergence': 0.026611328125, 'loss': 0.0010791621170938015, 'mean_advantage': 0.0, 'mean_reward': 0.83984375, 'std_advantage': 0.2576576769351959} |
| 2026-05-21T13:42:08.514955041Z | grpo_trainer process   0 step      260] loss=0.00078981195 aux={'kl_divergence': 0.019287109375, 'loss': 0.0007898119511082768, 'mean_advantage': 0.0, 'mean_reward': 0.80859375, 'std_advantage': 0.28637051582336426} |
| 2026-05-21T14:14:29.498439384Z | grpo_trainer process   0 step      270] loss=0.0013654947 aux={'kl_divergence': 0.031494140625, 'loss': 0.0013654946815222502, 'mean_advantage': 0.0, 'mean_reward': 0.84375, 'std_advantage': 0.3186436593532562} |
| 2026-05-21T14:46:50.348814242Z | grpo_trainer process   0 step      280] loss=0.0010428817 aux={'kl_divergence': 0.0255126953125, 'loss': 0.0010428817477077246, 'mean_advantage': 0.0, 'mean_reward': 0.8515625, 'std_advantage': 0.3061429262161255} |
| 2026-05-21T15:19:07.077788536Z | grpo_trainer process   0 step      290] loss=0.0018607009 aux={'kl_divergence': 0.0458984375, 'loss': 0.0018607008969411254, 'mean_advantage': 0.0, 'mean_reward': 0.88671875, 'std_advantage': 0.2576576769351959} |
| 2026-05-21T15:51:30.866416636Z | grpo_trainer process   0 step      300] loss=0.0011701463 aux={'kl_divergence': 0.02978515625, 'loss': 0.0011701462790369987, 'mean_advantage': 0.0, 'mean_reward': 0.8203125, 'std_advantage': 0.3186436891555786} |
| 2026-05-21T16:56:48.579458039Z | grpo_trainer process   0 step      310] loss=0.0018503338 aux={'kl_divergence': 0.04638671875, 'loss': 0.0018503337632864714, 'mean_advantage': 0.0, 'mean_reward': 0.82421875, 'std_advantage': 0.28637051582336426} |
| 2026-05-21T17:50:57.495902099Z | grpo_trainer process   0 step      320] loss=0.0020006076 aux={'kl_divergence': 0.05029296875, 'loss': 0.0020006075501441956, 'mean_advantage': 0.0, 'mean_reward': 0.734375, 'std_advantage': 0.29310956597328186} |
| 2026-05-21T18:22:49.187300548Z | grpo_trainer process   0 step      330] loss=0.0018963539 aux={'kl_divergence': 0.047607421875, 'loss': 0.001896353904157877, 'mean_advantage': 0.0, 'mean_reward': 0.80078125, 'std_advantage': 0.33652520179748535} |
| 2026-05-21T18:54:24.922694855Z | grpo_trainer process   0 step      340] loss=0.0054456657 aux={'kl_divergence': 0.1357421875, 'loss': 0.005445665679872036, 'mean_advantage': 0.0, 'mean_reward': 0.79296875, 'std_advantage': 0.3247136175632477} |
| 2026-05-21T19:25:55.882906259Z | grpo_trainer process   0 step      350] loss=0.007454818 aux={'kl_divergence': 0.1865234375, 'loss': 0.0074548181146383286, 'mean_advantage': 0.0, 'mean_reward': 0.8203125, 'std_advantage': 0.33067217469215393} |
| 2026-05-21T19:57:51.066915109Z | grpo_trainer process   0 step      360] loss=0.0056595854 aux={'kl_divergence': 0.1416015625, 'loss': 0.005659585352987051, 'mean_advantage': 0.0, 'mean_reward': 0.84375, 'std_advantage': 0.27946901321411133} |
| 2026-05-21T20:30:01.532842459Z | grpo_trainer process   0 step      370] loss=0.005690878 aux={'kl_divergence': 0.14453125, 'loss': 0.005690877791494131, 'mean_advantage': 0.0, 'mean_reward': 0.82421875, 'std_advantage': 0.3124558627605438} |
| 2026-05-21T21:01:54.699033834Z | grpo_trainer process   0 step      380] loss=0.004170566 aux={'kl_divergence': 0.1826171875, 'loss': 0.004170565865933895, 'mean_advantage': 0.0, 'mean_reward': 0.82421875, 'std_advantage': 0.33652520179748535} |
| 2026-05-21T21:33:41.274742246Z | grpo_trainer process   0 step      390] loss=0.0031784268 aux={'kl_divergence': 0.07958984375, 'loss': 0.0031784267630428076, 'mean_advantage': 0.0, 'mean_reward': 0.83984375, 'std_advantage': 0.3247135877609253} |
| 2026-05-21T22:05:42.243565384Z | grpo_trainer process   0 step      400] loss=0.0021257214 aux={'kl_divergence': 0.051513671875, 'loss': 0.002125721424818039, 'mean_advantage': 0.0, 'mean_reward': 0.765625, 'std_advantage': 0.3749469816684723} |
| 2026-05-21T22:37:48.809997051Z | grpo_trainer process   0 step      410] loss=0.0030547904 aux={'kl_divergence': 0.076171875, 'loss': 0.0030547904316335917, 'mean_advantage': 0.0, 'mean_reward': 0.8125, 'std_advantage': 0.33067217469215393} |
| 2026-05-21T23:09:55.881353340Z | grpo_trainer process   0 step      420] loss=0.0028431346 aux={'kl_divergence': 0.07275390625, 'loss': 0.002843134570866823, 'mean_advantage': 0.0, 'mean_reward': 0.828125, 'std_advantage': 0.27946901321411133} |
| 2026-05-21T23:41:53.512144675Z | grpo_trainer process   0 step      430] loss=0.003304979 aux={'kl_divergence': 0.08154296875, 'loss': 0.0033049790654331446, 'mean_advantage': 0.0, 'mean_reward': 0.7890625, 'std_advantage': 0.33067217469215393} |
| 2026-05-22T00:13:55.887190536Z | grpo_trainer process   0 step      440] loss=0.0039181598 aux={'kl_divergence': 0.09765625, 'loss': 0.003918159753084183, 'mean_advantage': 0.0, 'mean_reward': 0.83203125, 'std_advantage': 0.2723926901817322} |
| 2026-05-22T00:45:44.122941486Z | grpo_trainer process   0 step      450] loss=0.0046657296 aux={'kl_divergence': 0.1162109375, 'loss': 0.0046657295897603035, 'mean_advantage': 0.0, 'mean_reward': 0.8359375, 'std_advantage': 0.3186436593532562} |
| 2026-05-22T01:17:53.695116630Z | grpo_trainer process   0 step      460] loss=0.005848777 aux={'kl_divergence': 0.1455078125, 'loss': 0.005848777014762163, 'mean_advantage': 0.0, 'mean_reward': 0.83984375, 'std_advantage': 0.24202725291252136} |
| 2026-05-22T01:50:10.674041534Z | grpo_trainer process   0 step      470] loss=0.0055388883 aux={'kl_divergence': 0.138671875, 'loss': 0.0055388882756233215, 'mean_advantage': 0.0, 'mean_reward': 0.82421875, 'std_advantage': 0.2723926901817322} |
| 2026-05-22T02:22:26.004694592Z | grpo_trainer process   0 step      480] loss=0.005008635 aux={'kl_divergence': 0.1259765625, 'loss': 0.005008635111153126, 'mean_advantage': 0.0, 'mean_reward': 0.83203125, 'std_advantage': 0.28637048602104187} |
| 2026-05-22T02:54:50.519501962Z | grpo_trainer process   0 step      490] loss=0.0058812983 aux={'kl_divergence': 0.1494140625, 'loss': 0.005881298333406448, 'mean_advantage': 0.0, 'mean_reward': 0.84375, 'std_advantage': 0.3061429262161255} |
| 2026-05-22T03:27:06.446059476Z | grpo_trainer process   0 step      500] loss=0.0019979917 aux={'kl_divergence': 0.05029296875, 'loss': 0.0019979916978627443, 'mean_advantage': 0.0, 'mean_reward': 0.83203125, 'std_advantage': 0.31245583295822144} |
| 2026-05-22T03:59:24.588697566Z | grpo_trainer process   0 step      510] loss=0.0009672041 aux={'kl_divergence': 0.02392578125, 'loss': 0.0009672041051089764, 'mean_advantage': 0.0, 'mean_reward': 0.82421875, 'std_advantage': 0.33652523159980774} |
| 2026-05-22T04:31:20.127201852Z | grpo_trainer process   0 step      520] loss=0.0011656838 aux={'kl_divergence': 0.0303955078125, 'loss': 0.0011656838469207287, 'mean_advantage': 0.0, 'mean_reward': 0.83203125, 'std_advantage': 0.2723926901817322} |
| 2026-05-22T05:03:12.649262331Z | grpo_trainer process   0 step      530] loss=0.001120667 aux={'kl_divergence': 0.0281982421875, 'loss': 0.0011206669732928276, 'mean_advantage': 0.0, 'mean_reward': 0.84765625, 'std_advantage': 0.31245583295822144} |
| 2026-05-22T05:35:17.823372914Z | grpo_trainer process   0 step      540] loss=0.0021542164 aux={'kl_divergence': 0.0546875, 'loss': 0.0021542164031416178, 'mean_advantage': 0.0, 'mean_reward': 0.8515625, 'std_advantage': 0.27946901321411133} |
| 2026-05-22T06:07:31.346706271Z | grpo_trainer process   0 step      550] loss=0.005200909 aux={'kl_divergence': 0.1298828125, 'loss': 0.005200908984988928, 'mean_advantage': 0.0, 'mean_reward': 0.890625, 'std_advantage': 0.29310956597328186} |
| 2026-05-22T06:39:36.244942006Z | grpo_trainer process   0 step      560] loss=0.008610529 aux={'kl_divergence': 0.21484375, 'loss': 0.008610528893768787, 'mean_advantage': 0.0, 'mean_reward': 0.79296875, 'std_advantage': 0.31245583295822144} |
| 2026-05-22T07:11:33.924264235Z | grpo_trainer process   0 step      570] loss=0.003486838 aux={'kl_divergence': 0.0908203125, 'loss': 0.0034868379589170218, 'mean_advantage': 0.0, 'mean_reward': 0.8671875, 'std_advantage': 0.3061429560184479} |
| 2026-05-22T07:43:41.869713054Z | grpo_trainer process   0 step      580] loss=0.0017850895 aux={'kl_divergence': 0.11376953125, 'loss': 0.0017850894946604967, 'mean_advantage': 0.0, 'mean_reward': 0.82421875, 'std_advantage': 0.3247136175632477} |
| 2026-05-22T08:15:38.562563168Z | grpo_trainer process   0 step      590] loss=-0.010442441 aux={'kl_divergence': 0.5, 'loss': -0.010442441329360008, 'mean_advantage': 0.0, 'mean_reward': 0.7734375, 'std_advantage': 0.33067217469215393} |
| 2026-05-22T08:47:50.362776123Z | grpo_trainer process   0 step      600] loss=0.034464348 aux={'kl_divergence': 1.7890625, 'loss': 0.03446434810757637, 'mean_advantage': 0.0, 'mean_reward': 0.78515625, 'std_advantage': 0.34793609380722046} |
| 2026-05-22T09:19:51.979916643Z | grpo_trainer process   0 step      610] loss=-0.012390678 aux={'kl_divergence': 0.6015625, 'loss': -0.012390677817165852, 'mean_advantage': 0.0, 'mean_reward': 0.76953125, 'std_advantage': 0.3124558627605438} |
| 2026-05-22T10:34:23.464028301Z | grpo_trainer process   0 step      620] loss=0.081923306 aux={'kl_divergence': 3.5, 'loss': 0.08192330598831177, 'mean_advantage': 0.0, 'mean_reward': 0.7890625, 'std_advantage': 0.3643829822540283} |
| 2026-05-22T11:07:01.636018161Z | grpo_trainer process   0 step      630] loss=0.09963063 aux={'kl_divergence': 4.40625, 'loss': 0.09963063150644302, 'mean_advantage': 0.0, 'mean_reward': 0.73046875, 'std_advantage': 0.40978196263313293} |
| 2026-05-22T11:39:36.538371633Z | grpo_trainer process   0 step      640] loss=0.18580464 aux={'kl_divergence': 5.53125, 'loss': 0.18580463528633118, 'mean_advantage': 0.0, 'mean_reward': 0.85546875, 'std_advantage': 0.2996971011161804} |
| 2026-05-22T12:12:03.783374015Z | grpo_trainer process   0 step      650] loss=0.020792492 aux={'kl_divergence': 2.71875, 'loss': 0.020792491734027863, 'mean_advantage': 0.0, 'mean_reward': 0.6875, 'std_advantage': 0.4145195186138153} |
| 2026-05-22T12:45:01.018651464Z | grpo_trainer process   0 step      660] loss=0.07614714 aux={'kl_divergence': 5.375, 'loss': 0.07614713907241821, 'mean_advantage': 0.0, 'mean_reward': 0.6640625, 'std_advantage': 0.459214448928833} |
| 2026-05-22T13:17:35.197225585Z | grpo_trainer process   0 step      670] loss=-0.07165395 aux={'kl_divergence': 0.1953125, 'loss': -0.07165394723415375, 'mean_advantage': 0.0, 'mean_reward': 0.76953125, 'std_advantage': 0.36970269680023193} |
| 2026-05-22T13:50:08.690478521Z | grpo_trainer process   0 step      680] loss=-0.050928295 aux={'kl_divergence': 0.291015625, 'loss': -0.05092829465866089, 'mean_advantage': 0.0, 'mean_reward': 0.77734375, 'std_advantage': 0.34793609380722046} |
| 2026-05-22T14:22:37.831674367Z | grpo_trainer process   0 step      690] loss=-0.090814 aux={'kl_divergence': 0.28125, 'loss': -0.0908140018582344, 'mean_advantage': 0.0, 'mean_reward': 0.671875, 'std_advantage': 0.4145195186138153} |
| 2026-05-22T14:55:14.076143674Z | grpo_trainer process   0 step      700] loss=-0.06022806 aux={'kl_divergence': 0.1484375, 'loss': -0.06022806093096733, 'mean_advantage': 0.0, 'mean_reward': 0.72265625, 'std_advantage': 0.39025720953941345} |
| 2026-05-22T15:27:50.283085825Z | grpo_trainer process   0 step      710] loss=-0.06994404 aux={'kl_divergence': 0.26171875, 'loss': -0.06994403898715973, 'mean_advantage': 0.0, 'mean_reward': 0.70703125, 'std_advantage': 0.4001386761665344} |
| 2026-05-22T16:00:01.060380575Z | grpo_trainer process   0 step      720] loss=-0.09920842 aux={'kl_divergence': 0.431640625, 'loss': -0.09920842200517654, 'mean_advantage': 0.0, 'mean_reward': 0.5859375, 'std_advantage': 0.4238356947898865} |
| 2026-05-22T16:32:26.879066228Z | grpo_trainer process   0 step      730] loss=-0.10878732 aux={'kl_divergence': 0.44921875, 'loss': -0.10878732055425644, 'mean_advantage': 0.0, 'mean_reward': 0.6875, 'std_advantage': 0.4049890339374542} |
| 2026-05-22T17:04:38.653022444Z | grpo_trainer process   0 step      740] loss=-0.075865075 aux={'kl_divergence': 0.2001953125, 'loss': -0.07586507499217987, 'mean_advantage': 0.0, 'mean_reward': 0.7734375, 'std_advantage': 0.3852214515209198} |
| 2026-05-22T17:37:03.042670263Z | grpo_trainer process   0 step      750] loss=-0.06815426 aux={'kl_divergence': 0.33203125, 'loss': -0.06815426051616669, 'mean_advantage': 0.0, 'mean_reward': 0.72265625, 'std_advantage': 0.4192034602165222} |
| 2026-05-22T17:53:16.399139616Z | grpo_trainer process   0 step      750] loss=-0.069291085 aux={'kl_divergence': 0.50390625, 'loss': -0.06929108500480652, 'mean_advantage': 0.0, 'mean_reward': 0.73046875, 'std_advantage': 0.4001387059688568} |
| 2026-05-22T18:25:22.927145354Z | grpo_trainer process   0 step      760] loss=-0.0668161 aux={'kl_divergence': 0.279296875, 'loss': -0.06681609898805618, 'mean_advantage': 0.0, 'mean_reward': 0.81640625, 'std_advantage': 0.36970269680023193} |
| 2026-05-22T18:57:32.424465497Z | grpo_trainer process   0 step      770] loss=-0.0634951 aux={'kl_divergence': 0.13671875, 'loss': -0.06349509954452515, 'mean_advantage': 0.0, 'mean_reward': 0.7421875, 'std_advantage': 0.3643829822540283} |
| 2026-05-22T19:30:23.105309243Z | grpo_trainer process   0 step      780] loss=-0.14850381 aux={'kl_divergence': 0.439453125, 'loss': -0.14850381016731262, 'mean_advantage': 0.0, 'mean_reward': 0.5234375, 'std_advantage': 0.45063015818595886} |
| 2026-05-22T20:02:28.565239559Z | grpo_trainer process   0 step      790] loss=-0.124386005 aux={'kl_divergence': 0.208984375, 'loss': -0.1243860051035881, 'mean_advantage': 0.0, 'mean_reward': 0.7265625, 'std_advantage': 0.4418792724609375} |
| 2026-05-22T21:50:22.116266570Z | grpo_trainer process   0 step      800] loss=-0.13364542 aux={'kl_divergence': 0.2578125, 'loss': -0.1336454153060913, 'mean_advantage': 0.0, 'mean_reward': 0.6640625, 'std_advantage': 0.45063021779060364} |
| 2026-05-22T22:56:24.496360051Z | grpo_trainer process   0 step      810] loss=-0.14158413 aux={'kl_divergence': 0.38671875, 'loss': -0.1415841281414032, 'mean_advantage': 0.0, 'mean_reward': 0.6015625, 'std_advantage': 0.4329514801502228} |
| 2026-05-22T23:28:32.510353636Z | grpo_trainer process   0 step      820] loss=-0.066071026 aux={'kl_divergence': 0.1533203125, 'loss': -0.0660710260272026, 'mean_advantage': 0.0, 'mean_reward': 0.7109375, 'std_advantage': 0.3852214515209198} |
| 2026-05-23T00:00:49.963098298Z | grpo_trainer process   0 step      830] loss=-0.077516325 aux={'kl_divergence': 0.177734375, 'loss': -0.07751632481813431, 'mean_advantage': 0.0, 'mean_reward': 0.74609375, 'std_advantage': 0.36970269680023193} |
| 2026-05-23T00:33:06.136014391Z | grpo_trainer process   0 step      840] loss=-0.073309146 aux={'kl_divergence': 0.2333984375, 'loss': -0.07330914586782455, 'mean_advantage': 0.0, 'mean_reward': 0.6484375, 'std_advantage': 0.3749469816684723} |
| 2026-05-23T01:05:44.310713186Z | grpo_trainer process   0 step      850] loss=-0.12693672 aux={'kl_divergence': 0.41015625, 'loss': -0.12693671882152557, 'mean_advantage': 0.0, 'mean_reward': 0.57421875, 'std_advantage': 0.428417831659317} |
| 2026-05-23T01:38:14.544007316Z | grpo_trainer process   0 step      860] loss=-0.06829094 aux={'kl_divergence': 0.1904296875, 'loss': -0.06829094141721725, 'mean_advantage': 0.0, 'mean_reward': 0.7578125, 'std_advantage': 0.3422781825065613} |
| 2026-05-23T02:10:53.600397443Z | grpo_trainer process   0 step      870] loss=-0.09320154 aux={'kl_divergence': 0.251953125, 'loss': -0.09320154041051865, 'mean_advantage': 0.0, 'mean_reward': 0.734375, 'std_advantage': 0.4049890339374542} |
| 2026-05-23T02:43:09.556622928Z | grpo_trainer process   0 step      880] loss=-0.12533228 aux={'kl_divergence': 0.2353515625, 'loss': -0.1253322809934616, 'mean_advantage': 0.0, 'mean_reward': 0.66796875, 'std_advantage': 0.4549425542354584} |
| 2026-05-23T03:16:07.432820635Z | grpo_trainer process   0 step      890] loss=-0.094314456 aux={'kl_divergence': 0.265625, 'loss': -0.09431445598602295, 'mean_advantage': 0.0, 'mean_reward': 0.69921875, 'std_advantage': 0.4192034900188446} |
| 2026-05-23T03:48:31.271703809Z | grpo_trainer process   0 step      900] loss=-0.07618472 aux={'kl_divergence': 0.27734375, 'loss': -0.0761847198009491, 'mean_advantage': 0.0, 'mean_reward': 0.7265625, 'std_advantage': 0.3952288329601288} |
| 2026-05-23T04:20:56.618090038Z | grpo_trainer process   0 step      910] loss=-0.08086206 aux={'kl_divergence': 0.35546875, 'loss': -0.08086206018924713, 'mean_advantage': 0.0, 'mean_reward': 0.7109375, 'std_advantage': 0.4145195186138153} |
| 2026-05-23T04:53:11.031356898Z | grpo_trainer process   0 step      920] loss=-0.09914622 aux={'kl_divergence': 0.291015625, 'loss': -0.09914621710777283, 'mean_advantage': 0.0, 'mean_reward': 0.6875, 'std_advantage': 0.4238356947898865} |
| 2026-05-23T05:25:25.436153684Z | grpo_trainer process   0 step      930] loss=-0.098998345 aux={'kl_divergence': 0.1494140625, 'loss': -0.09899834543466568, 'mean_advantage': 0.0, 'mean_reward': 0.7265625, 'std_advantage': 0.4049890339374542} |
| 2026-05-23T05:57:59.276675037Z | grpo_trainer process   0 step      940] loss=-0.08310325 aux={'kl_divergence': 0.1826171875, 'loss': -0.083103246986866, 'mean_advantage': 0.0, 'mean_reward': 0.72265625, 'std_advantage': 0.36970269680023193} |
| 2026-05-23T06:30:29.761157065Z | grpo_trainer process   0 step      950] loss=-0.050403107 aux={'kl_divergence': 0.23828125, 'loss': -0.050403106957674026, 'mean_advantage': 0.0, 'mean_reward': 0.78515625, 'std_advantage': 0.35898441076278687} |
| 2026-05-23T07:02:57.012105371Z | grpo_trainer process   0 step      960] loss=-0.05283285 aux={'kl_divergence': 0.126953125, 'loss': -0.05283284932374954, 'mean_advantage': 0.0, 'mean_reward': 0.828125, 'std_advantage': 0.3422781825065613} |
| 2026-05-23T07:35:29.761920602Z | grpo_trainer process   0 step      970] loss=-0.13346703 aux={'kl_divergence': 0.388671875, 'loss': -0.13346703350543976, 'mean_advantage': 0.0, 'mean_reward': 0.5234375, 'std_advantage': 0.4145195186138153} |
| 2026-05-23T08:08:12.593694253Z | grpo_trainer process   0 step      980] loss=-0.09901131 aux={'kl_divergence': 0.1650390625, 'loss': -0.09901130944490433, 'mean_advantage': 0.0, 'mean_reward': 0.75390625, 'std_advantage': 0.4097820222377777} |
| 2026-05-23T08:40:51.088112437Z | grpo_trainer process   0 step      990] loss=-0.08227306 aux={'kl_divergence': 0.185546875, 'loss': -0.08227305859327316, 'mean_advantage': 0.0, 'mean_reward': 0.7421875, 'std_advantage': 0.3852214515209198} |
| 2026-05-23T09:13:19.974985949Z | grpo_trainer process   0 step     1000] loss=-0.0815361 aux={'kl_divergence': 0.2294921875, 'loss': -0.08153609931468964, 'mean_advantage': 0.0, 'mean_reward': 0.7109375, 'std_advantage': 0.3952288329601288} |
