# Overview
Group Relative Policy Optimization (GRPO) is a reinforcement learning algorithm designed to efficiently train Large Language Models (LLMs) without requiring a Critic model. In standard PPO, a Critic model (often of equal size to the Actor) must be kept in memory to estimate baseline value functions for advantage computation. GRPO eliminates this memory overhead entirely. Instead, for each input prompt, GRPO samples a group of distinct generations (rollouts) from the Actor policy, scores them using a reward model or rule-based function, and computes relative advantages by normalizing the rewards within that group (subtracting the group mean and dividing by the group standard deviation). The policy is then optimized using a PPO-style clipped surrogate objective alongside a KL divergence constraint against a frozen Reference model to prevent policy drift.

# Separate Sampler and Trainer with Pathways
The generation and training workloads are decoupled onto distinct hardware partitions within a global mesh. For example, half of the available accelerators are assigned to a rollout_mesh optimized for low-latency autoregressive decoding (using tensor/data parallelism), while the other half forms a trainer_mesh optimized for FSDP-based gradient updates. Weights are synchronized across meshes after each training step.


# Setup
* Training dataset: [gsm8k  ](https://www.tensorflow.org/datasets/catalog/gsm8k)
* Pretrained model: [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) compatible with fuji-8B. The model is the base for actor, reference, and sampler
* TPU Topology: v6e-16
* GKE Nodepool
  * Pathways: n2-standard-64 (autoscale up to 10)
  * Workers: 4 x ct6e-standard-4t
* Global mesh: (1, 1, 8, 8, 1, 1), data and fsdp parallelism on data and fsdp dimensions
* rollout_mesh: (1, 1, 8, 1, 1, 1), tensor/data parallelism on data dimension
* trainer_mesh: (1, 1, 1, 8, 1, 1), fsdp parallelism on fsdp dimension

# Pre Requisite
The utility scripts rely on Axlearn and follow the [instructions](https://github.com/apple/axlearn/blob/main/docs/01-start.md#installation) to set up your Axlearn env and install the packages for dev. You may need a machine with high memory to run the script.

## Training Dataset gsm8k
To prepare the GSM8K reasoning dataset for AXLearn training pipelines, the raw data must be downloaded and serialized into sharded TFRecord files. The example provides a dedicated conversion utility convert_gsm8k_to_tfrecord.py.

Run the following command to download and serialize the GSM8K dataset into 8 shards:

```shell
python3 -m axlearn.tools.convert_gsm8k_to_tfrecord \
    --output_dir="gs://ericshen-axlearn/tensorflow-datasets" \ #change to your gcs location
    --num_shards=8
```

## Pretrained Model Checkpoint from Huggingface
An automated utility can be used to download pre-trained weights from the Hugging Face Hub, convert them into AXLearn's native JAX parameter structures (Fuji), and serialize them to GCS via TensorStore.

Run the following command to download and convert the LLaMA-3.1 8B Instruct model checkpoint:

```shell
python3 -m axlearn.tools.download_llama3_checkpoint \
    --model_id="meta-llama/Llama-3.1-8B-Instruct" \
    --model_size="8B" \
    --output_dir="gs://ericshen-axlearn/checkpoints/llama-3-1-8B-instruct" # change to your gcs location

```

# Train
To launch the combined trainer/sampler GRPO experiment on Google Cloud TPU instances using AXLearn's launch utility, set up your environment variables and execute `launch_trainer_main`:

```shell
# 1. Set the following envs
export BASTION_TIER=disabled
export NAME=eshen-v6e-rl-grpo-pathways # change to your name
export RUNNER_NAME=gke_tpu_single
export CONFIG=grpo-fuji-8B-v1
export INSTANCE_TYPE=tpu-v6e-16 #change to your instance type
export CLUSTER=ericshen-axlearn #change to your cluster name
export OUTPUT_DIR="gs://cloud-tpu-multipod-dev-axlearn/users/ericshen/${NAME}/$(date +%s)" #change to your gcs location for training output


# 2. Launch the trainer job
(.venv) ➜  axlearn git:(rl-example-pathways) ✗ nohup axlearn gcp launch run \
  --cluster=$CLUSTER \
  --instance_type=$INSTANCE_TYPE \
  --name=$NAME \
  --num_replicas=1 \
  --runner_name=$RUNNER_NAME \
  --bundler_spec=allow_dirty=True \
  --bundler_type=artifactregistry \
  --bundler_spec=image=tpu \
  --bundler_spec=dockerfile=Dockerfile \
  --bundler_spec=target=tpu --pathways_head_mem=128 --pathways_head_cpu=8 --env=GRPO_REWARD_TYPE:gsm8k \ #set the reward_type to gsm8k. otherwise it'd be dummy reward.
  -- \
  python3 -m axlearn.common.launch_trainer_main \
  --module=text.gpt.grpo_native_example \
  --config=$CONFIG \
  --trainer_dir=$OUTPUT_DIR \
  --data_dir=gs://ericshen-axlearn/tensorflow-datasets \
  --jax_backend=proxy \
   --trainer_log_every_n_steps=10 > axlearn_launcher.log 2>&1 &
```

# Output
## Explanation of training metrics
During GRPO training, AXLearn logs several crucial telemetry metrics to track policy optimization, reward improvements, and stability:
* `loss`: The total GRPO surrogate policy objective loss, which combines the clipped importance-sampling policy loss and the KL divergence penalty.
* `mean_reward`: The average raw math accuracy achieved by the generated rollouts in the current batch (e.g., 1.0 for correct boxed/extracted final numbers matching the ground truth, 0.0 otherwise). A rising mean_reward indicates successful mathematical reasoning alignment.
* `mean_advantage`: The average group-relative advantage value across the batch. While normalized advantages average to zero within each prompt group, tracking batch-wide stability ensures correct reward distribution.
* `std_advantage`: The standard deviation of the computed advantages across rollouts, reflecting the variance and diversity of rewards within the generated groups.
* `kl_divergence`: The token-level Kullback-Leibler (KL) divergence between the active Actor policy and the frozen Reference policy. This tracks how far the model's generation distribution has shifted from the initial pre-trained foundation checkpoint.

`Sample output`: [grpo_pathways_sample_output](./grpo_pythways_sample_output.md)

## Monitoring & Checking
### Kubectl logs

```shell
# tail the log of the pod
k logs -f eshen-v6e-rl-grpo-job-0-1-dgsvd


# only grep the metrics line like the following
# I0513 14:17:45.599639 136435580311488 trainer.py:409] grpo_trainer process   2 step     1000] loss=0.05171673 aux={'kl_divergence': 1.2890625, 'loss': 0.05171672999858856, 'mean_advantage': 0.0, 'mean_reward': 0.8203125, 'std_advantage': 0.3061429262161255}

k logs -f eshen-v6e-rl-grpo-job-0-1-dgsvd | grep kl_div
```
### Log Explorer

``` shell
– Set the time range during which the job has run
resource.type="k8s_container"
resource.labels.cluster_name="ericshen-axlearn" -- change to your cluster
(("kl_divergence" AND "grpo_trainer process   0") OR "Average step time:")
```
### GCS Output Location
```shell
# get model analysis
gcloud storage cat gs://cloud-tpu-multipod-dev-axlearn/users/ericshen/eshen-v6e-rl-grpo/1778611794/model_analysis.txt


# get trainer config
gcloud storage cat gs://cloud-tpu-multipod-dev-axlearn/users/ericshen/eshen-v6e-rl-grpo/1778611794/trainer_config
```
