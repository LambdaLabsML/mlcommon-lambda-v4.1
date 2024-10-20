## Lambda Cloud 1-Click Cluster

Our benchmark is conducted with a 64xGPU Lambda Cloud 1-Click Cluster. It has three head nodes and eight compute node. Each compute node has eight NVIDIA H100 80GB SXM5 GPUs, 208 CPU Cores, 1.9 TB RAM, and 24.2 TB SSD. Each node also has eight NVIDIA Quantum-2 400Gb/s InfiniBand non-blocking fabric in a rail-optimized topology, enabling in total 3.2Tb/s inter-node communication bandwdith. A persistent storage is attached to the cluster and shared across all head nodes and compute nodes. 

On the software side, the benchmark uses customized images based on NVIDIA’s NGC container. Lambda’s 1-Click Cluster also comes with all the main software dependencies pre-installed, including NVIDIA drivers, SLURM, Pyxis and Enroot.

## Results

Our benchmark covers the task of training BERT, Llama2, and Stable Diffusion with 8x, 16x, 32x, 64x GPUs. In all cases, we see ~6x speedup of raw training time from 8x to 64x GPUs. This translates to ~75% scaling efficiency for time to solution. As a side node, the reason for such a sub-linear speedup is expected, due to the [complexity of adapting hyper-parameters](https://arxiv.org/abs/1706.02677) for a larger cluster, and in general the diminishing return of training with large batch sizes.

Below are the results for each model. For each tasks we provide both the mean and two-sigmas for raw training time and throughput, as function of number of GPUs. 

### BERT
<p align="center">
  <img src="./Lambda/imgs/bert/raw_train_time.png" alt="bert-raw-train-time" width="45%" />
  <img src="./Lambda/imgs/bert/training_sequences_per_second.png" alt="bert-training-sequence-per-second" width="45%" />
</p>

### Llama2 70B LoRA Fine-Tuning
<p align="center">
  <img src="./Lambda/imgs/llama2_70b_lora/raw_train_time.png" alt="llama2_70b_lora-raw-train-time" width="45%" />
  <img src="./Lambda/imgs/llama2_70b_lora/training_sequences_per_second.png" alt="llama2_70b_lora-training-sequence-per-second" width="45%" />
</p>

### Stable Diffusion
<p align="center">
  <img src="./Lambda/imgs/stable_diffusion/raw_train_time.png" alt="stable_diffusion-raw-train-time" width="45%" />
  <img src="./Lambda/imgs/stable_diffusion/training_sequences_per_second.png" alt="stable_diffusion-training-sequence-per-second" width="45%" />
</p>


## Reproduction

An important goal of our study is to make the benchmark easy to reproduce. Here we list the major step for reproduction. You can find the detailed instructions in this repo ([bert](./Lambda/benchmarks/bert/implementations/pytorch/Lambda.md), [llama2_70b_lora](./Lambda/benchmarks/llama2_70b_lora/implementations/nemo/Lambda.md), [stable_diffusion](./Lambda/benchmarks/stable_diffusion/implementations/nemo/Lambda.md)).  

### Setup a docker registry on the head node
We launch the `registry:2.8` [docker container](https://github.com/distribution/distribution-library-image/tree/master) on the head node, which serves the benchmark images at port `5000`.

```
ubuntu@head1:~$ docker ps
CONTAINER ID   IMAGE          COMMAND                  CREATED      STATUS       PORTS     NAMES
7ac6b15b8518   registry:2.8   "/entrypoint.sh /etc…"   5 days ago   Up 3 hours             deepops-registry
```

### Build docker image
The first step is to build the docker image for each model. Here is the example of building the docker image for [BERT](./Lambda/benchmarks/bert/implementations/pytorch/Dockerfile) and push it to the local registry: 
```
# Recommend using a worker node for faster built
export HEADNODE_HOSTNAME=ml-64-head-001
docker build --build-arg CACHEBUST=$(date +%s) -t $HEADNODE_HOSTNAME:5000/local/mlperf-nvidia-bert:latest .
docker push $HEADNODE_HOSTNAME:5000/local/mlperf-nvidia-bert:latest

# Verify if the image has been pushed to the registry
curl http://$HEADNODE_HOSTNAME:5000/v2/_catalog
```

### Prepare data
For each task, we provide a SLURM job for preparing the data. Here is the example of submitting the data preparation job for BERT:

```
# From the head node
export HEADNODE_HOSTNAME=$(hostname)
export DATAPATH=/home/ubuntu/ml-1cc/data/mlperf/bert
sudo mkdir -p $DATAPATH
sudo chmod -R 777 $DATAPATH

sbatch  --export=HEADNODE_HOSTNAME=$HEADNODE_HOSTNAME,DATAPATH=$DATAPATH dataset.sub
```

The above BERT example will take ~48 hours to get all data prepared (~24 hours spent on packaging the dataset). Once it is done, you should see the following folder created:

```
ubuntu@head1:~/ml-1cc/data/mlperf/bert$ ls -la
total 0
drwxrwxrwx 2 root   root   4096 Jul  1 14:14 .
drwxrwxrwx 2 root   ubuntu 4096 Jun 30 15:11 ..
drwxrwxrwx 2 ubuntu ubuntu 4096 Jun 30 15:19 download
drwxrwxrwx 2 ubuntu ubuntu 4096 Jun 17 14:21 hdf5
drwxrwxrwx 2 ubuntu ubuntu 4096 Jul  1 14:04 packed_data
drwxrwxrwx 2 ubuntu ubuntu 4096 Jun 30 15:01 per_seqlen
drwxrwxrwx 2 ubuntu ubuntu 4096 Jun 17 22:44 per_seqlen_parts
drwxrwxrwx 2 ubuntu ubuntu 4096 Jun 30 15:24 phase1
```

### Run benchmark
The benchmark is submitted as a slurm job. For each task, a config file is used for hyper-parameters for each GPU configuration. This is a standard protocol of mlcommon. 

Here is an example of submitting benchmark jobs for the BERT model:

```
# Single node
export HEADNODE_HOSTNAME=$(hostname) && \
source ./config_1cc_1x8x48x1_pack.sh && \
sbatch -N1 --ntasks-per-node=8 --gres=gpu:8 run_1cc.sub

# 2x nodes
export HEADNODE_HOSTNAME=$(hostname) && \
source ./config_1cc_2x8x24x1_pack.sh && \
sbatch -N2 --ntasks-per-node=8 --gres=gpu:8 run_1cc.sub

# 4x nodes
export HEADNODE_HOSTNAME=$(hostname) && \
source ./config_1cc_4x8x12x1_pack.sh && \
sbatch -N4 --ntasks-per-node=8 --gres=gpu:8 run_1cc.sub

# 8x nodes
export HEADNODE_HOSTNAME=$(hostname) && \
source ./config_1cc_8x8x36x1_pack.sh && \
sbatch -N8 --ntasks-per-node=8 --gres=gpu:8 run_1cc.sub
```

You can find the logs of the runs [here](./Lambda/results/).