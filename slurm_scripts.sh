## ============================================================
## slurm/layer1.sh — Feature Extraction (Dask, 1 CPU node)
## ============================================================
#!/bin/bash
#SBATCH --job-name=hpc_layer1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/layer1_%j.out

module load python/3.11 hdf5

source venv/bin/activate

python -m layer1.extract_features \
    --msd-root  /data/msd \
    --out-dir   data/layer1 \
    --n-workers 32

python -m layer1.merge_shards \
    --shard-dir data/layer1/shards \
    --output    data/features.h5


## ============================================================
## slurm/layer2.sh — ALS Training (mpi4py, 1 CPU node)
## ============================================================
#!/bin/bash
#SBATCH --job-name=hpc_layer2
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/layer2_%j.out

module load python/3.11 openmpi hdf5

source venv/bin/activate

# Build sparse matrix (single process)
python -m layer2.build_sparse_matrix \
    --triplets /data/msd/taste_profile_subset.txt \
    --out-dir  data/layer2

# Distributed ALS
mpirun -n 8 python -m layer2.distributed_als \
    --data-dir  data/layer2 \
    --out       data/als_vectors.h5 \
    --n-factors 128 \
    --n-iters   15


## ============================================================
## slurm/layer3.sh — Transformer DDP Training (GPU nodes)
##   Submitted with: --dependency=afterok:$jid1:$jid2
## ============================================================
#!/bin/bash
#SBATCH --job-name=hpc_layer3
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=logs/layer3_%j.out

module load python/3.11 cuda/12.1 hdf5

source venv/bin/activate

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500

torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m layer3.train_ddp \
    --features-h5 data/features.h5 \
    --als-h5      data/als_vectors.h5 \
    --histories   data/layer2/user_histories.pkl \
    --out-dir     data/checkpoints \
    --epochs      10 \
    --batch-size  256
