CUDA_VISIBLE_DEVICES=0 vllm serve /mnt/workspace/models/Qwen3-8B/     --host 0.0.0.0     --port 8080     --max-model-len 10240     --tensor-parallel-size 1     --disable-log-requests    --enable-prefix-caching



source /mnt/workspace/miniconda3/etc/profile.d/conda.sh
conda init bash
conda activate py310