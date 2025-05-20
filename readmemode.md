apt-get update
apt-get install tmux

source /mnt/workspace/miniconda3/etc/profile.d/conda.sh
conda init bash
conda activate py310

tmux new -s vllm
conda activate py310
CUDA_VISIBLE_DEVICES=0 vllm serve /mnt/workspace/models/Qwen3-8B/     --host 0.0.0.0     --port 8080     --max-model-len 10240     --tensor-parallel-size 1     --disable-log-requests    --enable-prefix-caching

tmux new -s run
conda activate py310
python -u /mnt/workspace/src/start_async_retrivel_embedding.py






你每次开发只需要操作 本地 master：

# 在本地 master 分支上
git add .
git commit -m "新的开发内容"
git push origin master
确认无误后，执行一次手动合并流程：



git checkout main
git pull origin main
git merge origin/master --allow-unrelated-histories
git push origin main