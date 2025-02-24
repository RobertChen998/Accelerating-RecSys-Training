# python dlrm_baseline_cpu_gpu2.py --arch-sparse-feature-size=4 \
# 								--arch-mlp-bot="13-8-8-8-4" \
# 								--arch-mlp-top="512-8-1" \
# 								--data-generation=random \
# 								--data-set=kaggle \
# 								--raw-data-file=./input/kaggle/train.txt \
# 								--processed-data-file=./input/kaggle/kaggleAdDisplayChallenge_processed.npz \
# 								--loss-function=bce \
# 								--round-targets=True \
# 								--mini-batch-size=4096 \
# 								--print-freq=4096 \
# 								--print-time \
# 								--nepochs=1 

# python dlrm_baseline_cpu_gpu2.py --arch-sparse-feature-size=4 \
# 								--arch-mlp-bot="13-8-8-8-4" \
# 								--arch-mlp-top="512-8-1" \
# 								--data-generation=random \
# 								--data-set=kaggle \
# 								--raw-data-file=./input/kaggle/train.txt \
# 								--processed-data-file=./input/kaggle/kaggleAdDisplayChallenge_processed.npz \
# 								--loss-function=bce \
# 								--round-targets=True \
# 								--mini-batch-size=3 \
# 								--print-freq=4096 \
# 								--print-time \
# 								--nepochs=1 \
# 								--data-size=6

python dlrm_baseline_cpu_gpu_whole_store.py --mini-batch-size=2 \
								--data-size=4 \
								--arch-embedding-size=6-6