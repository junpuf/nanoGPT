# config for training GPT-2 (124M) down to a loss of ~3.56 on 1 node of 4X V100 16GB
# launch as the following (e.g. in a screen session)
# $ WANDB_MODE=offline torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2_p3_8xl.py

wandb_log = True
wandb_project = 'owt'
wandb_run_name='gpt2-124M'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
# i need to make total batch size be ~0.1M 
# since p3.8xlarge only have 64G gpu memory which is about 1/5 of p4d.24xlarge
# so maybe 6 batch size * 1024 block size * 4 gradaccum * 4 GPUs = 98,304
batch_size = 6
block_size = 1024
gradient_accumulation_steps = 4 * 4

# this makes total number of tokens be 300B
# try with 5000
max_iters = 5000
lr_decay_iters = 5000

# eval stuff
eval_interval = 250
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
