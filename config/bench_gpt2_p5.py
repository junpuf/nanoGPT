# launch as the following:
# $ torchrun --standalone --nproc_per_node=8 train.py config/bench_gpt2_p5.py

wandb_log = True
wandb_project = 'owt'
wandb_run_name='gpt2-124M'

# these make the total batch size be ~1.0M
# 24 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 983,040
batch_size = 24
block_size = 1024
gradient_accumulation_steps = 5 * 8

max_iters = 250
lr_decay_iters = 250

# eval stuff
eval_interval = 250
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
