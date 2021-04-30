$HOME/ts4uc/ts4uc/agents/a3c/train.py \
    --save_dir foo \
    --num_gen 5 \
    --num_epochs 10000 \
    --buffer_size 2000 \
    --ac_learning_rate 0.0001 \
    --cr_learning_rate 0.001 \
    --workers 4 \
    --entropy 0.05
