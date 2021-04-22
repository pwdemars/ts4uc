#!/usr/bin/env python 

import os
import sys
import numpy as np

NUM_SCENARIOS = 100

data_dir = sys.argv[1]
input_fn = sys.argv[2]

with open(input_fn, 'w') as f:
    f.close()

all_fns = [f for f in os.listdir(data_dir) if '.csv' in f]
all_fns_reps = np.repeat(all_fns, NUM_SCENARIOS)
init_seed = 123
# seeds = np.tile(np.arange(NUM_SCENARIOS), len(all_fns)) + init_seed
np.random.seed(999) # seed the seeds! for reproducibility
seeds = np.random.randint(1e6, size=(NUM_SCENARIOS*len(all_fns)))

for i, (fn, seed) in enumerate(zip(all_fns_reps, seeds)):
    prof_name = fn.split('.')[0]
    line = str(i).zfill(4) + ' ' + os.path.join(data_dir, fn) + ' ' + str(seed) + ' ' + str(prof_name)
    with open(input_fn, 'a') as f:
        f.write(line)
        f.write('\n')
        f.close()

print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")
print("ENSURE SUBMISSION SCRIPT IS CORRECT!")
print("Number of jobs should be {}".format(len(all_fns_reps)))
print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")
