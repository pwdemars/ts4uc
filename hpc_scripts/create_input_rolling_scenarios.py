#!/usr/bin/env python 

import os
import sys
import numpy as np

NUM_SCENARIOS = 1000

data_dir = sys.argv[1]
input_fn = sys.argv[2]

with open(input_fn, 'w') as f:
    f.close()

all_fns = [f for f in os.listdir(data_dir) if '.csv' in f]

i = 0
for fn in all_fns:
    for n in range(NUM_SCENARIOS):
        prof_name = fn.split('.')[0]
        line = str(i).zfill(5) + ' ' + os.path.join(data_dir, fn) + ' ' + str(n) + ' ' + str(prof_name)
        with open(input_fn, 'a') as f:
            f.write(line)
            f.write('\n')
            f.close()
        i += 1


print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")
print("ENSURE SUBMISSION SCRIPT IS CORRECT!")
print("Number of jobs should be {}".format(NUM_SCENARIOS*len(all_fns)))
print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")
