#!/usr/bin/env python 

import os
import sys

data_dir = sys.argv[1]
input_fn = sys.argv[2]

with open(input_fn, 'w') as f:
    f.close()

all_fns = [f for f in os.listdir(data_dir) if '.csv' in f]

for i, fn in enumerate(all_fns):
    line = str(i).zfill(4) + ' ' + os.path.join(data_dir, fn)
    with open(input_fn, 'a') as f:
        f.write(line)
        f.write('\n')
        f.close()

print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")
print("ENSURE SUBMISSION SCRIPT IS CORRECT!")
print("Number of jobs should be {}".format(len(all_fns)))
print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")
