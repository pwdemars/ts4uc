#!/usr/bin/env python

from rl4uc.environment import interpolate_profile
import pkg_resources
import pandas as pd
import numpy as np
import os

if __name__ == "__main__":

	SEED = 2
	np.random.seed(SEED)


	TEST_DATA_FN = pkg_resources.resource_stream('rl4uc', 'data/test_data_10gen.csv')	
	NUM_DAYS = 20
	NUM_GENS = [5, 10, 20, 30]
	DISPATCH_FREQS = [30]

	all_df = pd.read_csv(TEST_DATA_FN)

	test_days = np.random.choice(pd.unique(all_df.date), NUM_DAYS, replace=False)

	for num_gen in NUM_GENS: 
		for disp_freq in DISPATCH_FREQS:
			print(num_gen, disp_freq)
			upsample_factor = 30 / disp_freq
			dir_name = 'foo/{}gen/{}min'.format(num_gen, disp_freq)
			os.makedirs(dir_name, exist_ok=True)
			for day in test_days:
				fn = 'profile_{}.csv'.format(str(day))
				profile = all_df[all_df.date == day].copy()

				profile.demand = interpolate_profile(profile.demand.values, upsample_factor)
				profile.demand = profile.demand * num_gen / 10

				profile.wind = interpolate_profile(profile.wind.values, upsample_factor)
				profile.wind = profile.wind * num_gen / 10

				profile.to_csv(os.path.join(dir_name, fn), index=False)
