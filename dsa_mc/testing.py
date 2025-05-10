import numpy as np



if __name__ == '__main__':

	npoints = 10000
	rng = np.random.default_rng()
	ran_sum = 0
	for i in range(npoints):

		x = rng.uniform(low=-1, high=1)
		y = rng.uniform(low=-1, high=1)
		# z = rng.uniform(low=-1, high=1)
		z = 0

		ran_sum += np.exp(-(x**2) - (y**2) - (z**2))


	integral = (2**2)*(1/npoints)*ran_sum

	print(integral)


