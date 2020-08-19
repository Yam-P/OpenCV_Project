import numpy as np
import nonlinear as nl

def main():
	#x = np.arange(-10, 10, 0.1)
	x = np.array([0, 1, 2, 3, 4, 5])
	y = np.array([-1, 0.2, 0.9, 2.1, 2, 3.5])
	#coef = np.array([1.04, 1.39, -0.15])
	
	#coef = nl.opt_NR_LMS(x, y, 3, 0.005, 100)
	F = nl.gen_poly(x, y, 2)
	coef = nl.opt_LMS(F[:,1:], y)
	nl.plot_func(coef, nl.func_origin, x, y)


if __name__ == "__main__":
	main()
