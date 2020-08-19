import numpy as np
from matplotlib import pyplot as plt

def func_origin(coef, x_val):
	ret = 0
	for i in range(coef.shape[0]):
		ret = ret + coef[i] * (x_val ** i)
	return ret	

def func_derive(coef, x_val):
	ret = 0
	for i in range(1, coef.shape[0]):
		ret = ret + i * coef[i] * (x_val ** (i - 1))
	return ret

def opt_NR(coef, x_init, tol, cnt_tol):
	err = 987654321.0
	itr_cnt = 0

	x_pre = x_init
	while itr_cnt < cnt_tol:
		f_der = func_origin(coef, x_pre) / func_derive(coef, x_pre)
		x_new = x_pre - f_der

		err = abs(x_new - x_pre)
		x_pre = x_new
		itr_cnt += 1
		if tol > err:
			break 
	return x_pre

def opt_NR_LMS(x, y, rank, tol, cnt_tol):
	itr_cnt = 0
	
	C_pre = np.ones(rank + 1)
	print(C_pre)
	while itr_cnt < cnt_tol:
		F = np.matmul(gen_poly(x, y, rank), np.insert(C_pre, 0, 1))
		J = gen_jacob(x, rank)
		P = opt_LMS(J, F)
		
		print("F:", F)		
		print("J:", J)		
		print("P:", P)		
		C_new = C_pre - P
		itr_cnt += 1

		err = 0
		for i in range(C_pre.shape[0]):
			err = err + (C_new[i] - C_pre[i])
		if tol > abs(err):
			break
		C_pre = C_new
		
	return C_new	

def opt_LMS(A, b):
	# Least Mean Square in python
	# https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
	data_size = A.shape[0]
	A_t = np.transpose(A)
	A_sqr = np.matmul(A_t, A)
	A_inv = np.linalg.inv(A_sqr)
	x = np.matmul(np.matmul(A_inv, A_t), b)
	return x

def gen_poly(x, y, rank):
	ret = np.zeros(shape=(x.shape[0], rank + 2))
	for i in range(ret.shape[0]):
		ret[i, 0] = -y[i]
		for j in range(1, ret.shape[1]):
			ret[i, j] = x[i] ** (j - 1)
	return ret

def mat_poly_jacob(i, j, x):
	return x[i] ** j
		
def gen_jacob(x, rank):
	ret = np.zeros(shape=(x.shape[0], rank + 1))
	for i in range(ret.shape[0]):
		for j in range(ret.shape[1]):
			ret[i, j] = mat_poly_jacob(i, j, x)
	return ret
		

def plot_func(coef, func, x_val, y_val):
	#fig = plt.figure()
	#fig.subtitle('fitted graph and data')
	#fig, ax_lst = plt.subplots(2, 2)
	x = np.arange(x_val[0], x_val[-1], 0.1)
	y = []
	for i in x:
		y.append(func(coef, i))
	y = np.array(y)
	#ax_lst[0][0].plot(x, y)
	plt.plot(x, y, x_val, y_val, 'ro')
	plt.savefig('savefile.png')
