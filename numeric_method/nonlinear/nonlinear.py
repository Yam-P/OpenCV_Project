import numpy as np

def func_origin(coef, x):
	ret = 0
	for i in range(len(coef)):
		ret = ret + coef[i] * (x ** i)
	return ret	

def func_derive(coef, x):
	ret = 0
	for i in range(1, len(coef)):
		ret = ret + i * coef[i] * (x ** (i - 1))
	return ret
