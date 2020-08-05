import nonlinear as nl

def main():
	coef = [3, 0, -2, 1]
	res = nl.func_derive(coef, 2)
	print(res)

if __name__ == "__main__":
	main()


