from presto.linalg import * 

def main():
    L = np.array([[3, 0, 0, 0], [-1, 5, 0, 0], [2, 1, 4, 0], [7, 0, 3, 6]])
    A = L @ L.T
    print(A)
    print(np.linalg.eigvalsh(A))
    print(ssor(A))
    B, C = incomplete_cholesky(A)
    print(np.linalg.inv(A))
    print(B)
    print(C)

if __name__ == "__main__":
    main()