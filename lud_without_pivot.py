import numpy as np

def lu_decomp(mat):
  m, n = mat.shape
  s = min(m, n)
  L = np.zeros((m, s))
  L[range(s), range(s)] = 1.0
  U = np.zeros((s, n))
  A = mat.copy()

  for k in range(0, s):
    for i in range(k + 1, m):
      L[i, k] = A[i, k] / A[k, k]
    U[k, k:] = A[k, k:]
    for i in range(k + 1, m):
      for j in range(k + 1, n):
        A[i, j] = A[i, j] - L[i, k] * U[k, j]

  return L, U

if __name__ == '__main__':
  A = np.array([[2, 6, 4], [5, 7, 9]], np.float32)
  L, U = lu_decomp(A)
  print(np.allclose(A, L @ U))
  print(L @ U)

  A = np.array([[3, 5], [2, 4]], np.float32)
  L, U = lu_decomp(A)
  print(np.allclose(A, L @ U))
  print(L @ U)

  A = np.array([[3, 5, 5], [2, 4, 8]], np.float32)
  L, U = lu_decomp(A)
  print(np.allclose(A, L @ U))
  print(L @ U)


  A = np.array([[3, 5, 5], [2, 4, 8]], np.float32)
  L, U = lu_decomp(A)
  print(np.allclose(A, L @ U))
  print(L @ U)
