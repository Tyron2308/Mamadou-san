import numpy as np

if __name__ == '__main__':
   array = np.array([1, 2, 3, 6, 9, 10, 45])
   print(array)


   b = array[::2]
   b1 = array[2::]

   print(b, b1)