import numpy as np

if __name__ == '__main__':
    vector = [1, 2, 3, 4, 5] # The vector is a 1D array with 5 elements
    np_array = np.array(vector) # The shape of the array is (5,) because it is a 1D array with 5 elements
    print(f"vector={vector}, np_array={np_array}, np_array.shape={np_array.shape}")

    for i in range(1, 5):
        print(f"vector value at index {i}={vector[i]}")
        print(f"np_array value at index {i}={np_array[i]}")

    np_array = np_array.reshape(5, 1) # Now the shape of the array is (5, 1) because it is a 2D array with 5 rows and 1 column
    print(f"np_array={np_array}, np_array.shape={np_array.shape}")

    for x in range(0, np_array.shape[0]):
        for y in range(0, np_array.shape[1]):
            print(f"np_array value at index ({x}, {y})={np_array[x, y]}")


    vector = [[[1], [2], [3], [4], [5]], [[6], [7], [8], [9], [10]], [[11], [12], [13], [14], [15]], [[16], [17], [18], [19], [20]], [[21], [22], [23], [24], [25]]] # The vector is a 3D array with 5 rows, 5 columns and 1 depth
    np_array = np.array(vector) # The shape of the array is (5, 5, 1) because it is a 3D array with 5 rows, 5 columns and 1 depth
    print(f"np_array.shape={np_array.shape}")
