import numpy as np

retrival_scores_transpose = np.array([[0.62111, 0.62698, 0.63558, 0.595, 0.64696],
                                     [0.57783, 0.5779, 0.5862, 0.57881, 0.59518]])

# Find indices and values of elements bigger than 0.6 in each row
index_value_pairs = [(np.where(row > 0.6)[0], row[row > 0.6]) for row in retrival_scores_transpose]

for i, (ind, val) in enumerate(index_value_pairs):
    print(f"Row {i}:")
    print(f"Indices: {ind}")
    print(f"Values: {val}")
    print("-------------------")
