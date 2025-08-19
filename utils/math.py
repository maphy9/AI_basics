def subtract_vectors(v1, v2):
    if len(v1) != len(v2):
        raise Exception('Can\'t subtract vectors of different sizes')
    return [v1[i] - v2[i] for i in range(len(v1))]


def multiply_vector_by_scalar(v, a):
    return [a * val for val in v]


def vector_outer_product(v1, v2):
    result = []
    for val1 in v1:
        row = []
        for val2 in v2:
            row.append(val1 * val2)
        result.append(row)
    return result


def multiply_matrix_by_scalar(m, a):
    result = []
    for row in m:
        result_row = []
        for val in row:
            result_row.append(val * a)
        result.append(result_row)
    return result


def subtract_matrices(m1, m2):
    if len(m1) != len(m2) or len(m1[0]) != len(m2[0]):
        raise Exception('Can\'t subtract matrices with different dimensions')
    return [[m1[i][j] - m2[i][j] for j in range(len(m1[0]))] for i in range(len(m1))]
