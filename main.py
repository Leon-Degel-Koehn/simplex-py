import numpy as np

# TODO:
# 1. (function to check if given problem in SEF)
# 2. (function to convert from SEF to canonical form for given basis)


# linear programs are always to be created in SEF
# one could potentially later consider implementing a function for generic lp
class LinearProgram:
    def __init__(self, c, A, b, z=0):
        """
        z is a constant summand one can add to have the objective
        function of form c^T * x + z which doesnt change the optimal
        x.
        """
        self.c = c
        self.A = A
        self.b = b
        self.z = z

    def canonicalize(self, basis: np.array):
        """
        Turn this LP into its canonical form for the given basis
        in-place.
        """
        A_b = self.A[:, basis]
        c_b = self.c[basis]
        A_b_inv = np.linalg.inv(A_b)
        A_b_inv_t = np.transpose(A_b_inv)
        y_t = np.transpose(A_b_inv_t @ c_b)
        self.c = np.transpose(self.c) - (y_t @ self.A)
        self.z = self.z + (y_t @ self.b)
        self.A = A_b_inv @ self.A
        self.b = A_b_inv @ self.b


def simplex(problem: LinearProgram,
            basis: np.array,
            basic_solution: np.array) -> np.array:
    # TODO: check correctness of input i.e. basic solution
    # from here on assume lp is in canonical form for basis
    # 1. pick k not in B such that c[k] > 0
    #   1.1 if no such k exists return current solution as its optimal
    found_k = False
    for k in range(len(problem.c)):
        if k in basis:
            continue
        if problem.c[k] >= 0:
            found_k = True
            break
    if not found_k:
        return basic_solution

    # 2. solve for t such that x >= 0 still holds
    A_k = problem.A[:, [k]]
    exiting = basis[0]
    target_val = A_k[0][0]
    if target_val > 0:
        t_temp = problem.b[0] / target_val
    elif target_val == 0:
        t_temp = 0
    else:
        t_temp = np.inf
    t = t_temp
    for b in range(1, len(problem.b)):
        target_val = A_k[b][0]
        if target_val > 0:
            t_temp = problem.b[b] / target_val
        elif target_val == 0:
            t_temp = 0
        else:
            t_temp = np.inf
        if t > t_temp:
            exiting = basis[b]
            t = t_temp
    # 3. compute new feasible solution via t
    basic_solution[k] = t
    for idx, entry in enumerate(basis):
        basic_solution[entry] = (problem.b[idx] - t*A_k[idx])
    # 4. compute new basis
    basis = list(filter(lambda x: x != exiting, list(basis)))
    basis.append(k)
    basis.sort()
    basis = np.array(basis)
    # 4. canonicalize problem for the new basis
    problem.canonicalize(basis)
    return simplex(problem, basis, basic_solution)


if __name__ == "__main__":
    A = np.array([
        [1, 1, 2, 0],
        [0, 1, 1, 1]
    ])
    c = np.transpose(np.array([0, 1, 3, 0]))
    b = np.transpose(np.array([2, 5]))
    basis = np.array([0, 3])
    basic_solution = np.transpose(np.array([2, 0, 0, 5]))
    lp = LinearProgram(c, A, b)
    print(simplex(lp, basis, basic_solution))
    # TODO:
    # two step method were basis and basic_solution is automatically calced.
