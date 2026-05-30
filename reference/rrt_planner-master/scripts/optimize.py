import numpy as np
from scipy.interpolate import CubicSpline

class BSplineOptimizer:
    def __init__(self, degree=3, num_points=20):
        self.degree = degree
        self.num_points = num_points

    def optimize(self, path):
        path = np.array(path)
        if path.ndim != 2:
            raise ValueError("Expected input path shape (N, 6)")
        
        N, D = path.shape
        t = np.linspace(0, 1, N)
        t_fine = np.linspace(0, 1, self.num_points)

        smooth_path = np.zeros((self.num_points, D))
        for d in range(D):
            cs = CubicSpline(t, path[:, d])
            smooth_path[:, d] = cs(t_fine)

        return smooth_path.tolist()
