import numpy as np

class NLinkArm:
    def __init__(self, dh_params, joint_limits=None):
        """
        dh_params: List of DH parameters in the order [alpha, a, d, theta]
        joint_limits: list of (min, max) tuples in radians
        """
        self.dh_params = dh_params
        self.num_joints = len(dh_params)
        if joint_limits is not None:
            self.joint_limits = np.array(joint_limits)
        else:
            # default: all joints [-pi, pi]
            self.joint_limits = np.array([[-np.pi, np.pi]] * self.num_joints)

    def dh_transform(self, alpha, a, d, theta):
        ca, sa = np.cos(alpha), np.sin(alpha)
        ct, st = np.cos(theta), np.sin(theta)
        return np.array([
            [ct, -st, 0, a],
            [st * ca, ct * ca, -sa, -d * sa],
            [st * sa, ct * sa, ca, d * ca],
            [0, 0, 0, 1]
        ])

    def forward_kinematics(self, joint_angles):
        assert len(joint_angles) == self.num_joints
        T = np.eye(4)
        for i, (alpha, a, d, theta) in enumerate(self.dh_params):
            T_i = self.dh_transform(alpha, a, d, theta + joint_angles[i])
            T = T @ T_i
        return np.array(T[:3, 3])

    def get_joint_positions(self, q):
        positions = [np.array([0.0, 0.0, 0.0])]
        T = np.eye(4)
        for i in range(self.num_joints):
            alpha, a, d, theta_offset = self.dh_params[i]
            theta = theta_offset + q[i]
            T_i = self.dh_transform(alpha, a, d, theta)
            T = T @ T_i
            positions.append(np.array(T[:3, 3]))  # Make a copy of each position
        return np.array(positions)

    def inverse_kinematics(self, target_pos, q_init=np.array([357, 21, 150, 272, 320, 273]) / 180 * np.pi, max_iter=100, tol=1e-3, alpha=0.5):
        # print(f"\nStarting IK for target position: {target_pos}")
        if q_init is None:
            q = np.zeros(len(self.dh_params))
        else:
            q = np.array(q_init)
            # print(f"Initial joint angles: {q}")

        weights = np.array([2.0, 2.0, 0.5, 1.0, 0.1, 0.1])
        # print(f"Using weights: {weights}")

        for iter_count in range(max_iter):
            current_pos = self.forward_kinematics(q)
            error = target_pos - current_pos
            error_norm = np.linalg.norm(error)
            
            if iter_count % 10 == 0:  # Print every 10 iterations
                # print(f"IK iteration {iter_count}: error = {error_norm:.6f}")
                # print(f"Current position: {current_pos}")
                # print(f"Current angles: {q}")
                pass

            if error_norm < tol:
                # print(f"IK converged after {iter_count} iterations")
                return q

            J = self.jacobian_numerical(q)
            try:
                dq = alpha * self.weighted_pseudo_inverse(J, weights).dot(error)
            except np.linalg.LinAlgError:
                print("IK failed: Pseudo-inverse calculation error")
                return None

            q = self.wrap_to_joint_limits(q + dq)

        print(f"IK failed to converge after {max_iter} iterations")
        print(f"Final error: {error_norm:.6f}")
        return None  # failed to converge

    def jacobian_numerical(self, q, delta=1e-6):
        n = len(q)
        J = np.zeros((3, n))
        f0 = self.forward_kinematics(q)
        for i in range(n):
            dq = np.zeros(n)
            dq[i] = delta
            f1 = self.forward_kinematics(q + dq)
            J[:, i] = (f1 - f0) / delta
        return J

    def weighted_pseudo_inverse(self, J, W_diag, lamb=1e-4):
        W = np.diag(W_diag)
        JTJ = J.T @ J
        regularizer = lamb * W.T @ W
        return np.linalg.inv(JTJ + regularizer) @ J.T

    def wrap_angle(self, theta):
        """Wrap angle to [-pi, pi]"""
        return (theta + np.pi) % (2 * np.pi) - np.pi

    def wrap_to_joint_limits(self, q):
        """Wrap angles with periodic joint limits"""
        q_wrapped = np.zeros_like(q)
        for i in range(len(q)):
            q_i = self.wrap_angle(q[i])
            lower, upper = self.joint_limits[i]
            if lower <= upper:
                # normal range
                q_wrapped[i] = np.clip(q_i, lower, upper)
            else:
                # wrapped range: valid if q_i < upper or q_i > lower
                if q_i < upper or q_i > lower:
                    q_wrapped[i] = q_i
                else:
                    # pick nearest limit
                    d_lower = np.abs(self.wrap_angle(q_i - lower))
                    d_upper = np.abs(self.wrap_angle(q_i - upper))
                    q_wrapped[i] = lower if d_lower < d_upper else upper
        return q_wrapped
