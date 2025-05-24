import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from mpl_toolkits.mplot3d import Axes3D

# Define three 3D linearly independent vectors
A = np.array([[3, 1, 1],
              [1, 2, 3],
              [0, 1, 2]], dtype=np.float_)

# Create a copy of A for Gram-Schmidt
B = np.array(A, dtype=np.float_)

# Step 1: Normalize the first vector
B[:, 0] = B[:, 0] / norm(B[:, 0])

# Step 2: Orthogonalize and normalize the second vector
proj_10 = B[:, 1] @ B[:, 0] * B[:, 0]
B[:, 1] = B[:, 1] - proj_10
if norm(B[:, 1]) > 1e-10:
    B[:, 1] = B[:, 1] / norm(B[:, 1])
else:
    B[:, 1] = np.zeros_like(B[:, 1])

# Step 3: Orthogonalize and normalize the third vector
proj_20 = B[:, 2] @ B[:, 0] * B[:, 0]
proj_21 = B[:, 2] @ B[:, 1] * B[:, 1]
B[:, 2] = B[:, 2] - proj_20 - proj_21
if norm(B[:, 2]) > 1e-10:
    B[:, 2] = B[:, 2] / norm(B[:, 2])
else:
    B[:, 2] = np.zeros_like(B[:, 2])

# 3D Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Original vectors (A)
colors_orig = ['blue', 'green', 'purple']
for i in range(3):
    ax.quiver(0, 0, 0, A[0, i], A[1, i], A[2, i], color=colors_orig[i], label=f'A[:,{i}]')

# Orthonormal vectors (B)
colors_ortho = ['red', 'orange', 'cyan']
for i in range(3):
    ax.quiver(0, 0, 0, B[0, i], B[1, i], B[2, i], color=colors_ortho[i], label=f'B[:,{i}]')

# Plot formatting
ax.set_xlim([-1, 4])
ax.set_ylim([-1, 4])
ax.set_zlim([-1, 4])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Gram-Schmidt Orthonormalization")
ax.legend()
plt.show()
