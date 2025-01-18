import numpy as np
import matplotlib.pyplot as plt

#onstants
h = 1 #for convience
m = 1
L = 1  
N = 100
dx = L /(N+1)

# Construct the Laplacian matrix
diag_main = np.full(N, 2, dtype=float)
diag_off = np.full(N-1, -1, dtype=float)

# Constant factor for dirrevatif approx
cst = h**2 / (2*m * dx**2)

H = np.diag(diag_main) + np.diag(diag_off, k=1) + np.diag(diag_off, k=-1)
H *= cst #Now we have difened the Hamiltonian matrix form

energies, eigvecs = np.linalg.eigh(H) #Using eigh here we know it is Hermitain 

#sorting everything to be save (normally come pre sorted)
idx = energies.argsort()
energies = energies[idx]
eigvecs = eigvecs[:,idx] #selecting the whole row corresponding to the index (the vector)

# We can compare the first few numeric energies to the analytical formula:
# E_n = (n^2 * pi^2 * hbar^2) / (2 m L^2)
n_range = np.arange(1, 5)  # 1st to 4th levels
E_analytic = (n_range**2 * np.pi**2 * h**2) / (2*m*L**2)

# Print out the first few numeric vs. analytic energies
print(" n   Numeric_E       Analytic_E")
for i, nval in enumerate(n_range):
    print(f"{nval}   {energies[i]:.5f}      {E_analytic[i]:.5f}")

#----plotting the first 3 eigenstates----
x_val = np.linspace(0, L, N+2) #N+2 because we have two boundary points

fig, ax = plt.subplots()
for k in range(3):
    #boundary conditions
    psi_k = np.zeros(N+2) #N is the number of inner points (points in itteration)
    psi_k[1:N+1] = eigvecs[:,k] #filling the inner points with the eigenvector

    #normalization
    norm = np.sqrt(np.sum(psi_k**2))
    psi_k /= norm

    ax.plot(x_val, psi_k, label=f"Energy: {energies[k]:.2f}")
ax.set_title("Infinite Square Well: First 3 Eigenfunctions")
ax.set_xlabel("x")
ax.set_ylabel("psi(x)")
ax.legend()
plt.show()