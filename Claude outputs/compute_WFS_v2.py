#!/usr/bin/env python
# coding: utf-8
"""
compute_WFS_v2.py
=================
Memory-efficient exact quantum-quench time evolution — rotated TFIM.

Key improvements over v1
------------------------
1. **Step-by-step evolution** (main fix):
   expm_multiply is called once per dt, not for the whole trajectory at once.
   Peak RAM = O(dim) instead of O(Ndt × dim).

2. **Closed-form initial state for angle=90**:
   |+⟩^⊗L is the exact GS — no expensive eigsh needed.

3. **Two backends**:
   - 'krylov'    : scipy expm_multiply (Krylov–Chebyshev, adaptive order)
   - 'chebyshev' : explicit Chebyshev expansion (Tal-Ezer & Kosloff 1984)
                   — fixed cost per step, no internal allocations, slightly
                   faster when r·dt ≳ 5 or when calling with many steps.

4. Norm monitoring, periodic flush, clean progress output.

Scalability note
----------------
dim = 2^L grows exponentially:
  L=22 → dim = 4 M,  one ψ ≈  64 MB,  full trajectory ≈ 6.5 GB (memmap)
  L=24 → dim = 16 M, one ψ ≈ 256 MB,  full trajectory ≈ 26 GB
  L=26 → dim = 67 M, one ψ ≈   1 GB,  full trajectory ≈ 107 GB
  L=28 → dim = 268 M, one ψ ≈  4 GB  (need a large-memory node)

For L > 28–30 exact methods are impractical on a single node.
Consider t-DMRG / TEBD (TeNPy, iTensor) if entanglement growth allows it.
"""

import os
import sys
import time as _wall

# ── Pin threads BEFORE importing JAX / NumPy / NetKet ────────────────────────
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2"
)
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, expm_multiply
import netket as nk

sys.path.append("../netket_with_NQS")
from class_WF import rotated_IsingModel


# ═══════════════════════════════════════════════════════════════════════════════
#  Parameters — edit here
# ═══════════════════════════════════════════════════════════════════════════════
L       = 22
gi      = 0.0          # initial transverse-field strength
gf      = 1.0          # final transverse-field strength
angle   = 90.0         # rotation angle in degrees (90 → closed-form GS)
pbc     = True

t_start = 0.0
dt      = 0.01
Ndt     = 100

# --- Solver options ---
BACKEND     = "krylov"   # "krylov"  |  "chebyshev"
TOL         = 1e-12      # Krylov tolerance / Chebyshev convergence threshold
FLUSH_EVERY = 10         # flush memmap to disk every N steps
# ═══════════════════════════════════════════════════════════════════════════════


# ── Memory / disk forecast ────────────────────────────────────────────────────
dim = 2**L
bytes_psi = dim * 16   # complex128 = 16 bytes
print(f"L={L}  →  dim = {dim:,}")
print(f"  One wavefunction : {bytes_psi / 1e6:.1f} MB")
print(f"  Full trajectory  : {bytes_psi * (Ndt + 1) / 1e9:.2f} GB  (streamed to memmap)")
print()

hi = nk.hilbert.Spin(s=1/2, N=L, inverted_ordering=True)


# ── Initial state ─────────────────────────────────────────────────────────────
if angle == 90.0:
    # |+⟩^⊗L is the exact GS of −∑ Xᵢ (gi=0, angle=90°) — no diagonalisation
    psi_t = np.full(dim, 1.0 / np.sqrt(dim), dtype=np.complex128)
    print("Initial state: |+⟩^L  (uniform superposition — exact, no eigsh).")
else:
    print(f"Computing GS of H_in (gi={gi}, angle={angle}°) …", end=" ", flush=True)
    H_in  = rotated_IsingModel(angle * np.pi / 180, gi, L, hi, pbc=pbc).to_sparse()
    E0, v = eigsh(H_in, k=1, which="SA")
    psi_t = np.asarray(v[:, 0], dtype=np.complex128)
    del H_in            # free immediately — we only needed the GS
    print(f"E₀ = {E0[0]:.10f}")


# ── Final Hamiltonian ─────────────────────────────────────────────────────────
print(f"Building H_out (gf={gf}) …", end=" ", flush=True)
H_out  = rotated_IsingModel(angle * np.pi / 180, gf, L, hi, pbc=pbc).to_sparse()
A_step = -1j * dt * H_out      # exp(A_step)|ψ⟩ = e^{−iH dt}|ψ⟩  per step
print("done.\n")


# ── Chebyshev backend ─────────────────────────────────────────────────────────
# Reference: Tal-Ezer & Kosloff, J. Chem. Phys. 81, 3967 (1984)
#
#   exp(−iHt) = exp(−ict) · Σ_n  ε_n · (−i)^n · J_n(r·t) · T_n(H̃)
#
# where  c = (Emax+Emin)/2,  r = (Emax−Emin)/2,  H̃ = (H−c)/r ∈ [−1,1],
#        ε_0 = 1,  ε_n = 2 for n≥1,  J_n = Bessel functions of the first kind,
#        T_n = Chebyshev polynomials (via three-term recurrence on vectors).

_cheb: dict = {}   # module-level cache; built once at first step

def _cheb_init(H_sparse: sp.spmatrix) -> None:
    """Estimate spectral bounds and store rescaled Hamiltonian."""
    print("  Chebyshev: estimating spectral bounds …", end=" ", flush=True)
    E_min = eigsh(H_sparse, k=1, which="SA", return_eigenvectors=False)[0]
    E_max = eigsh(H_sparse, k=1, which="LA", return_eigenvectors=False)[0]
    # small margin so T_n(H̃) stays in [−1, 1] despite floating-point rounding
    margin = max(0.01 * (E_max - E_min), 1e-6)
    c = 0.5 * (E_max + E_min)
    r = 0.5 * (E_max - E_min) + margin
    _cheb["c"] = c
    _cheb["r"] = r
    _cheb["Hs"] = (H_sparse - c * sp.eye(H_sparse.shape[0], format="csr")) / r
    print(f"E ∈ [{E_min:.4f}, {E_max:.4f}],  r = {r:.4f}")


def chebyshev_step(psi: np.ndarray, dt: float) -> np.ndarray:
    """Return exp(−i H dt)|ψ⟩ via truncated Chebyshev series."""
    from scipy.special import jn as _jn

    c, r, Hs = _cheb["c"], _cheb["r"], _cheb["Hs"]
    x = r * dt                              # Bessel argument

    # Number of terms: |J_n(x)| drops super-exponentially for n ≫ x.
    # Safe upper bound: n_max = ceil(x) + 10·log₁₀(1/TOL) + 20
    n_max = max(int(np.ceil(x)) + int(10 * np.log10(1.0 / (TOL + 1e-30))) + 20, 30)

    phi0 = psi.copy()                       # T_0(H̃)|ψ⟩ = |ψ⟩
    phi1 = Hs.dot(psi)                      # T_1(H̃)|ψ⟩ = H̃|ψ⟩
    mi   = -1j                              # (−i)^1

    res = _jn(0, x) * phi0 + 2.0 * mi * _jn(1, x) * phi1

    for n in range(2, n_max + 1):
        mi   *= -1j                         # (−i)^n
        phi2  = 2.0 * Hs.dot(phi1) - phi0  # Chebyshev recurrence
        coeff = 2.0 * mi * _jn(n, x)
        res  += coeff * phi2
        if abs(coeff) < TOL * 1e-3:        # converged
            break
        phi0, phi1 = phi1, phi2

    return np.exp(-1j * c * dt) * res


# ── Initialise Chebyshev cache if needed ─────────────────────────────────────
if BACKEND == "chebyshev":
    _cheb_init(H_out)
elif BACKEND != "krylov":
    raise ValueError(f"Unknown backend: {BACKEND!r}  (choose 'krylov' or 'chebyshev')")


# ── Output memmap ─────────────────────────────────────────────────────────────
suffix  = ("_pbc" if pbc else "") + (f"_{angle}" if angle != 0.0 else "")
out_dir = f"wavefunctions_data{suffix}"
os.makedirs(out_dir, exist_ok=True)

fname = os.path.join(
    out_dir,
    f"wavefunctions_L{L}gi{gi:.2f}gf{gf:.2f}ti{t_start}dt{dt}Nt{Ndt}.npy"
)

psi_hist = np.lib.format.open_memmap(
    fname, mode="w+", dtype=np.complex128, shape=(Ndt + 1, dim)
)
psi_hist[0] = psi_t


# ── Main evolution loop ───────────────────────────────────────────────────────
# We apply exp(−iH dt) one step at a time:
#   - Peak RAM = one wavefunction  (not Ndt×dim like the all-at-once approach)
#   - Each step writes directly to the memmap on disk
#   - Norm is monitored to detect drift; renormalisation keeps errors bounded

print(f"Evolving  L={L}  dt={dt}  Ndt={Ndt}  backend={BACKEND}  tol={TOL}")
t_wall = _wall.perf_counter()
max_norm_err = 0.0

for step in range(1, Ndt + 1):
    # ── apply propagator ─────────────────────────────────────────────────────
    if BACKEND == "krylov":
        # expm_multiply(M, v) → exp(M) · v   [Krylov–Chebyshev, adaptive order]
        psi_t = expm_multiply(A_step, psi_t, tol=TOL)
    else:
        psi_t = chebyshev_step(psi_t, dt)

    # ── norm check + renormalisation ─────────────────────────────────────────
    norm    = np.linalg.norm(psi_t)
    err     = abs(norm - 1.0)
    max_norm_err = max(max_norm_err, err)
    psi_t  /= norm                          # keep ‖ψ‖ = 1 at every step

    # ── stream to disk ───────────────────────────────────────────────────────
    psi_hist[step] = psi_t

    if step % FLUSH_EVERY == 0:
        psi_hist.flush()
        elapsed = _wall.perf_counter() - t_wall
        eta     = elapsed / step * (Ndt - step)
        print(f"  step {step:4d}/{Ndt}   {elapsed:7.1f} s elapsed   "
              f"~{eta:5.0f} s remaining   max|‖ψ‖−1| = {max_norm_err:.1e}")
        max_norm_err = 0.0

psi_hist.flush()
del psi_hist

total = _wall.perf_counter() - t_wall
print(f"\nDone.  gf={gf}  total wall time: {total:.1f} s")
print(f"Saved → {fname}")
