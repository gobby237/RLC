import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
from scipy.stats import chi2 as chi2_dist
import multiprocessing.pool

# ============================================================
# CONFIGURAZIONE
# ============================================================

FILE_NAME = "dati.txt"

# conversioni unità
TIME_SCALE = 1e-12   # tempi -> s
VOLT_SCALE = 1e-6    # tensioni -> V

# capacità
Ccap = 4.76e-9
sigmaC = 0.02e-9

# guess iniziali
VL0_init = 2.3
omega0_init = 8.34e5
delta_init = 1.10e5
Voff_init = 0.02
t0_init = -9.0e-8

# parametri scansione chi2
NSI = 2.0        # scansione entro +- NSI * sigma dal best fit
STEP = 81        # discretizzazione della griglia 3D
NPOOL = 16       # processi/threads per il calcolo della mappa

# eventuale piccolo margine oltre t0_bestfit nel taglio
SHIFT_EXTRA = 0.0

# ============================================================
# FUNZIONI MODELLO
# ============================================================

def model_vl_full(t, VL0, omega0, delta, Voff, t0):
    x = t - t0
    wd2 = omega0**2 - delta**2
    if np.any(wd2 <= 0):
        return np.full_like(t, 1e30, dtype=float)
    wd = np.sqrt(wd2)
    return VL0 * np.exp(-delta * x) * (
        np.cos(wd * x) - (delta / wd) * np.sin(wd * x)
    ) + Voff


def model_vl_reduced(t, VL0, omega0, delta):
    wd2 = omega0**2 - delta**2
    if np.any(wd2 <= 0):
        return np.full_like(t, 1e30, dtype=float)
    wd = np.sqrt(wd2)
    return VL0 * np.exp(-delta * t) * (
        np.cos(wd * t) - (delta / wd) * np.sin(wd * t)
    )

# ============================================================
# FUNZIONI CHI2 SU GRIGLIA
# ============================================================

tempo = None
Vout = None
eVout = None

VL0_chi = None
omega0_chi = None
delta_chi = None
mappa = None
step = STEP


def fitchi2(i, j, k):
    global mappa
    AA = VL0_chi[i]
    BB = omega0_chi[j]
    CC = delta_chi[k]

    if BB <= CC:
        mappa[i, j, k] = 1e30
        return

    residui = Vout - model_vl_reduced(tempo, AA, BB, CC)
    chi2_val = np.sum((residui / eVout) ** 2)
    mappa[i, j, k] = chi2_val


def profi2D(axis, matrix3D):
    # 1 -> profilo sul primo asse escluso: restituisce (omega0, delta)
    if axis == 1:
        return np.array([[np.min(matrix3D[:, b, c]) for b in range(step)] for c in range(step)])
    # 2 -> restituisce (VL0, delta)
    if axis == 2:
        return np.array([[np.min(matrix3D[a, :, c]) for a in range(step)] for c in range(step)])
    # 3 -> restituisce (VL0, omega0)
    if axis == 3:
        return np.array([[np.min(matrix3D[a, b, :]) for a in range(step)] for b in range(step)])


def profi1D(axis, matrix3D):
    # axis è la lista degli assi da eliminare
    if 1 in axis:
        mappa2D = np.array([[np.min(matrix3D[:, b, c]) for b in range(step)] for c in range(step)])
        if 2 in axis:
            # resta asse 3? No: in questa convenzione restituiamo profilo di delta
            mappa1D = np.array([np.min(mappa2D[:, c]) for c in range(step)])
        if 3 in axis:
            # profilo di omega0
            mappa1D = np.array([np.min(mappa2D[b, :]) for b in range(step)])
    else:
        # profilo di VL0
        mappa2D = np.array([[np.min(matrix3D[a, :, c]) for a in range(step)] for c in range(step)])
        mappa1D = np.array([np.min(mappa2D[a, :]) for a in range(step)])

    return mappa1D


def find_crossings_nearest(xgrid, ygrid, level, x0):
    diff = np.abs(ygrid - level)

    mask_left = xgrid < x0
    mask_right = xgrid > x0

    if np.any(mask_left):
        idx_left_local = np.argmin(diff[mask_left])
        idx_left = np.where(mask_left)[0][idx_left_local]
    else:
        idx_left = None

    if np.any(mask_right):
        idx_right_local = np.argmin(diff[mask_right])
        idx_right = np.where(mask_right)[0][idx_right_local]
    else:
        idx_right = None

    return idx_left, idx_right

# ============================================================
# LETTURA DATI
# ============================================================

raw = np.loadtxt(FILE_NAME)

if raw.ndim != 2 or raw.shape[1] < 3:
    raise ValueError("Il file deve contenere almeno 3 colonne: tempo, tensione, errore_tensione")

t_raw = raw[:, 0] * TIME_SCALE
V_raw = raw[:, 1] * VOLT_SCALE
sigma_raw = raw[:, 2] * VOLT_SCALE

mask = np.isfinite(t_raw) & np.isfinite(V_raw) & np.isfinite(sigma_raw) & (sigma_raw > 0)
t_raw = t_raw[mask]
V_raw = V_raw[mask]
sigma_raw = sigma_raw[mask]

if len(t_raw) < 10:
    raise ValueError("Troppi pochi punti validi per il fit")

# ============================================================
# PRIMO FIT A 5 PARAMETRI
# ============================================================

p0_full = [VL0_init, omega0_init, delta_init, Voff_init, t0_init]

lower_bounds_full = [0.0, 1e4, 1e3, -10.0, t_raw.min() - 1e-5]
upper_bounds_full = [20.0, 1e7, 1e7,  10.0, t_raw.max() + 1e-5]

popt_full, pcov_full = curve_fit(
    model_vl_full,
    t_raw,
    V_raw,
    p0=p0_full,
    sigma=sigma_raw,
    absolute_sigma=True,
    bounds=(lower_bounds_full, upper_bounds_full),
    maxfev=300000
)

perr_full = np.sqrt(np.diag(pcov_full))

VL0_full, omega0_full, delta_full, Voff_full, t0_full = popt_full
sVL0_full, somega0_full, sdelta_full, sVoff_full, st0_full = perr_full

Vfit_full = model_vl_full(t_raw, *popt_full)
res_full = V_raw - Vfit_full
chi2_full = np.sum((res_full / sigma_raw) ** 2)
ndf_full = len(t_raw) - len(popt_full)
chi2red_full = chi2_full / ndf_full
prob_full = 1.0 - chi2_dist.cdf(chi2_full, ndf_full)

print("\n================ PRIMO FIT A 5 PARAMETRI ================")
print(f"chi2/ndf = {chi2_full:.1f} / {ndf_full} = {chi2red_full:.3f}")
print(f"Prob     = {prob_full:.6f}")
print()
print(f"VL0     = ({VL0_full:.4f} +/- {sVL0_full:.4f}) V")
print(f"omega0  = ({omega0_full:.4e} +/- {somega0_full:.2e}) rad/s")
print(f"delta   = ({delta_full:.4e} +/- {sdelta_full:.2e}) rad/s")
print(f"Voff    = ({Voff_full:.4f} +/- {sVoff_full:.4f}) V")
print(f"t0      = ({t0_full:.4e} +/- {st0_full:.2e}) s")
print("=========================================================")

# ============================================================
# SHIFT RIGIDO DEI DATI
# ============================================================

mask2 = t_raw > (t0_full + SHIFT_EXTRA)

tempo = t_raw[mask2] - t0_full
Vout = V_raw[mask2] - Voff_full
eVout = sigma_raw[mask2]

# forza asse dei tempi a partire da zero
tempo = tempo - tempo.min()

N = len(tempo)
if N < 10:
    raise ValueError("Dopo il taglio in t0 restano troppo pochi punti.")

# ============================================================
# SECONDO FIT A 3 PARAMETRI
# ============================================================

p0_red = [VL0_full, omega0_full, delta_full]

lower_bounds_red = [0.0, 1e4, 1e3]
upper_bounds_red = [20.0, 1e7, 1e7]

popt, pcov = curve_fit(
    model_vl_reduced,
    tempo,
    Vout,
    p0=p0_red,
    sigma=eVout,
    absolute_sigma=True,
    bounds=(lower_bounds_red, upper_bounds_red),
    maxfev=300000
)

perr = np.sqrt(np.diag(pcov))

VL0_BF, omega0_BF, delta_BF = popt
eVL0_BF, eomega0_BF, edelta_BF = perr

Vfit = model_vl_reduced(tempo, *popt)
residuals = Vout - Vfit

chi2 = np.sum((residuals / eVout) ** 2)
ndf = N - 3
chi2_red = chi2 / ndf
prob = 1.0 - chi2_dist.cdf(chi2, ndf)

wd_BF = np.sqrt(omega0_BF**2 - delta_BF**2)

dwd_domega0 = omega0_BF / wd_BF
dwd_ddelta = -delta_BF / wd_BF
cov_omega0_delta = pcov[1, 2]

var_wd = (
    dwd_domega0**2 * pcov[1, 1]
    + dwd_ddelta**2 * pcov[2, 2]
    + 2.0 * dwd_domega0 * dwd_ddelta * cov_omega0_delta
)
swd_BF = np.sqrt(max(var_wd, 0.0))

Td_BF = 2.0 * np.pi / wd_BF
sTd_BF = (2.0 * np.pi / wd_BF**2) * swd_BF

Q_BF = omega0_BF / (2.0 * delta_BF)
dQ_domega0 = 1.0 / (2.0 * delta_BF)
dQ_ddelta = -omega0_BF / (2.0 * delta_BF**2)
var_Q = (
    dQ_domega0**2 * pcov[1, 1]
    + dQ_ddelta**2 * pcov[2, 2]
    + 2.0 * dQ_domega0 * dQ_ddelta * cov_omega0_delta
)
sQ_BF = np.sqrt(max(var_Q, 0.0))

L_BF = 1.0 / (omega0_BF**2 * Ccap)
dL_domega0 = -2.0 / (Ccap * omega0_BF**3)
dL_dC = -1.0 / (omega0_BF**2 * Ccap**2)
var_L = dL_domega0**2 * pcov[1, 1] + dL_dC**2 * sigmaC**2
sL_BF = np.sqrt(max(var_L, 0.0))

R_BF = 2.0 * L_BF * delta_BF
cov_L_delta = dL_domega0 * cov_omega0_delta
dR_dL = 2.0 * delta_BF
dR_ddelta = 2.0 * L_BF
var_R = (
    dR_dL**2 * var_L
    + dR_ddelta**2 * pcov[2, 2]
    + 2.0 * dR_dL * dR_ddelta * cov_L_delta
)
sR_BF = np.sqrt(max(var_R, 0.0))

print("\n================ FIT A 3 PARAMETRI ======================")
print(f"chi2/ndf = {chi2:.1f} / {ndf} = {chi2_red:.3f}")
print(f"Prob     = {prob:.6f}")
print()
print(f"VL0     = ({VL0_BF:.4f} +/- {eVL0_BF:.4f}) V")
print(f"omega0  = ({omega0_BF:.4e} +/- {eomega0_BF:.2e}) rad/s")
print(f"delta   = ({delta_BF:.4e} +/- {edelta_BF:.2e}) rad/s")
print(f"omega_d = ({wd_BF:.4e} +/- {swd_BF:.2e}) rad/s")
print(f"T_d     = ({Td_BF:.4e} +/- {sTd_BF:.2e}) s")
print(f"Q       = ({Q_BF:.4f} +/- {sQ_BF:.2e})")
print()
print(f"L = ({L_BF:.4e} +/- {sL_BF:.2e}) H")
print(f"R = ({R_BF:.4f} +/- {sR_BF:.4f}) Ohm")
print("=========================================================")

# ============================================================
# GRAFICO FIT + RESIDUI
# ============================================================

t_dense = np.linspace(0.0, tempo.max(), 5000)
V_dense = model_vl_reduced(t_dense, *popt)

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(10, 7),
    sharex=True,
    gridspec_kw={'height_ratios': [3, 1]}
)

ax1.errorbar(
    tempo, Vout, yerr=eVout,
    fmt='o', color='black', ecolor='black',
    elinewidth=0.8, capsize=2, ms=1.5,
    label='Dati'
)
ax1.plot(t_dense, V_dense, lw=2, color='tab:blue', label='Fit')
ax1.set_ylabel(r'$V_L$ [V]')
ax1.set_title('Fit scarica oscillante RLC ai capi di L')
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2.errorbar(
    tempo, residuals, yerr=eVout,
    fmt='o', color='black', ecolor='black',
    elinewidth=0.8, capsize=2, ms=1.5
)
ax2.axhline(0, color='red', lw=1)
ax2.set_xlabel('t [s]')
ax2.set_ylabel('Residui')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, tempo.max())

plt.tight_layout()
plt.show()

# ============================================================
# MAPPA 3D DEL CHI2
# ============================================================

A0, A1 = VL0_BF - NSI * eVL0_BF, VL0_BF + NSI * eVL0_BF
B0, B1 = omega0_BF - NSI * eomega0_BF, omega0_BF + NSI * eomega0_BF
C0, C1 = delta_BF - NSI * edelta_BF, delta_BF + NSI * edelta_BF

VL0_chi = np.linspace(A0, A1, STEP)
omega0_chi = np.linspace(B0, B1, STEP)
delta_chi = np.linspace(C0, C1, STEP)

mappa = np.zeros((STEP, STEP, STEP), dtype=float)

items = [(i, j, k) for i in range(STEP) for j in range(STEP) for k in range(STEP)]

pool = multiprocessing.pool.ThreadPool(NPOOL)
pool.starmap(fitchi2, items, chunksize=20)
pool.close()
pool.join()

mappa = np.asarray(mappa)

chi2_min = np.min(mappa)
argchi2_min = np.unravel_index(np.argmin(mappa), mappa.shape)

VL0_chi_BF = VL0_chi[argchi2_min[0]]
omega0_chi_BF = omega0_chi[argchi2_min[1]]
delta_chi_BF = delta_chi[argchi2_min[2]]

wd_chi_BF = np.sqrt(omega0_chi_BF**2 - delta_chi_BF**2)
Td_chi_BF = 2.0 * np.pi / wd_chi_BF
Q_chi_BF = omega0_chi_BF / (2.0 * delta_chi_BF)
L_chi_BF = 1.0 / (omega0_chi_BF**2 * Ccap)
R_chi_BF = 2.0 * L_chi_BF * delta_chi_BF

print("\n================ MINIMO DEL CHI2 SU GRIGLIA =============")
print(f"chi2_min = {chi2_min:.4f}")
print(f"VL0     = {VL0_chi_BF:.6f} V")
print(f"omega0  = {omega0_chi_BF:.6e} rad/s")
print(f"delta   = {delta_chi_BF:.6e} rad/s")
print(f"omega_d = {wd_chi_BF:.6e} rad/s")
print(f"T_d     = {Td_chi_BF:.6e} s")
print(f"Q       = {Q_chi_BF:.6f}")
print(f"L       = {L_chi_BF:.6e} H")
print(f"R       = {R_chi_BF:.6f} Ohm")
print("=========================================================")

# ============================================================
# PROFILAZIONI 1D E 2D
# ============================================================

# 2D
chi2_omega0_delta = profi2D(1, mappa)   # assi: omega0, delta
chi2_VL0_delta = profi2D(2, mappa)      # assi: VL0, delta

# 1D
prof_omega0 = profi1D([1, 3], mappa)    # profilo su omega0
prof_delta = profi1D([1, 2], mappa)     # profilo su delta
prof_VL0 = profi1D([2, 3], mappa)       # profilo su VL0

lvl = chi2_min + 1.0

idx_omega0_left, idx_omega0_right = find_crossings_nearest(omega0_chi, prof_omega0, lvl, omega0_chi_BF)
idx_delta_left, idx_delta_right = find_crossings_nearest(delta_chi, prof_delta, lvl, delta_chi_BF)
idx_VL0_left, idx_VL0_right = find_crossings_nearest(VL0_chi, prof_VL0, lvl, VL0_chi_BF)

err_omega0_minus = omega0_chi_BF - omega0_chi[idx_omega0_left]
err_omega0_plus = omega0_chi[idx_omega0_right] - omega0_chi_BF

err_delta_minus = delta_chi_BF - delta_chi[idx_delta_left]
err_delta_plus = delta_chi[idx_delta_right] - delta_chi_BF

err_VL0_minus = VL0_chi_BF - VL0_chi[idx_VL0_left]
err_VL0_plus = VL0_chi[idx_VL0_right] - VL0_chi_BF

print("\n================ ERRORI DA CHI2_min + 1 =================")
print(f"VL0     = ({VL0_chi_BF:.6f} - {err_VL0_minus:.3e} + {err_VL0_plus:.3e}) V")
print(f"omega0  = ({omega0_chi_BF:.6e} - {err_omega0_minus:.3e} + {err_omega0_plus:.3e}) rad/s")
print(f"delta   = ({delta_chi_BF:.6e} - {err_delta_minus:.3e} + {err_delta_plus:.3e}) rad/s")
print("=========================================================")

# ============================================================
# PLOT MAPPA CHI2: omega0 vs delta
# ============================================================

cmap = mpl.colormaps['plasma'].reversed()
levels_fill_1 = np.linspace(np.min(chi2_omega0_delta), np.max(chi2_omega0_delta), 100)
line_c = 'gray'

fig, ax = plt.subplots(
    2, 2, figsize=(6.4, 5.6),
    constrained_layout=True,
    height_ratios=[3, 1],
    width_ratios=[1, 3],
    sharex='col',
    sharey='row'
)

fig.suptitle(r'$\chi^2(\omega_0,\delta)$')

im = ax[0, 1].contourf(omega0_chi, delta_chi, chi2_omega0_delta, levels=levels_fill_1, cmap=cmap)
cbar = fig.colorbar(
    im, ax=ax[0, 1], shrink=0.9,
    ticks=[chi2_min, chi2_min + 1, chi2_min + 2.3, chi2_min + 3.8]
)
cbar.set_label(r'$\chi^2$', rotation=360)

CS = ax[0, 1].contour(
    omega0_chi, delta_chi, chi2_omega0_delta,
    levels=[chi2_min + 0.0001, chi2_min + 1, chi2_min + 2.3, chi2_min + 3.8],
    linewidths=1,
    colors='k',
    alpha=0.5,
    linestyles='dotted'
)
ax[0, 1].clabel(CS, inline=True, fontsize=9, fmt='%.0f')
ax[0, 1].text(omega0_chi_BF, delta_chi_BF, f'{chi2_min:.0f}', color='k', alpha=0.6, fontsize=9)

ax[0, 1].plot([B0, B1], [delta_chi[idx_delta_right], delta_chi[idx_delta_right]], color=line_c, ls='dashed')
ax[0, 1].plot([B0, B1], [delta_chi[idx_delta_left], delta_chi[idx_delta_left]], color=line_c, ls='dashed')
ax[0, 1].plot([omega0_chi[idx_omega0_right], omega0_chi[idx_omega0_right]], [C0, C1], color=line_c, ls='dashed')
ax[0, 1].plot([omega0_chi[idx_omega0_left], omega0_chi[idx_omega0_left]], [C0, C1], color=line_c, ls='dashed')

ax[0, 0].plot(prof_delta, delta_chi, ls='-')
ax[0, 0].plot([chi2_min - 2, chi2_min + 10], [delta_chi[idx_delta_right], delta_chi[idx_delta_right]], color=line_c, ls='dashed')
ax[0, 0].plot([chi2_min - 2, chi2_min + 10], [delta_chi[idx_delta_left], delta_chi[idx_delta_left]], color=line_c, ls='dashed')

ax[0, 0].set_xticks([chi2_min, chi2_min + 2, chi2_min + 4, chi2_min + 6])
ax[0, 0].text(chi2_min + 2, delta_chi_BF, f'{delta_chi_BF:.3e}', color='k', alpha=0.5, fontsize=9)
ax[0, 0].text(chi2_min + 4, delta_chi[idx_delta_right], f'{err_delta_plus:.0e}', color='b', alpha=0.7, fontsize=9)
ax[0, 0].text(chi2_min + 4, delta_chi[idx_delta_left], f'{-err_delta_minus:.0e}', color='r', alpha=0.7, fontsize=9)

ax[1, 1].plot(omega0_chi, prof_omega0)
ax[1, 1].plot([omega0_chi[idx_omega0_right], omega0_chi[idx_omega0_right]], [chi2_min - 2, chi2_min + 10], color=line_c, ls='dashed')
ax[1, 1].plot([omega0_chi[idx_omega0_left], omega0_chi[idx_omega0_left]], [chi2_min - 2, chi2_min + 10], color=line_c, ls='dashed')

ax[1, 1].text(omega0_chi_BF, chi2_min + 2, f'{omega0_chi_BF:.5e}', color='k', alpha=0.5, fontsize=9)
ax[1, 1].text(omega0_chi[idx_omega0_right], chi2_min + 4, f'{err_omega0_plus:.0e}', color='b', alpha=0.7, fontsize=9)
ax[1, 1].text(omega0_chi[idx_omega0_left], chi2_min + 4, f'{-err_omega0_minus:.0e}', color='r', alpha=0.7, fontsize=9)

ax[1, 1].set_yticks([chi2_min, chi2_min + 4, chi2_min + 8])

ax[1, 0].set_axis_off()
ax[0, 0].set_ylabel(r'$\delta\;(\mathrm{rad/s})$')
ax[1, 1].set_xlabel(r'$\omega_0\;(\mathrm{rad/s})$')
ax[0, 0].set_xlim(chi2_min - 2, chi2_min + 10)
ax[1, 1].set_ylim(chi2_min - 2, chi2_min + 10)

plt.show()

# ============================================================
# PLOT MAPPA CHI2: VL0 vs delta
# ============================================================

levels_fill_2 = np.linspace(np.min(chi2_VL0_delta), np.max(chi2_VL0_delta), 100)

fig, ax = plt.subplots(
    2, 2, figsize=(6.4, 5.6),
    constrained_layout=True,
    height_ratios=[3, 1],
    width_ratios=[1, 3],
    sharex='col',
    sharey='row'
)

fig.suptitle(r'$\chi^2(V_{L0},\delta)$')

im = ax[0, 1].contourf(VL0_chi, delta_chi, chi2_VL0_delta, levels=levels_fill_2, cmap=cmap)
cbar = fig.colorbar(
    im, ax=ax[0, 1], shrink=0.9,
    ticks=[chi2_min, chi2_min + 1, chi2_min + 2.3, chi2_min + 3.8]
)
cbar.set_label(r'$\chi^2$', rotation=360)

CS = ax[0, 1].contour(
    VL0_chi, delta_chi, chi2_VL0_delta,
    levels=[chi2_min + 0.0001, chi2_min + 1, chi2_min + 2.3, chi2_min + 3.8],
    linewidths=1,
    colors='k',
    alpha=0.5,
    linestyles='dotted'
)
ax[0, 1].clabel(CS, inline=True, fontsize=9, fmt='%.0f')
ax[0, 1].text(VL0_chi_BF, delta_chi_BF, f'{chi2_min:.0f}', color='k', alpha=0.6, fontsize=9)

ax[0, 1].plot([A0, A1], [delta_chi[idx_delta_right], delta_chi[idx_delta_right]], color=line_c, ls='dashed')
ax[0, 1].plot([A0, A1], [delta_chi[idx_delta_left], delta_chi[idx_delta_left]], color=line_c, ls='dashed')
ax[0, 1].plot([VL0_chi[idx_VL0_right], VL0_chi[idx_VL0_right]], [C0, C1], color=line_c, ls='dashed')
ax[0, 1].plot([VL0_chi[idx_VL0_left], VL0_chi[idx_VL0_left]], [C0, C1], color=line_c, ls='dashed')

ax[0, 0].plot(prof_delta, delta_chi, ls='-')
ax[0, 0].plot([chi2_min - 2, chi2_min + 10], [delta_chi[idx_delta_right], delta_chi[idx_delta_right]], color=line_c, ls='dashed')
ax[0, 0].plot([chi2_min - 2, chi2_min + 10], [delta_chi[idx_delta_left], delta_chi[idx_delta_left]], color=line_c, ls='dashed')

ax[0, 0].set_xticks([chi2_min, chi2_min + 2, chi2_min + 4, chi2_min + 6])
ax[0, 0].text(chi2_min + 2, delta_chi_BF, f'{delta_chi_BF:.3e}', color='k', alpha=0.5, fontsize=9)
ax[0, 0].text(chi2_min + 4, delta_chi[idx_delta_right], f'{err_delta_plus:.0e}', color='b', alpha=0.7, fontsize=9)
ax[0, 0].text(chi2_min + 4, delta_chi[idx_delta_left], f'{-err_delta_minus:.0e}', color='r', alpha=0.7, fontsize=9)

ax[1, 1].plot(VL0_chi, prof_VL0)
ax[1, 1].plot([VL0_chi[idx_VL0_right], VL0_chi[idx_VL0_right]], [chi2_min - 2, chi2_min + 10], color=line_c, ls='dashed')
ax[1, 1].plot([VL0_chi[idx_VL0_left], VL0_chi[idx_VL0_left]], [chi2_min - 2, chi2_min + 10], color=line_c, ls='dashed')

ax[1, 1].text(VL0_chi_BF, chi2_min + 2, f'{VL0_chi_BF:.4f}', color='k', alpha=0.5, fontsize=9)
ax[1, 1].text(VL0_chi[idx_VL0_right], chi2_min + 4, f'{err_VL0_plus:.2e}', color='b', alpha=0.7, fontsize=9)
ax[1, 1].text(VL0_chi[idx_VL0_left], chi2_min + 4, f'{-err_VL0_minus:.2e}', color='r', alpha=0.7, fontsize=9)

ax[1, 1].set_yticks([chi2_min, chi2_min + 4, chi2_min + 8])

ax[1, 0].set_axis_off()
ax[0, 0].set_ylabel(r'$\delta\;(\mathrm{rad/s})$')
ax[1, 1].set_xlabel(r'$V_{L0}\;(\mathrm{V})$')
ax[0, 0].set_xlim(chi2_min - 2, chi2_min + 10)
ax[1, 1].set_ylim(chi2_min - 2, chi2_min + 10)

plt.show()