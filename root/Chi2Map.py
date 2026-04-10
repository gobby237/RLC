# ============================================================
#  chi2map_vL.py  –  Mappa del chi² per fit multiparametrico V_L(t)
#
#  Formula (tensione ai capi di L):
#    V_L(t) = VL0 * exp(-delta*(t-t0)) *
#             [cos(omega_d*(t-t0)) - (delta/omega_d)*sin(omega_d*(t-t0))] + Voff
#    con  omega_d = sqrt(omega0^2 - delta^2)
#
#  5 parametri: VL0, omega0, delta, Voff, t0
#
#  Il grafico segue ESATTAMENTE lo stile del codice RLC_smorzC_v0.py:
#    mappa 2D con contourf + contour chi2+1 al centro
#    proiezione 1D su X (in basso)
#    proiezione 1D su Y (a sinistra)
#    colorbar a destra
#
#  Per scegliere i due parametri: modifica PARAM_X e PARAM_Y
#    0 → VL0     (V)
#    1 → omega0  (rad/s)
#    2 → delta   (rad/s)
#    3 → Voff    (V)
#    4 → t0      (s)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import mplhep as hep
import warnings
import sys

# ============================================================
#  SEZIONE 1 — SCEGLI QUI I DUE PARAMETRI DA VISUALIZZARE
# ============================================================
PARAM_X = 1   # omega0
PARAM_Y = 2   # delta

# Risoluzione griglia (N_GRID x N_GRID punti)
N_GRID = 80

# Finestra attorno al best-fit (in unità di sigma dal fit)
NSIGMA = 2.0

# Range colorbar: mostra fino a chi2_min + CHI2_DISPLAY_DELTA
CHI2_DISPLAY_DELTA = 12.0

# ============================================================
#  SEZIONE 2 — DATI E PARAMETRI DEL FIT
# ============================================================

# File dati: tre colonne  tempo[ps]  tensione[µV]  errore[µV]
# Tutte le colonne vengono moltiplicate per 1e-6 (ps→s, µV→V)
DATA_FILE = "dati.txt"

# Range temporale del fit (secondi)
T_FIT_MIN = 0.0
T_FIT_MAX = 46.78e-6

# Capacità misurata col METRIX (per calcolo L e R)
C_METRIX  = 4.76e-9   # F
sC_METRIX = 0.02e-9   # F

# Limiti dei parametri (stessi del codice C++)
PAR_BOUNDS = {
    'VL0'   : (0.5,    5.0  ),
    'omega0': (6.0e5,  1.1e6),
    'delta' : (1.0e4,  2.0e5),
    'Voff'  : (-0.3,   0.3  ),
    't0'    : (-2e-6,  2e-6 ),
}
PAR_NAMES  = ['VL0', 'omega0', 'delta', 'Voff', 't0']
PAR_LABELS = [r'$V_{L0}$', r'$\omega_0$', r'$\delta$', r'$V_\mathrm{off}$', r'$t_0$']
PAR_UNITS  = ['V', 'rad/s', 'rad/s', 'V', 's']

# Valori iniziali griglia multi-start
N_OM0   = 5
N_DELTA = 4
VL0_INIT  = 2.3
VOFF_INIT = 0.0
T0_INIT   = 0.0

# ============================================================
#  SEZIONE 3 — FUNZIONE DI FIT  V_L(t)
# ============================================================

def vL_func(t, VL0, omega0, delta, Voff, t0):
    """V_L(t) ai capi dell'induttore in regime sottosmorzato."""
    dt  = t - t0
    od2 = omega0**2 - delta**2
    if od2 <= 0:
        return np.full_like(t, Voff)
    omega_d = np.sqrt(od2)
    return (VL0 * np.exp(-delta * dt)
            * (np.cos(omega_d * dt) - (delta / omega_d) * np.sin(omega_d * dt))
            + Voff)

def vL_func_vec(t, VL0, omega0, delta, Voff, t0):
    """Versione vettorizzata per curve_fit."""
    dt  = t - t0
    od2 = omega0**2 - delta**2
    od2_safe = np.where(od2 > 0, od2, 1.0)
    omega_d  = np.sqrt(od2_safe)
    result = (VL0 * np.exp(-delta * dt)
              * (np.cos(omega_d * dt) - (delta / omega_d) * np.sin(omega_d * dt))
              + Voff)
    return np.where(od2 > 0, result, Voff)

# ============================================================
#  SEZIONE 4 — CALCOLO CHI²
# ============================================================

def compute_chi2(t, V, sV, pars):
    """chi² con errori individuali: sum[(V_i - f(t_i))^2 / sV_i^2]."""
    VL0, omega0, delta, Voff, t0 = pars
    V_pred = vL_func_vec(t, VL0, omega0, delta, Voff, t0)
    return np.sum(((V - V_pred) / sV)**2)

# ============================================================
#  SEZIONE 5 — LETTURA DATI
# ============================================================

print(f"\n  Lettura dati da '{DATA_FILE}'...")
try:
    raw = np.loadtxt(DATA_FILE)
except Exception as e:
    sys.exit(f"ERRORE lettura file: {e}")

if raw.ndim != 2 or raw.shape[1] < 3:
    sys.exit("ERRORE: il file deve avere almeno 3 colonne (t, V, sV).")

# Converte: ps→s, µV→V, µV→V
t_all  = raw[:, 0] * 1e-12  # (ps * 1e-6 = µs; qui raw sono già in µs se segue conv C++)
V_all  = raw[:, 1] * 1e-6
sV_all = raw[:, 2] * 1e-6

# Nota: il codice C++ fa t_raw*1e-12 (ps→s). Se il tuo file ha t in ps usa 1e-12.
# Se ha t nelle unità raw del file precedente (dove raw*1e-6 = s), usa 1e-6.
# CONTROLLA con: print(t_all[:5])  e verifica che siano in secondi (ordine ~µs = 1e-6 s)

# Filtra nel range del fit
mask = (t_all >= T_FIT_MIN) & (t_all <= T_FIT_MAX)
t    = t_all[mask]
V    = V_all[mask]
sV   = sV_all[mask]

# Scarta punti con errore nullo o negativo
valid = sV > 0
t, V, sV = t[valid], V[valid], sV[valid]
N = len(t)

print(f"  Punti nel range [{T_FIT_MIN:.2e}, {T_FIT_MAX:.2e}] s con errore > 0: {N}")
print(f"  sigma_V: min={sV.min():.3g} V  max={sV.max():.3g} V  media={sV.mean():.3g} V\n")

# ============================================================
#  SEZIONE 6 — FIT MULTI-START (stessa logica del C++)
# ============================================================

print("  Fit multi-start in corso...")
bounds_lo = [PAR_BOUNDS[p][0] for p in PAR_NAMES]
bounds_hi = [PAR_BOUNDS[p][1] for p in PAR_NAMES]

best_chi2  = np.inf
best_popt  = None
best_pcov  = None
n_converged = 0

om0_grid   = np.linspace(PAR_BOUNDS['omega0'][0], PAR_BOUNDS['omega0'][1], N_OM0  + 1)[:-1]
om0_grid  += (om0_grid[1] - om0_grid[0]) * 0.5
delta_grid = np.linspace(PAR_BOUNDS['delta'][0],  PAR_BOUNDS['delta'][1],  N_DELTA + 1)[:-1]
delta_grid += (delta_grid[1] - delta_grid[0]) * 0.5

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for om0_try in om0_grid:
        for delta_try in delta_grid:
            if om0_try <= delta_try:
                continue
            p0 = [VL0_INIT, om0_try, delta_try, VOFF_INIT, T0_INIT]
            try:
                popt, pcov = curve_fit(
                    vL_func_vec, t, V,
                    p0=p0,
                    sigma=sV, absolute_sigma=True,
                    bounds=(bounds_lo, bounds_hi),
                    method='trf',
                    maxfev=5000
                )
                chi2_try = compute_chi2(t, V, sV, popt)
                n_converged += 1
                if chi2_try < best_chi2:
                    best_chi2 = chi2_try
                    best_popt = popt.copy()
                    best_pcov = pcov.copy()
            except Exception:
                pass

print(f"  Tentativi convergiti: {n_converged} / {N_OM0 * N_DELTA}")

if best_popt is None:
    sys.exit("ERRORE: nessun fit convergito. Allarga i limiti dei parametri.")

# Fit finale raffinato con i migliori parametri trovati
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        best_popt, best_pcov = curve_fit(
            vL_func_vec, t, V,
            p0=best_popt,
            sigma=sV, absolute_sigma=True,
            bounds=(bounds_lo, bounds_hi),
            method='trf',
            maxfev=10000
        )
    except Exception as e:
        print(f"  Attenzione fit finale: {e}")

best_perr = np.sqrt(np.diag(best_pcov))
best_chi2 = compute_chi2(t, V, sV, best_popt)
ndf       = N - len(best_popt)
chi2_r    = best_chi2 / ndf
prob      = 1.0  # approssimazione; per chi2 esatta usare scipy.stats.chi2.sf

try:
    from scipy.stats import chi2 as chi2dist
    prob = chi2dist.sf(best_chi2, ndf)
except ImportError:
    pass

VL0_fit, omega0_fit, delta_fit, Voff_fit, t0_fit = best_popt
sVL0, somega0, sdelta, sVoff, st0               = best_perr

# ============================================================
#  SEZIONE 7 — GRANDEZZE DERIVATE  (L, R, omega_d, Q)
# ============================================================

od2     = omega0_fit**2 - delta_fit**2
omega_d = np.sqrt(od2) if od2 > 0 else 0.0
T_d     = 2 * np.pi / omega_d if omega_d > 0 else 0.0
Q       = omega0_fit / (2 * delta_fit) if delta_fit > 0 else 0.0

# Propagazione errore omega_d
if omega_d > 0:
    somega_d = np.sqrt((omega0_fit / omega_d * somega0)**2
                       + (delta_fit  / omega_d * sdelta )**2)
else:
    somega_d = 0.0

# L = 1 / (omega0^2 * C)
L        = 1.0 / (omega0_fit**2 * C_METRIX)
sL_over_L = np.sqrt((2 * somega0 / omega0_fit)**2 + (sC_METRIX / C_METRIX)**2)
sL       = L * sL_over_L

# R = 2 * L * delta
R  = 2.0 * L * delta_fit
sR = R * np.sqrt(sL_over_L**2 + (sdelta / delta_fit)**2)

# ============================================================
#  SEZIONE 8 — STAMPA RISULTATI
# ============================================================

print("\n========================================")
print("  RISULTATI FIT MULTIPARAMETRICO V_L(t)")
print("  (fit pesato con errori individuali)")
print("========================================")
print(f"  chi2/ndf = {best_chi2:.1f} / {ndf} = {chi2_r:.3f}")
print(f"  Prob     = {prob:.6f}\n")
print(f"  VL0    = ({VL0_fit:.4f} +/- {sVL0:.4f}) V")
print(f"  omega0 = ({omega0_fit:.4e} +/- {somega0:.2e}) rad/s")
print(f"  delta  = ({delta_fit:.4e} +/- {sdelta:.2e}) rad/s")
print(f"  Voff   = ({Voff_fit:.4f} +/- {sVoff:.4f}) V")
print(f"  t0     = ({t0_fit:.4e} +/- {st0:.2e}) s")
print(f"  omega_d = ({omega_d:.4e} +/- {somega_d:.2e}) rad/s")
print(f"  T_d    = {T_d:.4e} s")
print(f"  Q      = {Q:.3f}\n")
print(f"  L = ({L:.4e} +/- {sL:.2e}) H")
print(f"  R = ({R:.4f} +/- {sR:.4f}) Ohm\n")

# ============================================================
#  SEZIONE 9 — GRIGLIA CHI² 2D
# ============================================================

cx = best_popt[PARAM_X]
cy = best_popt[PARAM_Y]
sx = best_perr[PARAM_X]
sy = best_perr[PARAM_Y]

xlo = cx - NSIGMA * sx
xhi = cx + NSIGMA * sx
ylo = cy - NSIGMA * sy
yhi = cy + NSIGMA * sy

print(f"  Mappa chi2: {PAR_NAMES[PARAM_X]} (X) vs {PAR_NAMES[PARAM_Y]} (Y)")
print(f"  Griglia {N_GRID}x{N_GRID}, finestra ±{NSIGMA} sigma\n")

x_grid = np.linspace(xlo, xhi, N_GRID)
y_grid = np.linspace(ylo, yhi, N_GRID)
XX, YY = np.meshgrid(x_grid, y_grid)  # shape (N_GRID, N_GRID)

chi2_map = np.zeros_like(XX)
pars_scan = best_popt.copy()

print("  Calcolo mappa chi2...")
for iy in range(N_GRID):
    for ix in range(N_GRID):
        pars_scan[PARAM_X] = XX[iy, ix]
        pars_scan[PARAM_Y] = YY[iy, ix]
        chi2_map[iy, ix]   = compute_chi2(t, V, sV, pars_scan)

chi2_min_map = chi2_map.min()
print(f"  chi2 mappa: min={chi2_min_map:.2f}  max={chi2_map.max():.2f}\n")

# ============================================================
#  SEZIONE 10 — PROIEZIONI 1D (fissando l'altro parametro al BF)
# ============================================================

NP = N_GRID * 5

x_prof = np.linspace(xlo, xhi, NP)
chi2_x = np.zeros(NP)
pars_scan = best_popt.copy()
for k in range(NP):
    pars_scan[PARAM_X] = x_prof[k]
    pars_scan[PARAM_Y] = cy
    chi2_x[k] = compute_chi2(t, V, sV, pars_scan)

y_prof = np.linspace(ylo, yhi, NP)
chi2_y = np.zeros(NP)
pars_scan = best_popt.copy()
for k in range(NP):
    pars_scan[PARAM_X] = cx
    pars_scan[PARAM_Y] = y_prof[k]
    chi2_y[k] = compute_chi2(t, V, sV, pars_scan)

# ============================================================
#  SEZIONE 11 — CONTORNO chi2_min + 1  (bisection su N_ANGLE raggi)
# ============================================================

N_ANGLE  = 720
chi2_lev = chi2_min_map + 1.0
xc_pts, yc_pts = [], []

pars_scan = best_popt.copy()
for ia in range(N_ANGLE):
    angle = 2 * np.pi * ia / N_ANGLE
    dxd, dyd = np.cos(angle), np.sin(angle)
    r_lo = 0.0
    r_hi = NSIGMA * np.sqrt((sx * dxd)**2 + (sy * dyd)**2) * 1.5

    pars_scan[PARAM_X] = cx + r_hi * dxd
    pars_scan[PARAM_Y] = cy + r_hi * dyd
    if compute_chi2(t, V, sV, pars_scan) < chi2_lev:
        continue  # il contorno non è raggiunto in questa direzione

    for _ in range(60):
        r_mid = 0.5 * (r_lo + r_hi)
        pars_scan[PARAM_X] = cx + r_mid * dxd
        pars_scan[PARAM_Y] = cy + r_mid * dyd
        if compute_chi2(t, V, sV, pars_scan) < chi2_lev:
            r_lo = r_mid
        else:
            r_hi = r_mid
        if (r_hi - r_lo) < 1e-12 * (abs(cx) + abs(cy) + 1e-30):
            break

    rb = 0.5 * (r_lo + r_hi)
    xc_pts.append(cx + rb * dxd)
    yc_pts.append(cy + rb * dyd)

if xc_pts:
    xc_pts.append(xc_pts[0])
    yc_pts.append(yc_pts[0])
    xc_min, xc_max = min(xc_pts), max(xc_pts)
    yc_min, yc_max = min(yc_pts), max(yc_pts)
else:
    xc_min, xc_max = cx - sx, cx + sx
    yc_min, yc_max = cy - sy, cy + sy

print(f"  Contorno chi2_min+1:")
print(f"    {PAR_NAMES[PARAM_X]}: [{xc_min:.4g}, {xc_max:.4g}]  sigma={0.5*(xc_max-xc_min):.3g} {PAR_UNITS[PARAM_X]}")
print(f"    {PAR_NAMES[PARAM_Y]}: [{yc_min:.4g}, {yc_max:.4g}]  sigma={0.5*(yc_max-yc_min):.3g} {PAR_UNITS[PARAM_Y]}\n")

# ============================================================
#  SEZIONE 12 — GRAFICO  (stile RLC_smorzC_v0.py)
# ============================================================

plt.style.use(hep.style.ROOT)
plt.rcParams.update({
    'text.usetex': True,
    'axes.formatter.min_exponent': 2,
    'figure.constrained_layout.use': True,
})

# Palette tenue giallo→arancio→viola  (stessa del codice C++)
cmap_colors = [
    (0.98, 0.95, 0.72),
    (0.94, 0.78, 0.44),
    (0.82, 0.55, 0.38),
    (0.55, 0.25, 0.50),
    (0.24, 0.03, 0.49),
]
cmap_muted = mcolors.LinearSegmentedColormap.from_list("muted_chi2", cmap_colors, N=256)

# Clip della mappa al display range
chi2_clipped  = np.clip(chi2_map, chi2_min_map, chi2_min_map + CHI2_DISPLAY_DELTA)
display_range = np.linspace(chi2_min_map, chi2_min_map + CHI2_DISPLAY_DELTA, 256)

# Range proiezioni con margini coerenti
chi2x_rng = chi2_x.max() - chi2_x.min()
chi2y_rng = chi2_y.max() - chi2_y.min()
yX_lo = chi2_x.min() - 0.12 * chi2x_rng
yX_hi = max(chi2_x.max() + 0.25 * chi2x_rng,
            chi2_min_map + 1.0 + 0.15 * chi2x_rng)
xY_lo = chi2_y.min() - 0.12 * chi2y_rng
xY_hi = max(chi2_y.max() + 0.25 * chi2y_rng,
            chi2_min_map + 1.0 + 0.15 * chi2y_rng)

dash_col = 'gray'
dash_kw  = dict(color=dash_col, linestyle='dashed', linewidth=1.0)

fig, ax = plt.subplots(
    2, 2,
    figsize=(6.5, 6.0),
    height_ratios=[3, 1],
    width_ratios=[1, 3],
    sharex='col',
    sharey='row',
    constrained_layout=True
)
fig.suptitle(
    rf'$\chi^2\left({PAR_LABELS[PARAM_X]},\, {PAR_LABELS[PARAM_Y]}\right)$',
    fontsize=13
)

# ---- Mappa 2D  (ax[0,1]) ----
level_cont = np.linspace(chi2_min_map, chi2_min_map + CHI2_DISPLAY_DELTA, 200)
im = ax[0, 1].contourf(x_grid, y_grid, chi2_clipped,
                        levels=level_cont, cmap=cmap_muted)

# Contorno chi2_min+1
if xc_pts:
    ax[0, 1].plot(xc_pts, yc_pts, 'k--', linewidth=2, label=r'$\chi^2_\mathrm{min}+1$')

# Linee tratteggiate tangenti al contorno (attraversano tutto il pad)
ax[0, 1].axvline(xc_min, **dash_kw)
ax[0, 1].axvline(xc_max, **dash_kw)
ax[0, 1].axhline(yc_min, **dash_kw)
ax[0, 1].axhline(yc_max, **dash_kw)
# Linee centrali (minimo)
ax[0, 1].axvline(cx, color='lightgray', linestyle=':', linewidth=0.8)
ax[0, 1].axhline(cy, color='lightgray', linestyle=':', linewidth=0.8)

# Punto del minimo
ax[0, 1].plot(cx, cy, 'kx', markersize=8, markeredgewidth=2)

# Label chi2_min
ax[0, 1].text(cx, cy - 0.35*sy, f'{chi2_min_map:.0f}',
              ha='center', va='top', fontsize=8, color=dash_col)
# Label chi2_min+1 (fuori dal contorno)
xl = cx + (xc_max - cx) * 1.15
yl = cy + (yc_max - cy) * 1.15
xl = min(xl, xhi - 0.08*(xhi - xlo))
yl = min(yl, yhi - 0.08*(yhi - ylo))
ax[0, 1].text(xl, yl, f'{chi2_min_map+1:.0f}',
              ha='left', va='bottom', fontsize=8, color=dash_col)

# Colorbar a destra, più corta della mappa
cbar = fig.colorbar(im, ax=ax[0, 1], shrink=0.80, pad=0.02)
cbar.set_label(r'$\chi^2$', rotation=0, labelpad=8, fontsize=11)
cbar.ax.tick_params(labelsize=9)
# Linea sul livello chi2_min+1
cbar.ax.axhline(chi2_min_map + 1.0, color='black', linestyle='--', linewidth=1.5)

ax[0, 1].set_xlim(xlo, xhi)
ax[0, 1].set_ylim(ylo, yhi)
ax[0, 1].tick_params(labelbottom=False, labelleft=False)

# ---- Proiezione X  (ax[1,1]) — sotto la mappa ----
ax[1, 1].plot(x_prof, chi2_x, color='steelblue', linewidth=2)
ax[1, 1].axhline(chi2_min_map + 1.0, color='red', linestyle='--', linewidth=1.5)
ax[1, 1].axvline(xc_min, **dash_kw)
ax[1, 1].axvline(xc_max, **dash_kw)
ax[1, 1].set_xlim(xlo, xhi)
ax[1, 1].set_ylim(yX_lo, yX_hi)
ax[1, 1].set_xlabel(f'{PAR_LABELS[PARAM_X]} [{PAR_UNITS[PARAM_X]}]', fontsize=10)
ax[1, 1].tick_params(labelleft=False)

# Etichette ±sigma
ylab = chi2_x.min() + 0.30 * chi2x_rng
ax[1, 1].text(xc_min, ylab, f'$-{sx:.3g}$', ha='center', va='bottom',
              fontsize=8, color='red')
ax[1, 1].text(xc_max, ylab, f'$+{sx:.3g}$', ha='center', va='bottom',
              fontsize=8, color='red')
# Valore best-fit
ax[1, 1].text(cx, chi2_x.min() + 0.88*chi2x_rng,
              f'{cx:.4g}', ha='center', va='bottom', fontsize=7.5, color='dimgray')

# ---- Proiezione Y  (ax[0,0]) — a sinistra della mappa ----
ax[0, 0].plot(chi2_y, y_prof, color='steelblue', linewidth=2)
ax[0, 0].axvline(chi2_min_map + 1.0, color='red', linestyle='--', linewidth=1.5)
ax[0, 0].axhline(yc_min, **dash_kw)
ax[0, 0].axhline(yc_max, **dash_kw)
ax[0, 0].set_xlim(xY_lo, xY_hi)
ax[0, 0].set_ylim(ylo, yhi)
ax[0, 0].set_ylabel(f'{PAR_LABELS[PARAM_Y]} [{PAR_UNITS[PARAM_Y]}]', fontsize=10)
ax[0, 0].tick_params(labelbottom=False)

# Etichette ±sigma
xlab = chi2_y.min() + 0.08 * chi2y_rng
ax[0, 0].text(xlab, yc_min, f'$-{sy:.3g}$', ha='left', va='center',
              fontsize=8, color='red')
ax[0, 0].text(xlab, yc_max, f'$+{sy:.3g}$', ha='left', va='center',
              fontsize=8, color='red')
# Valore best-fit
ax[0, 0].text(chi2_y.min() + 0.50*chi2y_rng, cy,
              f'{cy:.4g}', ha='left', va='center', fontsize=7.5, color='dimgray')

# ---- Cella in basso a sinistra: off ----
ax[1, 0].set_axis_off()

# Tick label formato scientifico ove necessario
from matplotlib.ticker import ScalarFormatter
for axi in [ax[1, 1], ax[0, 0]]:
    for axis in [axi.xaxis, axi.yaxis]:
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((-2, 2))
        axis.set_major_formatter(fmt)

plt.savefig('chi2map_vL.png', dpi=150, bbox_inches='tight')
print("  Grafico salvato: chi2map_vL.png")
plt.show()