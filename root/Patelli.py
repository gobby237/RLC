import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
import mplhep as hep
from cycler import cycler
import matplotlib.colors as colors
import multiprocessing.pool


# settaggio globale grafici    
#print(plt.style.available)
#plt.style.use('classic')
plt.style.use(hep.style.ROOT)
params = {'legend.fontsize': '10',
         'legend.loc': 'upper right',
          'legend.frameon':       'True',
          'legend.framealpha':    '0.8',      # legend patch transparency
          'legend.facecolor':     'w', # inherit from axes.facecolor; or color spec
          'legend.edgecolor':     'w',      # background patch boundary color
          'figure.figsize': (6, 4),
         'axes.labelsize': '10',
         'figure.titlesize' : '14',
         'axes.titlesize':'12',
         'xtick.labelsize':'10',
         'ytick.labelsize':'10',
         'lines.linewidth': '1',
         'text.usetex': False,
#         'axes.formatter.limits': '-5, -3',
         'axes.formatter.min_exponent': '2',
#         'axes.prop_cycle': cycler('color', 'bgrcmyk')
         'figure.subplot.left':'0.125',
         'figure.subplot.bottom':'0.125',
         'figure.subplot.right':'0.925',
         'figure.subplot.top':'0.925',
         'figure.subplot.wspace':'0.1',
         'figure.subplot.hspace':'0.1',
         'figure.constrained_layout.use' : True
          }
plt.rcParams.update(params)
plt.rcParams['axes.prop_cycle'] = cycler(color=['b','g','r','c','m','y','k'])

# Enable debug mode
DEB = False


# Function definition
"""
Si definiscono le funzioni da richiamare nel codice. 
Oltre a renderlo piu' leggibile permette di mappare le funzioni su matrici e quindi su cicli 'for'
 sugli indici su diversi core del computer e quindi di parallelizzare il calcolo.
Un ciclo for esplicito su indici di una matrice viene svolto su un singolo core, un indice alla volta. 
Con i comandi pool di mappatura i singoli indici vengono suddivisi sui diversi core. 
"""

# funzione del fit non lineare dell'oscillazione smorzata ai capi di C. Sono stati aggiunti un termine t0 che tiene conto di errori nell'individuazione dell'inizio dell'oscillazione e un termine v0 che tiene conto di uno shift della curva in voltaggio.
# A = v0, B = Omega, C = tau
def fitf(t, A, B, C, v0, t0):
    x = t-t0
    Omega = np.sqrt(B**2-1/C**2)  # 
    fitval = A/Omega*np.exp(-x/C)*(1/C*np.sin(Omega*x)+Omega*np.cos(Omega*x))+v0
    return fitval

# Stessa funzione di fitf, ma senza to e v0, solo i parametri dell'oscillazione smorzata
def fitf2(t, A, B, C):
    x = t
    Omega = np.sqrt(B**2-1/C**2)  # 
    fitval = A/Omega*np.exp(-x/C)*(1/C*np.sin(Omega*x)+Omega*np.cos(Omega*x))
    return fitval

# Funzione per calcolare il chi2 sugli inidici i,j,k dei vettori rispettivamente A_chi, B_chi, C_chi
def fitchi2(i,j,k):
    x = tempo
    y = Vout
    y_err = eVout
    AA,BB,CC = A_chi[i],B_chi[j],C_chi[k]
    omega = 2.0 * np.pi * x * 1e3  # input in kHz angolari
    residuals = (y - fitf2(tempo,AA,BB,CC))
    chi2 = np.sum((residuals / y_err)**2)
    mappa[i,j,k] = chi2

# Funzione per effettuare la profilazione su due assi di una matrice 3D escludendo il terzo (1,2 o 3).
# la funzione per ogni indice della matrice 2D inserisce il minimo del valore sul terzo asse.
def profi2D(axis,matrix3D):
    if axis == 1 :
        mappa2D = np.array([[np.min(mappa[:,b,c]) for b in range(step)] for c in range(step)])
    if axis == 2 :
        mappa2D = np.array([[np.min(mappa[a,:,c]) for a in range(step)] for c in range(step)])
    if axis == 3 :
        mappa2D = np.array([[np.min(mappa[a,b,:]) for a in range(step)] for b in range(step)])
    return mappa2D

# Funzione per effettuare la profilazione su un asse (1,2 o 3) di una matrice 3D.
# la funzione profila prima in 2D e poi in 1D sull'asse selezionato.
def profi1D(axis, mappa):
    if 1 in axis :
        mappa2D = np.array([[np.min(mappa[:,b,c]) for b in range(step)] for c in range(step)])
#        print('1')
        if 2 in axis:
            mappa1D = np.array([np.min(mappa2D[b,:]) for b in range(step)])
#            print('2')
        if 3 in axis:
            mappa1D = np.array([np.min(mappa2D[:,c]) for c in range(step)])
#            print('3')
    else :
#        print('2-3')
        mappa2D = np.array([[np.min(mappa[a,:,c]) for a in range(step)] for c in range(step)])
        mappa1D = np.array([np.min(mappa2D[a,:]) for a in range(step)])
    return mappa1D



## PARAMATERS

# Input file name
file = 'dati_patelli'
inputname = file+'.txt'


# Initial parameter values
Ainit= 2.5 # voltaggio (V)
Binit = 833798.  # frequenza angolare (Hz)
Cinit = 9e-6 # tempo caratteristico di decadimento tau (s)
v0init = 0.020 # voltaggio di 0 dell'oscillazione (V)
t0init = 0e-6 # shift temporale dell'inizio dell'oscillazione (s)


# Assumed reading errors
letturaV = 0.05*0.41  # errore di lettura dell'oscilloscopio nel voltaggio 1/10 div * distr.triangolare
errscalaV = 0.03*0.41 #  errore percentuale dell'oscilloscopio nel voltaggio 3% * distr.triangolare
letturaT = 0.1e-6*0.41  # errore di lettura dell'oscilloscopio nel tempo 1/10 div * distr.triangolare


#### LOAD DATA
    

# Read data from the input file
data = np.loadtxt(inputname).T # carica i dati per riga e poi traspone per avere le colonne
tempo = data[0,data[0]>0] # prima colonna e' il tempo. Ne prende solo i valori per t>0 assumendo che l'oscillazione inizi in t=0. Cosa che accade se il trigger e' posizionato sul segnale in ingresso in discesa.
Vout = data[1,data[0]>0] # seconda colonna sono le tensioni. Ne prende solo i valori per cui il tempo t>0. 

# Number of points to fit
# Numero dei dati a disposizione.
n = len(tempo)

# Calculate errors on x and y
eVout = np.sqrt((letturaV)**2 + (errscalaV * Vout)**2)
etempo = letturaT

# Grafico dei dati cosi' come caricati  (Vout vs. tempo)
fig, ax = plt.subplots(1, 1, figsize=(5, 4),constrained_layout = True)
ax.errorbar(tempo,Vout,yerr=eVout,xerr=etempo, fmt='o', label=r'$V_{out}$',ms=2)
ax.legend(prop={'size': 10}, loc='best')
ax.set_ylabel(r'Voltaggio (V)')
ax.set_xlim(bottom=0)          # <-- aggiungi questa linea

plt.savefig(file+'_1'+'.png',
            bbox_inches ="tight",
            pad_inches = 1,
            transparent = True,
            facecolor ="w",
            edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

plt.show()


# Perform the first fit to define t0
"""
Si procede ad effettuare un primo fit utilizzando la funzione fitf che contiene offset sia nelle ascisse che nelle ordinate.
Questo per valutarne l'eventuale compatibilita' con lo zero e in ogni caso per procedere eventualmente a uno shift nei due assi.
In questo modo riduco il sistema a i soli 3 parametri relativi all'oscillazione. 

Per le regressioni utilizziamo un algoritmo ai minimi quadrati in modo da avere piena corrispondenza con l'analisi esplicita che faremo del minimo del chi2.
"""

popt, pcov = curve_fit(fitf, tempo, Vout, p0=[Ainit, Binit, Cinit, v0init, t0init], method='lm', sigma=eVout, absolute_sigma=True)

"""
POPT: Vettore con la stima dei parametri dal fit
PCOV: Matrice delle covarianze
method: 'lm' e' i minimi quadrati, quindi il minimo del chi2
"""

perr = np.sqrt(np.diag(pcov))
residuA = Vout - fitf(tempo, *popt)



# Grafico di questa prima regressione che tiene conto degli shift nel tempo e nel voltaggio.

x_fit = np.linspace(min(tempo), max(tempo), 1000) # genero i punti su cui calcolare la funzione ottenuta con la regressione (1000 pti)

fig, ax = plt.subplots(2, 1, figsize=(5, 4),sharex=True, constrained_layout = True, height_ratios=[2, 1])
ax[0].plot(x_fit, fitf(x_fit, *popt), label='Fit', linestyle='--', color='black')
ax[0].plot(x_fit,fitf(x_fit,Ainit,Binit,Cinit,v0init,t0init), label='init guess', linestyle='dashed', color='green')
ax[0].errorbar(tempo,Vout,yerr=eVout,xerr=etempo, fmt='o', label=r'$V_{out}$',ms=2,color='red')
ax[0].legend(loc='upper left')
ax[0].set_ylabel(r'Tensione ai capi di R (V)')
#ax[0].set_xticks([20,30,40,50])

ax[1].errorbar(tempo,residuA,yerr=eVout, fmt='o', label=r'Residui$',ms=2,color='red')
ax[1].set_ylabel(r'Residui (V)')
ax[1].set_xlabel(r'tempo (s)',loc='center')
ax[1].plot(tempo,np.zeros(len(tempo)))

plt.savefig(file+'_2'+'.png',
            bbox_inches ="tight",
            pad_inches = 1,
            transparent = True,
            facecolor ="w",
            edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

plt.show()


# Extract and print best fit (BF) parameters and errors
A_BF, B_BF, C_BF, v0_BF, t0_BF = popt #parametri del best fit
eA_BF, eB_BF, eC_BF, ev0_BF, et0_BF = np.sqrt(np.diag(pcov)) #errori sui parametri BF

print("============== BEST FIT with SciPy ====================")
print(r'A = ({a:.3e} +/- {b:.1e})'.format(a=A_BF,b=eA_BF))
print(r'B = ({c:.5e} +/- {d:.1e}) kHz'.format(c=B_BF * 1e-3, d=eB_BF * 1e-3))
print(r'C = ({e:.3e} +/- {f:.1e}) ms'.format(e=C_BF * 1e3, f=eC_BF * 1e3))
print(r'v0 = ({g:.4e} +/- {h:.1e}) mV'.format(g=v0_BF * 1e3, h=ev0_BF * 1e3))
print(r't0 = ({i:.4e} +/- {l:.1e}) ms'.format(i=t0_BF * 1e3, l=et0_BF * 1e3))
print("=======================================================")

"""
Come si puo' vedere t0 risulta compatibile con la 0 e piccolo rispetto al periodo dell'oscillazione, per cui possiamo correggere.
Il v0 anche lui e' quasi compatibile e inoltre ordini di grandezza inferiore rispetto all'ampiezza delle oscillazioni. Anche qui si puo' correggere.
Le due correzioni non influenzeranno la stima di RLC, sono eventualmente errori sistematici di cui discutere eventualmente la causa.
Il tempo e' legato alla banda del generatore di funzioni e dell'oscilloscopio, per cui la transizione tra 8V e 0V ha un tempo finito.
L'errore sul v0 e' un errore di ground dell'oscilloscopio, perfettamente compatibile con l'errore dato.
"""

# Perform the fit to masked from t0
"""
Procediamo quindi allo shift rigido della curva e alla sua regressione sono in A,B e C.
"""

# shift rigido della curva di v0 e t0
shift= 0 # 2e-6
Vout = Vout[tempo>t0_BF+shift]-v0_BF #Vout
eVout = eVout[tempo>t0_BF+shift]
tempo = tempo[tempo>t0_BF+shift]-t0_BF
N = len(tempo)

# Nuova regressione utilizzando fitf2 quindi sono nei parametri A,B e C, sempre con i minimi quadrati
p0_secondo_fit = [A_BF, B_BF, C_BF]
popt, pcov = curve_fit(
    fitf2, tempo, Vout,
    p0=p0_secondo_fit,
    method='lm',
    sigma=eVout,
    absolute_sigma=True,
    maxfev=20000
)

# variables error and chi2
perr = np.sqrt(np.diag(pcov))
residuA = Vout - fitf2(tempo, *popt)
chisq = np.sum((residuA/eVout)**2)
df = N - 3
chisq_rid = chisq/df
print(chisq_rid)

# Grafico di questa regressione in A, B e C
fig, ax = plt.subplots(2, 1, figsize=(5, 7),sharex=True, constrained_layout = True, height_ratios=[2, 1])
ax[0].plot(x_fit, fitf2(x_fit, *popt), label='Fit', linestyle='--', color='blue')
ax[0].plot(x_fit,fitf2(x_fit,Ainit,Binit,Cinit), label='init guess', linestyle='dashed', color='green')
ax[0].errorbar(tempo,Vout,yerr=eVout,xerr=etempo, fmt='o', label=r'$V_{out}$',ms=2,color='red')
ax[0].legend(loc='upper left')
ax[0].set_ylabel(r'Tensione ai capi di R')
#ax[0].set_xticks([20,30,40,50])

ax[1].errorbar(tempo,residuA,yerr=eVout, fmt='o', label=r'Residui$',ms=2,color='red')
ax[1].set_ylabel(r'Residui (V)')
ax[1].set_xlabel(r'tempo (s)',loc='center')
ax[1].plot(tempo,np.zeros(len(tempo)))

plt.savefig(file+'_2'+'.png',
            bbox_inches ="tight",
            pad_inches = 1,
            transparent = True,
            facecolor ="w",
            edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

plt.show()


# Extract and print best fit (BF) parameters and errors
A_BF, B_BF, C_BF = popt #parametri del best fit
eA_BF, eB_BF, eC_BF = np.sqrt(np.diag(pcov)) # errori del BF

print("============== BEST FIT with SciPy ====================")
print(r'A = ({a:.3e} +/- {b:.1e})'.format(a=A_BF,b=eA_BF))
print(r'B = ({c:.5e} +/- {d:.1e}) kHz'.format(c=B_BF * 1e-3, d=eB_BF * 1e-3))
print(r'C = ({e:.3e} +/- {f:.1e}) ms'.format(e=C_BF * 1e3, f=eC_BF * 1e3))
print(r'chisq = {m:.2f}'.format(m=chisq))
print("=======================================================")

"""
Ora che abbiamo effettuato la regressione ai minimi quadrati utilizzando una libreria di Python (Scipy), 
proviamo ad ottenere lo stesso risultato calcolando a mano dalla curva del chi2 i parametri piu' probabili
ed il loro errore.
"""

"""
Per prima cosa definiremo una matrice di punti A,B e C su cui calcolare il chi2. 
Per farlo la centriamo sul valore BF che abbiamo trovato con scipy e ci allarghiamo di un 2sigma
su ogni parametro (sempre considerando l'errore di scipy)
"""

# Define the interval for parameter limits
NSI = 2 # numero di sigma rispetto all'errore di scipy
A0, A1 = A_BF - NSI * eA_BF, A_BF + NSI * eA_BF # estremi di scansione del parametro A
B0, B1 = B_BF - NSI * eB_BF, B_BF + NSI * eB_BF # estremi di scansione del parametro B
C0, C1 = C_BF - NSI * eC_BF, C_BF + NSI * eC_BF # estremi di scansione del parametro C

"""
print(f"(A0 = {A0}, A1 = {A1}) V")
print(f"(B0 = {B0}, B1 = {B1}) Hz")
print(f"(C0 = {C0}, C1 = {C1}) s")
"""


# Calcolo mappa Chi2 3D

step = 100 # discretizzazione all'interno dell'intervallo di scansione

# array dei diversi parametri
A_chi = np.linspace(A0,A1,step)
B_chi = np.linspace(B0,B1,step)
C_chi = np.linspace(C0,C1,step)

# inizializzo la matrice 3D del chi2
mappa = np.zeros((step,step,step))
# creo una lista degli indici da mappare con il pool su piu processori
item = [(i,j,k) for i in range(step) for j in range(step) for k in range(step)]

# assign the global pool
# inizializzo il pool per mandare su piu processori (fino a 100 processi in parallelo)
pool = multiprocessing.pool.ThreadPool(100)
# issue top level tasks to pool and wait
# mappo la lista 'item' di indici nella funzione fitchi2 che si rifa' agli array dei parametri definiti
#e in uscita salva il valore del chi2 nella posizione corretta della matrice 3D. 
pool.starmap(fitchi2, item, chunksize=10)
# close the pool
pool.close()

# alla fine si ottiene la mappa 3D del chi2
mappa = np.asarray(mappa)            

# trovo il minimo del chi2 e la sua posizione nella matrice 3D
chi2_min = np.min(mappa)
argchi2_min = np.unravel_index(np.argmin(mappa),mappa.shape)

# calcolo i residui della regressione utilizzando i valori dei parametri del minimo del chi2
# ricontrollo che il minimo del chi2 sia coerente.
residui_chi2 = Vout - fitf2(tempo,A_chi[argchi2_min[0]],B_chi[argchi2_min[1]],C_chi[argchi2_min[2]])
chisq_res = np.sum((residui_chi2/eVout)**2)

print(chi2_min,argchi2_min, chisq_res)

# Grafico nuovamente la regressione e i residui, questa volta ottenuti calcolando a mano il minimo del chi2
fig, ax = plt.subplots(2, 1, figsize=(5, 7),sharex=True, constrained_layout = True, height_ratios=[2, 1])
ax[0].plot(x_fit, fitf2(x_fit, A_chi[argchi2_min[0]],B_chi[argchi2_min[1]],C_chi[argchi2_min[2]]), label='Fit', linestyle='--', color='blue')
#ax[0].plot(x_fit,fitf2(x_fit,Ainit,Binit,Cinit), label='init guess', linestyle='dashed', color='green')
ax[0].errorbar(tempo,Vout,yerr=eVout,xerr=etempo, fmt='o', label=r'$V_{out}$',ms=2,color='red')
ax[0].legend(loc='upper left')
ax[0].set_ylabel(r'Tensione ai capi di R')
#ax[0].set_xticks([20,30,40,50])

ax[1].errorbar(tempo,residuA,yerr=eVout, fmt='o', label=r'Residui$',ms=2,color='red')
ax[1].set_ylabel(r'Residui (V)')
ax[1].set_xlabel(r'tempo (s)', loc='center')
ax[1].plot(tempo,np.zeros(len(tempo)))

plt.savefig(file+'_3'+'.png',
            bbox_inches ="tight",
            pad_inches = 1,
            transparent = True,
            facecolor ="w",
            edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

plt.show()

"""
A questo punto devo calcolare l'errore sui singoli parametri con il chi2+1, 
e profilare il chi2 sui diversi parametri. Utilizzo le funzioni sopra definite.
"""

#Calcolo Profilazione 2D
chi2D = profi2D(1,mappa)
Achi2D = profi2D(2,mappa) 

#Calcolo Profilazioni 1D
prof_B = profi1D([1,3],mappa)
prof_C = profi1D([1,2],mappa)   
prof_A = profi1D([2,3],mappa)    

"""
Per trovare l'errore sui parametri devo trovare i valori a chi2+1
Per farlo sottraiamo alle profilazioni chi2+1 e ne facciamo il valore assoluto
In questo modo i dati saranno tutti positivi ed avranno un minimo a chi2+1
"""

lvl = chi2_min+1. # 2.3 (2parametri) # 3.8 (3parametri)
diff_B = abs(prof_B-lvl)
diff_C = abs(prof_C-lvl)
diff_A = abs(prof_A-lvl)

B_dx = np.argmin(diff_B[B_chi<B_BF]) # minimo di B per valori inferiori al BF
B_sx = np.argmin(diff_B[B_chi>B_BF])+len(diff_B[B_chi<B_BF]) # minimo di B per valori superiori al BF
C_dx = np.argmin(diff_C[C_chi<C_BF])
C_sx = np.argmin(diff_C[C_chi>C_BF])+len(diff_C[C_chi<C_BF])
A_dx = np.argmin(diff_A[A_chi<A_BF])
A_sx = np.argmin(diff_A[A_chi>A_BF])+len(diff_A[A_chi<A_BF])
#print(B_dx,B_sx,C_dx,C_sx,A_dx,A_sx)

# Facendo la differenza rispetto al BF ottengo gli errori a dx e a sx del BF
errA = A_chi[argchi2_min[0]]-A_chi[A_dx]
errAA = A_chi[A_sx]-A_chi[argchi2_min[0]]
errB = B_chi[argchi2_min[1]]-B_chi[B_dx] 
errBB = B_chi[B_sx] -B_chi[argchi2_min[1]]
errC = C_chi[argchi2_min[2]]-C_chi[C_dx]
errCC = C_chi[C_sx]-C_chi[argchi2_min[2]]


print("============== BEST FIT with chi2 ====================")
print(r'A = ({a:.3e} - {b:.1e} + {c:.1e})'.format(a=A_chi[argchi2_min[0]],b=errA,c=errAA))
print(r'B = ({d:.5e} - {e:.1e} + {f:.1e}) kHz'.format(d=B_chi[argchi2_min[1]] * 1e-3, e=errB * 1e-3, f=errBB* 1e-3))
print(r'C = ({g:.3e} - {h:.1e} + {n:.1e}) ms'.format(g=C_chi[argchi2_min[2]] * 1e3, h=errC * 1e3,  n=errCC * 1e3))
print(r'chisq = {m:.2f}'.format(m=np.min(mappa)))
print("=======================================================")


"""
Adesso faccio il grafico delle profilazioni 2D e 1D dei diversi parametri
"""

# definisco la mappa colore
cmap = mpl.colormaps['plasma'].reversed()
level = np.linspace(np.min(chi2D),np.max(chi2D),100)
line_c = 'gray'


# Profilazione di Omega e Tau
fig, ax = plt.subplots(2, 2, figsize=(5.5, 5),constrained_layout = True, height_ratios=[3, 1], width_ratios=[1,3], sharex='col', sharey='row')
fig.suptitle(r'$\chi^2 \left(\Omega, \tau \right)$')
im = ax[0,1].contourf(B_chi,C_chi,chi2D, levels=level, cmap=cmap) 
cbar = fig.colorbar(im, extend='both', shrink=0.9, ax=ax[0,1], ticks = [int(chi2_min),int(chi2_min+2),int(chi2_min+4),int(chi2_min+6)]) 
cbar.set_label(r'$\chi^2$',rotation=360)

CS = ax[0,1].contour(B_chi,C_chi,chi2D, levels=[chi2_min+0.0001,chi2_min+1,chi2_min+2.3,chi2_min+3.8],linewidths=1, colors='k',alpha=0.5,linestyles='dotted')
ax[0,1].clabel(CS, inline=True, fontsize=9, fmt='%.0f')
ax[0,1].text(B_chi[np.argmin(prof_B)],C_chi[np.argmin(prof_C)],r'{g:.0f}'.format(g=chi2_min), color='k',alpha=0.5, fontsize=9)
ax[0,1].plot([B0,B1],[C_chi[C_sx],C_chi[C_sx]],color=line_c, ls='dashed')
ax[0,1].plot([B0,B1],[C_chi[C_dx],C_chi[C_dx]],color=line_c, ls='dashed')
ax[0,1].plot([B_chi[B_sx],B_chi[B_sx]],[C0,C1],color=line_c, ls='dashed')
ax[0,1].plot([B_chi[B_dx],B_chi[B_dx]],[C0,C1],color=line_c, ls='dashed')

ax[0,0].plot(prof_B,C_chi,ls='-') 
ax[0,0].plot([int(chi2_min-2),int(chi2_min+10)],[C_chi[C_sx],C_chi[C_sx]], color=line_c, ls='dashed') 
ax[0,0].plot([int(chi2_min-2),int(chi2_min+10)],[C_chi[C_dx],C_chi[C_dx]], color=line_c, ls='dashed')

ax[0,0].set_xticks([int(chi2_min),int(chi2_min+2),int(chi2_min+4),int(chi2_min+6)])
ax[0,0].text(int(chi2_min+2),C_chi[np.argmin(prof_C)],r'{g:.3e}'.format(g=C_chi[np.argmin(prof_C)]), color='k',alpha=0.5, fontsize=9)
ax[0,0].text(int(chi2_min+4),C_chi[C_sx],r'{g:.0e}'.format(g=errCC), color='b',alpha=0.5, fontsize=9)
ax[0,0].text(int(chi2_min+4),C_chi[C_dx],r'{g:.0e}'.format(g=-1*errC), color='r',alpha=0.5, fontsize=9)

ax[1,1].plot(B_chi,prof_C) 
ax[1,1].plot([B_chi[B_sx],B_chi[B_sx]],[int(chi2_min-2),int(chi2_min+10)], color=line_c, ls='dashed') 
ax[1,1].plot([B_chi[B_dx],B_chi[B_dx]],[int(chi2_min-2),int(chi2_min+10)], color=line_c, ls='dashed')

ax[1,1].text(B_chi[np.argmin(prof_B)],int(chi2_min+2),r'{g:.5e}'.format(g=B_chi[np.argmin(prof_B)]), color='k',alpha=0.5, fontsize=9)
ax[1,1].text(B_chi[B_sx],int(chi2_min+4),r'{g:.0e}'.format(g=errBB), color='b',alpha=0.5, fontsize=9)
ax[1,1].text(B_chi[B_dx],int(chi2_min+4),r'{g:.0e}'.format(g=-1*errB), color='r',alpha=0.5, fontsize=9)
ax[1,1].set_yticks([int(chi2_min),int(chi2_min+4),int(chi2_min+8)])


ax[1,0].set_axis_off()
ax[0,0].set_ylabel(r'$\tau\left(s\right)$') 
ax[1,1].set_xlabel(r'$\Omega\left(Hz\right)$',loc='center') 
ax[0,0].set_xlim(int(chi2_min-2),int(chi2_min+10)) 
ax[1,1].set_ylim(int(chi2_min-2),int(chi2_min+10))

plt.savefig(file+'_4'+'.png',
            bbox_inches ="tight",
            pad_inches = 1,
            transparent = True,
           facecolor ="w",
           edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

plt.show()


# Profilazione di V0 e Tau
fig, ax = plt.subplots(2, 2, figsize=(5.5, 5),constrained_layout = True, height_ratios=[3, 1], width_ratios=[1,3], sharex='col', sharey='row')
im = ax[0,1].contourf(A_chi,C_chi,Achi2D, levels=level, cmap=cmap)
fig.suptitle(r'$\chi^2 \left(v_0, \tau \right)$')
cbar = fig.colorbar(im, extend='both', shrink=0.9, ax=ax[0,1], ticks = [int(chi2_min),int(chi2_min+2),int(chi2_min+4),int(chi2_min+6)]) 
cbar.set_label(r'$\chi^2$',rotation=360)

CS = ax[0,1].contour(A_chi,C_chi,Achi2D, levels=[chi2_min+0.0001,chi2_min+1,chi2_min+2.3,chi2_min+3.8],linewidths=1, colors='k',alpha=0.5,linestyles='dotted')
ax[0,1].clabel(CS, inline=True, fontsize=9, fmt='%.0f')
ax[0,1].text(A_chi[np.argmin(prof_A)],C_chi[np.argmin(prof_C)],r'{g:.0f}'.format(g=chi2_min), color='k',alpha=0.5, fontsize=9)
ax[0,1].plot([A0,A1],[C_chi[C_sx],C_chi[C_sx]],color=line_c, ls='dashed')
ax[0,1].plot([A0,A1],[C_chi[C_dx],C_chi[C_dx]],color=line_c, ls='dashed')
ax[0,1].plot([A_chi[A_sx],A_chi[A_sx]],[C0,C1],color=line_c, ls='dashed')
ax[0,1].plot([A_chi[A_dx],A_chi[A_dx]],[C0,C1],color=line_c, ls='dashed')

ax[0,0].plot(prof_A,C_chi,ls='-') 
ax[0,0].plot([int(chi2_min-2),int(chi2_min+10)],[C_chi[C_sx],C_chi[C_sx]], color=line_c, ls='dashed') 
ax[0,0].plot([int(chi2_min-2),int(chi2_min+10)],[C_chi[C_dx],C_chi[C_dx]], color=line_c, ls='dashed')

ax[0,0].set_xticks([int(chi2_min),int(chi2_min+2),int(chi2_min+4),int(chi2_min+6)])
ax[0,0].text(int(chi2_min+2),C_chi[np.argmin(prof_C)],r'{g:.3e}'.format(g=C_chi[np.argmin(prof_C)]), color='k',alpha=0.5, fontsize=9)
ax[0,0].text(int(chi2_min+4),C_chi[C_sx],r'{g:.0e}'.format(g=errCC), color='b',alpha=0.5, fontsize=9)
ax[0,0].text(int(chi2_min+4),C_chi[C_dx],r'{g:.0e}'.format(g=-1*errC), color='r',alpha=0.5, fontsize=9)

ax[1,1].plot(A_chi,prof_C) 
ax[1,1].plot([A_chi[A_sx],A_chi[A_sx]],[int(chi2_min-2),int(chi2_min+10)], color=line_c, ls='dashed') 
ax[1,1].plot([A_chi[A_dx],A_chi[A_dx]],[int(chi2_min-2),int(chi2_min+10)], color=line_c, ls='dashed')

ax[1,1].text(A_chi[np.argmin(prof_A)],int(chi2_min+2),r'{g:.2f}'.format(g=A_chi[np.argmin(prof_A)]), color='k',alpha=0.5, fontsize=9)
ax[1,1].text(A_chi[A_sx],int(chi2_min+4),r'{g:.2f}'.format(g=errAA), color='b',alpha=0.5, fontsize=9)
ax[1,1].text(A_chi[A_dx],int(chi2_min+4),r'{g:.2f}'.format(g=-1*errA), color='r',alpha=0.5, fontsize=9)
ax[1,1].set_yticks([int(chi2_min),int(chi2_min+4),int(chi2_min+8)])

ax[1,0].set_axis_off()
ax[0,0].set_ylabel(r'$\tau\left(s\right)$') 
ax[1,1].set_xlabel(r'$v_0 \left(V\right)$') 
ax[0,0].set_xlim(int(chi2_min-2),int(chi2_min+10)) 
ax[1,1].set_ylim(int(chi2_min-2),int(chi2_min+10))

plt.savefig(file+'_5'+'.png',
            bbox_inches ="tight",
            pad_inches = 1,
            transparent = True,
            facecolor ="w",
            edgecolor ='w',
            orientation ='Portrait',
            dpi = 100)

plt.show()



