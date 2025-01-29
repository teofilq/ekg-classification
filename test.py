import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk

# Generare semnal EKG simulat
sampling_rate = 500  # Hz
duration = 5  # secunde
ekg_signal = nk.ecg_simulate(duration=duration, sampling_rate=sampling_rate)

# Configurare grafic cu pătrățele
plt.figure(figsize=(10, 6))

# Setăm raportul de aspect pentru pătrățele perfecte
plt.gca().set_aspect('equal', adjustable='box')

# Desenare grilă mică (light red, 1 unitate)
plt.grid(which='minor', color='lightcoral', linestyle='-', linewidth=0.5)
plt.minorticks_on()  # Activează marcajele minore

# Desenare grilă mare (red, 5 unități)
plt.grid(which='major', color='red', linestyle='-', linewidth=1)

# Plasare semnal EKG
time = np.linspace(0, duration, len(ekg_signal))  # Timpul în secunde
plt.plot(time, ekg_signal, label="Semnal EKG", color="black", linewidth=1)

# Setări axă
plt.xlabel("Timp (secunde)")
plt.ylabel("Amplitudine (mV)")
plt.title("Semnal EKG cu pătrățele roșii")
plt.legend()

# Configurare intervale pentru grilă
plt.xticks(np.arange(0, duration + 1, 0.30))  # Marcaje de timp la 0.04 secunde
plt.yticks(np.arange(-2, 2.5, 0.2))  # Marcaje de amplitudine la fiecare 0.2 mV

# Afișare grafic
plt.show()

