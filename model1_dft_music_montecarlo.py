#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 11:20:26 2024

@author: paolapolanco
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, convolve
from numpy.linalg import eigh

def steering_vector(theta, M, d, fc, c):
    theta_rad = np.deg2rad(theta)
    k = 2 * np.pi * fc / c
    n = np.arange(M)
    return np.exp(1j * k * d * n * np.sin(theta_rad))

np.random.seed(11)

# Parameters
c = 3e8
fc = 3e9
lambda_c = c / fc
d = lambda_c / 2
M = 8
distances = np.array([100, 100])

W = 1e9
Ts = 1 / W
N = 2 ** 9
T = Ts * (N - 1)
t = np.linspace(0, T, N)

# Transmit signal
Nsend = 100
Tsend = Ts * (Nsend - 1)
s = np.where(t <= Tsend, 1, 0)

alphas = (lambda_c/ (4 * np.pi * distances)) **2

delays = distances / c

# Channel taps Martins model 
Lmin = 0
Lmax = N - Nsend
h = np.zeros((Nsend + Lmax - Lmin, Lmax - Lmin), dtype=np.complex128)
for n in range(Nsend + Lmax - Lmin):
    for j in range(Lmax - Lmin):
        l = j + Lmin
        h[n, j] = np.sum([alpha * np.sinc(t[l] - W * tau) for alpha, tau in zip(alphas, delays)])


# Simulate the signal do DFT and MUSIC
def simulate(snr_db, theta_sources):
    received_signal = np.zeros(Nsend + Lmax - Lmin, dtype=np.complex128)
    for n in range(Nsend + Lmax - Lmin):
        for j in range(Lmax - Lmin):
            b = n + Lmin
            l = j + Lmin
            if b - l < len(s) and b - l >= 0:
                received_signal[n] += s[b - l] * h[n, j]

    noise_power = np.mean(np.abs(received_signal)**2) / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*received_signal.shape) + 1j * np.random.randn(*received_signal.shape))
    received_signal += noise


    Y = np.zeros((M, N), dtype=complex)
    for theta in theta_sources:
        sv = steering_vector(theta, M, d, fc, c)
        Y += np.outer(sv, np.fft.fft(received_signal))

    R = Y @ Y.conj().T / N

    eigenvalues, eigenvectors = eigh(R)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    noise_subspace = eigenvectors[:, len(theta_sources):]


    # MUSIC
    angles = np.linspace(-90, 90, 360)
    music_spectrum = np.zeros_like(angles, dtype=float)
    for i, angle in enumerate(angles):
        a = steering_vector(angle, M, d, fc, c)
        music_spectrum[i] = 1 / np.abs(a.conj().T @ noise_subspace @ noise_subspace.conj().T @ a)
    music_spectrum = 10 * np.log10(music_spectrum / np.max(music_spectrum))
    music_peaks, _ = find_peaks(music_spectrum, height=np.max(music_spectrum) - 2, distance=5)
    music_detected_angles = angles[music_peaks]


    # DFT
    def dft_doa(Y, M, d, fc, c):
        Psi = Y

        angles = np.linspace(-90, 90, 360)  # Ensure angles align with size of Psi
        dft_spectrum = np.zeros_like(angles, dtype=float)
        for i, angle in enumerate(angles):
            sv = steering_vector(angle, M, d, fc, c)
            dft_spectrum[i] = np.abs(np.dot(sv.conj().T, np.dot(Psi, Psi.conj().T).dot(sv)))

        dft_spectrum = 10 * np.log10(dft_spectrum / np.max(dft_spectrum))
        peaks, _ = find_peaks(dft_spectrum, height=np.max(dft_spectrum) - 2, distance=5)
        detected_angles = angles[peaks]

        return detected_angles, dft_spectrum, angles

    dft_detected_angles, _, _ = dft_doa(Y, M, d, fc, c)
    
    print(f"SNR: {snr_db} dB")
    print(f"MUSIC detected angles: {music_detected_angles}")
    print(f"DFT detected angles: {dft_detected_angles}")

    return dft_detected_angles, music_detected_angles

# Montecarlo
snr_range = np.linspace(0, 30, 7)
num_trials = 2000


def monte_carlo_simulation(theta_sources):
    dft_rmse = {theta: np.zeros_like(snr_range, dtype=float) for theta in theta_sources}
    music_rmse = {theta: np.zeros_like(snr_range, dtype=float) for theta in theta_sources}
    
    for i, snr_db in enumerate(snr_range):
        dft_errors = {theta: [] for theta in theta_sources}
        music_errors = {theta: [] for theta in theta_sources}
        for _ in range(num_trials):
            dft_detected_angles, music_detected_angles = simulate(snr_db, theta_sources)
            
            if len(dft_detected_angles) > 0:
                for theta in theta_sources:
                    closest_angle = min(dft_detected_angles, key=lambda x: abs(x - theta))
                    dft_errors[theta].append(abs(closest_angle - theta))
            
            if len(music_detected_angles) > 0:
                for theta in theta_sources:
                    closest_angle = min(music_detected_angles, key=lambda x: abs(x - theta))
                    music_errors[theta].append(abs(closest_angle - theta))
        
        for theta in theta_sources:
            if dft_errors[theta]:
                dft_rmse[theta][i] = np.sqrt(np.mean(np.array(dft_errors[theta]) ** 2))
            if music_errors[theta]:
                music_rmse[theta][i] = np.sqrt(np.mean(np.array(music_errors[theta]) ** 2))
    
    return dft_rmse, music_rmse

#Run simulation
theta_sources = [10, -25]
dft_rmse, music_rmse = monte_carlo_simulation(theta_sources)

# Plotting RMSE vs SNR for both DFT and MUSIC algorithms
plt.figure(figsize=(12, 8))

# For 10°
plt.plot(snr_range, dft_rmse[10], label='DFT for 10°', marker='o', linestyle='--', color='blue')
plt.plot(snr_range, music_rmse[10], label='MUSIC for 10°', marker='x', linestyle='--', color='green')

# For -25°
plt.plot(snr_range, dft_rmse[-25], label='DFT for -25°', marker='o', linestyle='--', color='red')
plt.plot(snr_range, music_rmse[-25], label='MUSIC for -25°', marker='x', linestyle='--', color='purple')

plt.xlabel('SNR (dB)', fontsize=14)
plt.ylabel('RMSE (degrees)', fontsize=14)
plt.title('RMSE vs SNR for DFT and MUSIC Algorithms (10° and -25)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
