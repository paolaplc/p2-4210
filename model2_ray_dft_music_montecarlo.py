#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 04:47:35 2024

@author: paolapolanco
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, convolve
from numpy.linalg import eigh
import seaborn as sns
import os

# GPU configuration
gpu_num = 0  
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import Sionna
try:
    import sionna
except ImportError:
    os.system("pip install sionna")
    import sionna

# TensorFlow GPU
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
tf.get_logger().setLevel('ERROR')
tf.random.set_seed(1)  # Set global random seed for reproducibility

# Sionna RT components
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray

# Load scene
scene = load_scene(sionna.rt.scene.etoile)

# ULA
scene.tx_array = PlanarArray(num_rows=1, num_cols=8, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="tr38901", polarization="V")
scene.rx_array = PlanarArray(num_rows=1, num_cols=8, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="dipole", polarization="cross")

# Tx Rx
tx = Transmitter(name="tx", position=[8.5,21,27])
scene.add(tx)
rx = Receiver(name="rx", position=[8.5,21,27], orientation=[0, 0, 0])
scene.add(rx)
tx.look_at(rx)  #Tx points towards receiver

# Frequency and propagation paths
scene.frequency = 3e9  # in Hz
scene.synthetic_array = True  # Ray tracing done per antenna element
paths = scene.compute_paths(max_depth=5, num_samples=int(1e6))

#CIR data
a, tau = paths.cir()
print("Shape of tau:", tau.shape)
print("Shape of a:", a.shape)

# Plot data
t = tau[0, 0, 0, :] / 1e-9  # Convert time to nanoseconds
a_abs = np.abs(a)[0, 0, 0, 0, 0, :, 0]  # Magnitude of coefficients
a_max = np.max(a_abs)

t = np.concatenate([(0,), t, (np.max(t) * 1.1,)])
a_abs = np.concatenate([(np.nan,), a_abs, (np.nan,)])

plt.figure(figsize=(10, 5))
plt.stem(t, a_abs, linefmt='b-', markerfmt='bo', basefmt='r-')
plt.title("Channel Impulse Response Realization")
plt.xlabel(r"Delay $\tau$ [ns]")
plt.ylabel(r"Magnitude $|a|$")
plt.grid(True)
plt.xlim([0, np.max(t)])
plt.ylim([0, a_max * 1.1])
plt.show()

# Parameters
c = 3e8
fc = 3e9
lambda_c = c / fc
d = lambda_c / 2
M = 8
theta_sources = np.array([10, -25])
distances = tau / c
radius = 100

# Steering vector
def steering_vector(theta, M, d, fc, c):
    theta_rad = np.deg2rad(theta)
    k = 2 * np.pi * fc / c
    n = np.arange(M)
    return np.exp(1j * k * d * n * np.sin(theta_rad))

# Processing parameters
W = 1e09
Ts = 1/W  
N = 2**9
T = Ts * (N - 1)
t = np.linspace(0, T, N)

# Transmit signal
Nsend = 100
Tsend = Ts * (Nsend - 1)
s = np.where(t <= Tsend, 1, 0)

# Plot transmit baseband pulse
plt.plot(t, s)
plt.ylabel("Transmit baseband pulse")
plt.xlabel("Time")
plt.show()

# CIR data for a and tau 
alphas = np.abs(a)[0, 0, 0, 0, 0, :, 0]
delays = tau[0, 0, 0, :]

# Channel taps Martins model 
Lmin = 0
Lmax = N - Nsend
h = np.zeros((Nsend + Lmax - Lmin, Lmax - Lmin), dtype=np.complex128)
for n in range(Nsend + Lmax - Lmin):
    for j in range(Lmax - Lmin):
        l = j + Lmin
        h[n, j] = np.sum([alpha * np.sinc(t[l] - W * tau) for alpha, tau in zip(alphas, delays)])

# Monte Carlo simulation for DoA estimation
def simulate(snr_db, theta_sources):
    # Received signal
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

    # Signals for each source and each antenna
    Y = np.zeros((M, N), dtype=complex)
    for theta in theta_sources:
        sv = steering_vector(theta, M, d, fc, c)
        Y += np.outer(sv, np.fft.fft(received_signal))

    # Covariance matrix
    R = Y @ Y.conj().T / N

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = eigh(R)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Noise subspace
    noise_subspace = eigenvectors[:, len(theta_sources):]

    
    # MUSIC
    angles = np.linspace(-90, 90, 360)
    music_spectrum = np.zeros_like(angles, dtype=float)
    for i, angle in enumerate(angles):
        a = steering_vector(angle, M, d, fc, c)
        music_spectrum[i] = 1 / np.abs(a.conj().T @ noise_subspace @ noise_subspace.conj().T @ a)
    music_spectrum = 10 * np.log10(music_spectrum / np.max(music_spectrum))
    music_peaks, _ = find_peaks(music_spectrum, height=np.max(music_spectrum) - 1, distance=5)
    music_detected_angles = angles[music_peaks]
    
    # DFT
    def dft_doa(Y, M, d, fc, c):
        Psi = Y

        angles = np.linspace(-90, 90, 360)  # Ensure angles align with size of Psi
        dft_spectrum = np.zeros_like(angles, dtype=float)
        for i, angle in enumerate(angles):
            sv = steering_vector(angle, M, d, fc, c)
            dft_spectrum[i] = np.abs(np.dot(sv.conj().T, np.dot(Psi, Psi.conj().T).dot(sv)))

        # Normalize
        dft_spectrum = 10 * np.log10(dft_spectrum / np.max(dft_spectrum))
        peaks, _ = find_peaks(dft_spectrum, height=np.max(dft_spectrum) - 1, distance=5)
        detected_angles = angles[peaks]

        return detected_angles, dft_spectrum, angles

    dft_detected_angles, _, _ = dft_doa(Y, M, d, fc, c)
    
    print(f"SNR: {snr_db} dB")
    print(f"MUSIC detected angles: {music_detected_angles}")
    print(f"DFT detected angles: {dft_detected_angles}")

    return dft_detected_angles, music_detected_angles

# Monte Carlo simulation parameters
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
            
            # Errors for DFT
            if len(dft_detected_angles) > 0:
                for theta in theta_sources:
                    closest_angle = min(dft_detected_angles, key=lambda x: abs(x - theta))
                    dft_errors[theta].append(abs(closest_angle - theta))
            
            # Errors for MUSIC
            if len(music_detected_angles) > 0:
                for theta in theta_sources:
                    closest_angle = min(music_detected_angles, key=lambda x: abs(x - theta))
                    music_errors[theta].append(abs(closest_angle - theta))
        
        # Calculate RMSE for each angle
        for theta in theta_sources:
            if dft_errors[theta]:
                dft_rmse[theta][i] = np.sqrt(np.mean(np.array(dft_errors[theta]) ** 2))
            if music_errors[theta]:
                music_rmse[theta][i] = np.sqrt(np.mean(np.array(music_errors[theta]) ** 2))
    
    return dft_rmse, music_rmse


# Run Monte Carlo
theta_sources = [10, -25]
dft_rmse, music_rmse = monte_carlo_simulation(theta_sources)

# Plot RMSE vs SNR for both DFT and MUSIC algorithms
plt.figure(figsize=(12, 8))

# For 10°
plt.plot(snr_range, dft_rmse[10], label='DFT for 10°', marker='o', linestyle='--', color='blue')
plt.plot(snr_range, music_rmse[10], label='MUSIC for 10', marker='x', linestyle='--', color='green')

# For -25°
plt.plot(snr_range, dft_rmse[-25], label='DFT for -25°', marker='o', linestyle='--', color='red')
plt.plot(snr_range, music_rmse[-25], label='MUSIC for -25°', marker='x', linestyle='--', color='purple')

plt.xlabel('SNR (dB)', fontsize=14)
plt.ylabel('RMSE (degrees)', fontsize=14)
plt.title('RMSE vs SNR for DFT and MUSIC Algorithms (10° and -25°)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()