#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 04:39:11 2024

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

#Sionna
try:
    import sionna
except ImportError:
    os.system("pip install sionna")
    import sionna

# TensorFlow
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

#scene
scene = load_scene(sionna.rt.scene.etoile)

# ULA
scene.tx_array = PlanarArray(num_rows=1, num_cols=8, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="tr38901", polarization="V")
scene.rx_array = PlanarArray(num_rows=1, num_cols=8, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="dipole", polarization="cross")

#Tx RX 
tx = Transmitter(name="tx", position=[8.5,21,27])
scene.add(tx)
rx = Receiver(name="rx", position=[8.5,21,27], orientation=[0, 0, 0])
scene.add(rx)
tx.look_at(rx)  

#fc path 
scene.frequency = 3e9  # in Hz
scene.synthetic_array = True  # Ray tracing done per antenna element
paths = scene.compute_paths(max_depth=5, num_samples=int(1e6))

#CIR data
a, tau = paths.cir()
print("Shape of tau:", tau.shape)
print("Shape of a:", a.shape)

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
snr_db = 20
distances = tau / c
radius = 100

def steering_vector(theta, M, d, fc, c):
    theta_rad = np.deg2rad(theta)
    k = 2 * np.pi * fc / c
    n = np.arange(M)
    return np.exp(1j * k * d * n * np.sin(theta_rad))

W = 1e09
Ts = 1/W  
N = 2**9
T = Ts * (N - 1)
t = np.linspace(0, T, N)

# Transmit signal
Nsend = 100
Tsend = Ts * (Nsend - 1)
s = np.where(t <= Tsend, 1, 0)

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

# Received signal
received_signal = np.zeros(Nsend + Lmax - Lmin, dtype=np.complex128)
for n in range(Nsend + Lmax - Lmin):
    for j in range(Lmax - Lmin):
        l = j + Lmin
        received_signal[n] += s[n-l] * h[n, j]

# Noise 
noise_power = np.mean(np.abs(received_signal)**2) / (10 ** (snr_db / 10))
noise = np.sqrt(noise_power / 2) * (np.random.randn(*received_signal.shape) + 1j * np.random.randn(*received_signal.shape))
received_signal += noise


plt.plot(t[Lmin:Nsend+Lmax], np.real(received_signal))
plt.ylabel("Receive baseband signal")
plt.xlabel("Time")
plt.show()

# Matched filter
matched_filter = convolve(s, received_signal, "same")

plt.plot(t, np.real(matched_filter))
plt.ylabel("Matched filter")
plt.xlabel("Time")
plt.show()

# Signals source and sensor 
Y = np.zeros((M, N), dtype=complex)
for theta in theta_sources:
    sv = steering_vector(theta, M, d, fc, c)
    Y += np.outer(sv, np.fft.fft(received_signal))

# Covariance
R = Y @ Y.conj().T / N
print("Shape of R:", R.shape)
print("Rank of R (number of non-zero singular values):", np.linalg.matrix_rank(R))
print("Covariance:", R)

plt.figure(figsize=(8, 6))
sns.heatmap(np.real(R), cmap='viridis')
plt.title('Covariance Matrix')
plt.xlabel('Sensor Index')
plt.ylabel('Sensor Index')
plt.show()

# Eigenvalue Decomposition
eigenvalues, eigenvectors = eigh(R)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
print("EigenValues:", eigenvalues)
print("EigenVector:", eigenvectors)

plt.figure(figsize=(10, 6))
plt.stem(range(len(eigenvalues)), np.log(eigenvalues), basefmt=" ")
plt.title('Spectrum of Eigenvalues')
plt.xlabel('Index')
plt.ylabel('Eigenvalue Magnitude')
plt.legend()
plt.grid(True)
plt.show()

# Noise subspace
noise_subspace = eigenvectors[:, len(theta_sources):]
print("Shape of noise_subspace:", noise_subspace.shape)
print("NoiseSubspace:", noise_subspace)

# MUSIC
angles = np.linspace(-90, 90, 360)
music_spectrum = np.zeros_like(angles, dtype=float)
for i, angle in enumerate(angles):
    a = steering_vector(angle, M, d, fc, c)
    music_spectrum[i] = 1 / np.abs(a.conj().T @ noise_subspace @ noise_subspace.conj().T @ a)

music_spectrum = 10 * np.log10(music_spectrum / np.max(music_spectrum))
peaks, _ = find_peaks(music_spectrum, height=np.max(music_spectrum) - 1, distance=5)
detected_angles = angles[peaks]

plt.figure(figsize=(10, 6))
plt.plot(angles, music_spectrum, label='MUSIC Spectrum')
for detected_angle in detected_angles:
    plt.axvline(x=detected_angle, color='red', linestyle='--', label=f'Detected: {detected_angle:.2f}°' if detected_angle == detected_angles[0] else "")
plt.axvline(x=10, color='green', linestyle='-', label='Expected: 10°')
plt.axvline(x=-25, color='blue', linestyle='-', label='Expected: -25°')
plt.title('MUSIC Spectrum with Detected and Expected Angles')
plt.xlabel('Angle (degrees)')
plt.ylabel('Normalized Spectrum (dB)')
plt.legend()
plt.grid(True)
plt.show()

print("Angles:", angles)
print("Number of angles:", len(angles))
print("Music Spectrum Shape:", music_spectrum.shape)
print("Music Spectrum:", music_spectrum)
print("Detected Angles (Degrees):", detected_angles)

# DFT
def dft_doa(Y, M, d, fc, c):
    Psi = Y

    angles = np.linspace(-90, 90, 720)
    dft_spectrum = np.zeros_like(angles, dtype=float)
    for i, angle in enumerate(angles):
        sv = steering_vector(angle, M, d, fc, c)
        dft_spectrum[i] = np.abs(np.dot(sv.conj().T, np.dot(Psi, Psi.conj().T).dot(sv)))
        
    dft_spectrum = 10 * np.log10(dft_spectrum / np.max(dft_spectrum))
    peaks, _ = find_peaks(dft_spectrum, height=np.max(dft_spectrum) - 1, distance=5)
    detected_angles = angles[peaks]

    return detected_angles, dft_spectrum, angles

detected_angles_dft, dft_spectrum, angles_dft = dft_doa(Y, M, d, fc, c)

plt.figure(figsize=(10, 6))
plt.plot(angles_dft, dft_spectrum, label='DFT Spectrum')
for detected_angle in detected_angles_dft:
    plt.axvline(x=detected_angle, color='red', linestyle='--', label=f'Detected: {detected_angle:.2f}°' if detected_angle == detected_angles_dft[0] else "")
plt.axvline(x=10, color='green', linestyle='-', label='Expected: 10°')
plt.axvline(x=-25, color='blue', linestyle='-', label='Expected: -25°')
plt.title('DFT Spectrum with Detected and Expected Angles')
plt.xlabel('Angle (degrees)')
plt.ylabel('Normalized Spectrum (dB)')
plt.legend()
plt.grid(True)
plt.show()

print("MUSIC Detected Angles (Degrees):", detected_angles)
print("DFT Detected Angles (Degrees):", detected_angles_dft)


#Path Information
num_paths = paths.tau.shape[3]

print("Detailed Path Information:")
for path_idx in range(num_paths):
    # Properties
    channel_coefficient = paths.a[0, 0, 0, 0, 0, path_idx, 0].numpy()
    propagation_delay = paths.tau[0, 0, 0, path_idx].numpy()
    zenith_angle_departure = paths.theta_t[0, 0, 0, path_idx].numpy()
    azimuth_angle_departure = paths.phi_t[0, 0, 0, path_idx].numpy()
    zenith_angle_arrival = paths.theta_r[0, 0, 0, path_idx].numpy()
    azimuth_angle_arrival = paths.phi_r[0, 0, 0, path_idx].numpy()

    zenith_angle_departure_deg = np.degrees(zenith_angle_departure)
    azimuth_angle_departure_deg = np.degrees(azimuth_angle_departure)
    zenith_angle_arrival_deg = np.degrees(zenith_angle_arrival)
    azimuth_angle_arrival_deg = np.degrees(azimuth_angle_arrival)

    print(f"\n--- Detailed results for path {path_idx + 1} ---")
    print(f"Channel coefficient: {channel_coefficient}")
    print(f"Propagation delay: {propagation_delay * 1e6:.5f} us")  
    print(f"Zenith angle of departure: {zenith_angle_departure_deg:.4f}°")
    print(f"Azimuth angle of departure: {azimuth_angle_departure_deg:.4f}°")
    print(f"Zenith angle of arrival: {zenith_angle_arrival_deg:.4f}°")
    print(f"Azimuth angle of arrival: {azimuth_angle_arrival_deg:.4f}°")

