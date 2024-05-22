#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 17:29:55 2024

@author: paolapolanco
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, convolve
from numpy.linalg import eigh
import seaborn as sns

def steering_vector(theta, M, d, fc, c):
    theta_rad = np.deg2rad(theta)
    k = 2 * np.pi * fc / c
    n = np.arange(M)
    return np.exp(1j * k * d * n * np.sin(theta_rad))

def run_simulation(theta_sources):
    np.random.seed(11)

    # Parameters
    c = 3e8
    fc = 3e9
    lambda_c = c / fc
    d = lambda_c / 2
    M = 8
    snr_db = 10
    distances = np.array([100, 100])

    W = 1e09
    Ts = 1 / W
    N = 2**9
    T = Ts * (N - 1)
    t = np.linspace(0, T, N)

    # Transmit signal
    Nsend = 100
    Tsend = Ts * (Nsend - 1)
    s = np.where(t <= Tsend, 1, 0)

    alphas = (lambda_c/ (4 * np.pi * distances)) **2

    delays = distances / c

    plt.figure()
    plt.plot(t, s)
    plt.ylabel("Transmit baseband pulse")
    plt.xlabel("Time")
    plt.title("Transmit Signal")
    plt.show()


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
            b = n + Lmin
            l = j + Lmin
            received_signal[n] += s[b - l] * h[n, j]

    noise_power = np.mean(np.abs(received_signal)**2) / (10**(snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*received_signal.shape) + 1j * np.random.randn(*received_signal.shape))
    received_signal += noise

    plt.figure()
    plt.plot(t[Lmin:Nsend + Lmax], np.real(received_signal))
    plt.ylabel("Receive baseband signal")
    plt.xlabel("Time")
    plt.title("Received Signal")
    plt.show()


    # Matched filter
    matched_filter = convolve(s, received_signal, "same")

    plt.figure()
    plt.plot(t, np.real(matched_filter))
    plt.ylabel("Matched filter")
    plt.xlabel("Time")
    plt.title("Matched Filter Output")
    plt.show()

    matched_filter_2 = convolve(received_signal, s[::-1], "same")

    plt.figure()
    plt.plot(t, np.real(matched_filter_2))
    plt.ylabel("Matched filter")
    plt.xlabel("Time")
    plt.title("Matched Filter Output (Reverse)")
    plt.show()

    # Signals for each source and each antenna
    Y = np.zeros((M, N), dtype=complex)
    for theta in theta_sources:
        sv = steering_vector(theta, M, d, fc, c)
        Y += np.outer(sv, np.fft.fft(received_signal))

    R = Y @ Y.conj().T / N

    plt.figure(figsize=(8, 6))
    sns.heatmap(np.real(R), cmap='viridis')
    plt.title('Covariance Matrix')
    plt.xlabel('Sensor Index')
    plt.ylabel('Sensor Index')
    plt.show()

    # Eigenvalue D
    eigenvalues, eigenvectors = eigh(R)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    plt.figure(figsize=(10, 6))
    if 'use_line_collection' in plt.stem.__code__.co_varnames:
        plt.stem(range(len(eigenvalues)), np.log(eigenvalues), basefmt=" ", use_line_collection=True)
    else:
        plt.stem(range(len(eigenvalues)), np.log(eigenvalues), basefmt=" ")
    plt.title('Spectrum of Eigenvalues')
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue Magnitude')
    plt.grid(True)
    plt.show()
    
    noise_subspace = eigenvectors[:, len(theta_sources):]

    # MUSIC
    angles = np.linspace(-90, 90, 720)
    music_spectrum = np.zeros_like(angles, dtype=float)
    for i, angle in enumerate(angles):
        a = steering_vector(angle, M, d, fc, c)
        music_spectrum[i] = 1 / np.abs(a.conj().T @ noise_subspace @ noise_subspace.conj().T @ a)
        
    music_spectrum = 10 * np.log10(music_spectrum / np.max(music_spectrum))
    peaks, _ = find_peaks(music_spectrum, height=np.max(music_spectrum) - 1, distance=5)
    detected_angles = angles[peaks]
    colors = ['green', 'blue'] 
    plt.figure(figsize=(10, 6))
    plt.plot(angles, music_spectrum, label='MUSIC Spectrum')
    for detected_angle in detected_angles:
        plt.axvline(x=detected_angle, color='red', linestyle='--', label=f'Detected: {detected_angle:.2f}°' if detected_angle == detected_angles[0] else "")
    for theta, color in zip(theta_sources, colors):
        plt.axvline(x=theta, linestyle='-', color=color, label=f'Expected: {theta}°')
    plt.title('MUSIC Spectrum with Detected and Expected Angles')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Normalized Spectrum (dB)')
    plt.legend()
    plt.grid(True)
    plt.show()

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

    detected_angles_dft, dft_spectrum, angles = dft_doa(Y, M, d, fc, c)
    colors = ['green', 'blue'] 
    plt.figure(figsize=(10, 6))
    plt.plot(angles, dft_spectrum, label='DFT Spectrum')
    for detected_angle in detected_angles_dft:
        plt.axvline(x=detected_angle, color='red', linestyle='--', label=f'Detected: {detected_angle:.2f}°' if detected_angle == detected_angles_dft[0] else "")
    for theta, color in zip(theta_sources, colors):
        plt.axvline(x=theta, linestyle='-', color=color, label=f'Expected: {theta}°')
    plt.title('DFT Spectrum with Detected and Expected Angles')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Normalized Spectrum (dB)')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("MUSIC Detected Angles (Degrees):", detected_angles)
    print("DFT Detected Angles (Degrees):", detected_angles_dft)

# Run
print("Running simulation for angles [10, -25]...")
run_simulation([10, -25])


