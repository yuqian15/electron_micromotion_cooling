import numpy as np
import matplotlib.pyplot as plt

kB = 1.380649e-23 # Boltzman constant, given by Wikipedia
def noise_on_trap(deff, Temperature, Rp, fres, QualityFactor, TotalTime, dt, DrawNoise = False):

    # "white noise": https://www.quora.com/If-white-noise-has-a-uniform-pdf-then-how-can-it-have-a-Gaussian-distribution
    # Generate the JNNoise to be added into the simulation
    U_rms = np.sqrt(4 * kB * Temperature * Rp / dt)
    samples = TotalTime * 1.05 / dt
    # 1.5 is used to make  sure 
    samplerate = 1 / dt
    t = np.linspace(0, samples / samplerate, int(samples))
    bandwidth = U_rms

    # generate a initial noise signal(time domain)
    signal = np.random.normal(0, bandwidth, size = len(t))
    X = np.fft.fft(signal)
    N = len(X)
    freqs = np.fft.fftfreq(N) * samplerate
    freqs[0] = 1e-5
    f_U_noise = X / ( 1 + 1j * QualityFactor * (freqs / fres - fres / freqs))
    U_noise = np.fft.ifft(f_U_noise)
    JNNoise_Ex = U_noise * 1 / deff
    # finally, turn to electric field
    if DrawNoise:
        plt.plot(t * 1e6, JNNoise_Ex[:len(t)])
        plt.xlabel('Time( $\mu$s)')
        plt.ylabel('Noise field')
        plt.title('Noise with time')
        plt.grid()
        plt.show()
    return JNNoise_Ex[:len(t)]
    
    