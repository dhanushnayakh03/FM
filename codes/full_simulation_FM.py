from scipy.fftpack import fft,fftshift
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import scipy.signal as signal

# Load WAV file
path = r'D:\2and sem IITH\Signals and system\FM Transmitter\codes\Sample_Sound.wav'
sample_rate, audio = wavfile.read(path)



# Frequency modulation
fc = 100e6  # carrier frequency

N = len(audio)
dt = 1e-3/N # time span 1 msec
t = np.arange(N)
t = t*dt

audio = 127*(audio.astype(np.int16)/ np.power(2,15))
ym = audio



kf = 20 # freq sensitivity
cumsum = np.cumsum(ym)  # Discrete summation

c = np.cos(2*np.pi*fc*t)
y_fm = np.cos(2*np.pi*fc*t + kf*cumsum*(1/sample_rate)) #this is the FM signal 

fm_fft = np.fft.fft(y_fm) #FFT of Fm signal 
f_1 = np.fft.fftfreq(len(y_fm), d=dt) #range of freq for FM signal

plt.plot(f_1,abs(fm_fft))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.title('FM Signal in freq domain')
plt.savefig("figs/modulation_plot.png")
plt.clf()

# Calculate power spectral density of FM signal
fm_psd = np.abs(fm_fft)**2
# Calculate frequency range of FM signal
fm_freqs = np.fft.fftfreq(len(fm_psd), 1/sample_rate)

# Find frequency range with significant power of FM signal
fm_mask = fm_psd > 0.1*np.max(fm_psd)
fm_freq_range = fm_freqs[fm_mask]
fm_bandwidth = max(fm_freq_range) - min(fm_freq_range)
print('FM Bandwidth:', fm_bandwidth, 'Hz')


'''Now the receiving part of the FM'''

#First step :- FM Signal is multiplied by the cosine and then taken into freq domain for Filter Action

fm_mul=y_fm*np.cos(2*np.pi*fc*t)

fm_mul_fft=np.fft.fft(fm_mul)
f_2=np.fft.fftfreq(len(fm_mul), d=dt)


plt.plot(f_2,abs(fm_mul_fft))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.grid(True)
plt.title('local oscillator')
plt.savefig("figs/output_after_mixing.png")
plt.clf()

#Step 2 :- Look at the mixer plot and pass it through a low pass filter to eliminate the carrier


cutoff = 3e3 

b, a = signal.butter(6, cutoff/(sample_rate/2), 'low') # 6th order Butterworth filter
z = signal.filtfilt(b, a, fm_mul)
z1 = np.fft.fft(z)
z_f=np.fft.fftfreq(len(z),1/sample_rate)
plt.plot(z_f, abs(z1))
plt.title('Filtered FM signal')
plt.savefig("figs/result_after_LPF.png")


#Step 3:- Appying Cos inverter and differentiator 

y5 = 10*np.arccos(z)    # cos inverse of LPF output


# Differentiation of cos inverse
i=0
y6 = np.zeros(len(t))

#Derivative action using finite difference method 

while i< len(t)-1:
      y6[i] = (y5[i+1] - y5[i]) / (t[i+1] - t[i])
      i = i+1

z5 = np.fft.fft(y6) # received signal which should match with  the transmitted signal 

f_5 = np.fft.fftfreq(len(y6), d=1/sample_rate)
plt.plot(f_5,abs(z5),'b')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.grid(True)
plt.title('Final Output Signal')
plt.savefig("figs/Final_Received_Signal.png")
