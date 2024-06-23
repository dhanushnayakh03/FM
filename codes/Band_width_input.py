# This code calculates the bandwidth of the input signal
#And this is the generalised approach to find the bandwidth of any signal


import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,fftshift
import scipy.signal as signal
from scipy.io import wavfile

#extract the audio data and sampling rate
path = r'D:\2and sem IITH\Signals and system\FM Transmitter\codes\Sample_Sound.wav'
sample_rate , audio_data = wavfile.read(path)
print(sample_rate)


#take fft to go to freq domain
fft_audio = np.fft.fft(audio_data)

#calculate the freq range 
f_range = np.fft.fftfreq(len(audio_data),1/sample_rate)



#Power spectral density calculation
PSD = (np.abs(fft_audio))**2

#calculate the freq range
fr_PSD = np.fft.fftfreq(len(PSD),1/sample_rate)

# According to carsons rule maximum power is located in the range of frequencies of the bandwidth


#Now we need to find frequencies above threshold
threshold = PSD > 0.1*max(PSD) #here threshold is 0.1*max power
band_freq_range = fr_PSD[threshold] 


plt.plot(f_range , np.abs(fft_audio))
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Frequency domain representation of input signal")

plt.savefig("figs/input_fft.png")



bandwidth = max(band_freq_range) - min(band_freq_range)
print("Bandwidth of input signal: ",bandwidth) # in Hz