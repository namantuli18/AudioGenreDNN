import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
file="0.wav"
signal,sr=librosa.load(file,sr=22_050)
#signal->np.array of 22_050 x 30 sec files
librosa.display.waveplot(signal,sr=22_050)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()


#fast fourier transform
fft=np.fft.fft(signal)
magnitude=np.abs(fft)
frequency=np.linspace(0,22_050,len(magnitude))

left_frequency=frequency[:int(len(frequency)/2)]
left_magnitude=magnitude[:int(len(frequency)/2)]

plt.plot(left_frequency,left_magnitude)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.show()

#stft
n_fft=2048
hop_length=512
stft=librosa.core.stft(signal,n_fft=n_fft,hop_length=hop_length)
spectrogram=np.abs(stft)

log_spectrogram=librosa.amplitude_to_db(spectrogram)

librosa.display.specshow(spectrogram,sr=sr,hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.show()

librosa.display.specshow(log_spectrogram,sr=sr,hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.show()

#MFCCs
MFCCs=librosa.feature.mfcc(signal, n_fft=n_fft,hop_length=hop_length,n_mfcc=13)
librosa.display.specshow(MFCCs,sr=sr,hop_length=hop_length)

plt.xlabel("Time")
plt.ylabel("MFCC")
plt.colorbar()
plt.show()