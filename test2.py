from asteroid.models import BaseModel

import soundfile as sf

import numpy as np

from scipy.io import wavfile



# 'from_pretrained' automatically uses the right model class (asteroid.models.DPRNNTasNet).

model = BaseModel.from_pretrained("./models/ConvTasNet_Libri2Mix_sepclean_16k.bin")

# You can pass a NumPy array:

mixture, samplingrate = sf.read("./output/ff2_EIG_5_200_8000.wav", dtype="float32", always_2d=True)
# print(mixture.shape)
# mixture = mixture[:,0].reshape(-1,1)
# mixture = mixture / np.max(np.abs(mixture)) * 0.7

# # Soundfile returns the mixture as shape (time, channels), and Asteroid expects (batch, channels, time)

mixture = mixture.transpose()

out_wavs = model.separate(mixture)

# Enhance single cluster

model = BaseModel.from_pretrained("./models/ConvTasNet_Libri1Mix_enhsingle_16k.bin")

out_wavs1 = np.array([[out_wavs[0,0,:]]])

out_wavs2 = np.array([[out_wavs[0,1,:]]])

enh_wavs1 = model.separate(out_wavs1)

enh_wavs2 = model.separate(out_wavs2)

# print('2')

# # Enhance single cluster

# model = BaseModel.from_pretrained("./models/DCUNet_Libri1Mix_enhsingle_16k.bin")

print('load dcunet')

enh_wavs1 = model.separate(enh_wavs1)

enh_wavs2 = model.separate(enh_wavs2)

wavfile.write("sep1_5.wav", samplingrate, enh_wavs1[0,0,:])

wavfile.write("sep2_5.wav", samplingrate, enh_wavs2[0,0,:])