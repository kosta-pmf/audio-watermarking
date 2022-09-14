import numpy as np
from scipy.fftpack import dct, idct

#hyperparameters
fs = 3000 # starting frequency for watermark embedding
fe = 7000 # ending frequency for watermark embedding
k1 = 0.195
k2 = 0.08

def patchwork_multilayer_watermark_embedding(signal, watermark, sr=16000):
  '''
  Parameters:
    signal - 1D numpy array
    watermark - 1D numpy array of ones and zeros
    sr - sampling rate of the signal (default 16 kHz)  
  Returns:
    watermarked signal - 1D numpy array
  '''
  L = len(signal)
  
  si = int(fs/(sr/L))
  ei = int(fe/(sr/L))
  
  X = dct(signal, type=2, norm='ortho')
  
  Xs = X[si:(ei+1)]
  Ls = len(Xs)
  
  if Ls % (len(watermark)*2) != 0:
    Ls -= Ls % (len(watermark)*2)
    Xs = Xs[:Ls]

  Xsp = np.dstack((Xs[:Ls//2],Xs[:(Ls//2-1):-1])).flatten()

  # only the first layer
  segments = np.array_split(Xsp, len(watermark)*2)
  watermarked_segments = []
  for i in range(0, len(segments), 2):
    
    j = i//2 + 1
    rj = k1 * np.exp(-k2*j)

    m1j = np.mean(np.abs(segments[i]))
    m2j = np.mean(np.abs(segments[i+1]))

    mj = (m1j+m2j)/2
    mmj = min(m1j, m2j)

    m1jp = m1j
    m2jp = m2j

    if watermark[j-1] == 0 and (m1j - m2j) < rj * mmj:
      m1jp = mj + (rj*mmj/2)
      m2jp = mj - (rj*mmj/2)
    elif watermark[j-1] == 1 and (m2j - m1j) < rj * mmj:
      m1jp = mj - (rj*mmj/2)
      m2jp = mj + (rj*mmj/2)

    Y1j = segments[i] * m1jp/m1j
    Y2j = segments[i+1] * m2jp/m2j

    watermarked_segments.append(Y1j)
    watermarked_segments.append(Y2j)
    
  
  Ysp = np.hstack(watermarked_segments)
  Ys = np.hstack([Ysp[::2], Ysp[-1::-2]])

  Y = X[:]
  Y[si:(si+Ls)] = Ys
  watermarked_signal = idct(Y, type=2, norm='ortho')

  return watermarked_signal
  
def patchwork_multilayer_watermark_detection(watermarked_signal, watermark_length=40):
  '''
  Parameters:
    watermarked_signal - 1D numpy array
    watermark_length - integer representing the length of the embedded watermark
  Returns:
    detected watermark - 1D numpy array
  '''
  L = len(watermarked_signal)
  
  si = int(fs/(sr/L))
  ei = int(fe/(sr/L))
  
  X = dct(watermarked_signal, type=2, norm='ortho')
  
  Xs = X[si:(ei+1)]
  Ls = len(Xs)

  if Ls % (watermark_length*2) != 0:
    Ls -= Ls % (watermark_length*2)
    Xs = Xs[:Ls]

  Xsp = np.dstack((Xs[:Ls//2],Xs[:(Ls//2-1):-1])).flatten()

  segments = np.array_split(Xsp, watermark_length*2)
  watermark_bits = []

  for i in range(0, len(segments), 2):
    
    j = i//2 + 1
    rj = k1 * np.exp(-k2*j)

    m1j = np.mean(np.abs(segments[i]))
    m2j = np.mean(np.abs(segments[i+1]))

    dj = m1j - m2j

    if dj >= 0:
      watermark_bits.append(0)
    else:
      watermark_bits.append(1)

  return np.array(watermark_bits)
