import numpy as np
from scipy.fftpack import dct, idct
from numpy.linalg import svd

#hyperparameters
d0 = 0.83
delta = 3.5793656
D = 0.65
gamma1 = 0.136 # for 44.1 kHz sampling rate
gamma2 = 0.181 # for 44.1 kHz sampling rate
alpha = 3

def encrypt_watermark(w):
  d = np.empty(w.shape)
  d[0] = d0
  e = np.empty(w.shape, dtype='int')
  for i in range(1, n):
    d[i] = delta * d[i-1] *  (1 - d[i-1])

  e = (d >= D).astype('int')
  return  w ^ e
  
def decrypt_watermark(wp, e):
  return wp ^ e
  
  
def modify_svd_pair(l1, l2, watermark_bit):

  if watermark_bit == 0:
    if l1/l2 > 1/(1+alpha):
      l1 = (l1+l2*(1+alpha))/(alpha**2+2*alpha+2)
      l2 = (1+alpha)*l1

  else:
    if l1/l2 < 1+alpha:
      l2 = (l2+l1*(1+alpha))/(alpha**2+2*alpha+2)
      l1 = (1+alpha)*l2

  return l1, l2
  
def fsvc_watermark_embedding(signal, watermark, sr=16000):
  '''
  Parameters:
    signal - 1D numpy array
    watermark - 1D numpy array of ones and zeros
    sr - sampling rate of the signal (default 16 kHz)  
  Returns:
    watermarked signal - 1D numpy array
  '''
  gamma1 = gamma1 * sr/44100
  gamma2 = gamma2 * sr/44100

  frames = np.array_split(signal, len(watermark))

  watermarked_frames = []

  for i, frame in enumerate(frames):
    x1, x2 = np.array_split(frame, 2)

    X1 = dct(x1, type=2, norm='ortho')
    X2 = dct(x2, type=2, norm='ortho')

    low = int(gamma1*len(frame))
    high = int(gamma2*len(frame)+1)

    X1p = np.expand_dims(X1[low:high], axis=-1)
    X2p = np.expand_dims(X2[low:high], axis=-1)
    
    u1, s1, v1 = svd(X1p, full_matrices=False)
    u2, s2, v2 = svd(X2p, full_matrices=False)

    l1 = s1[0]
    l2 = s2[0]

    l1p, l2p = modify_svd_pair(l1, l2, watermark[i])

    s1p = np.array([l1p])
    s2p = np.array([l2p])

    X1p_em = u1 @ np.diag(s1p) @ v1
    X2p_em = u2 @ np.diag(s2p) @ v2
    
    X1p_em = np.squeeze(X1p_em)
    X2p_em = np.squeeze(X2p_em)
        
    X1[low:high] = X1p_em
    X2[low:high] = X2p_em
    
    x1_em = idct(X1, type=2, norm='ortho')
    x2_em = idct(X2, type=2, norm='ortho')
    
    frame_w = np.concatenate([x1_em, x2_em])

    watermarked_frames.append(frame_w)
  
  return np.concatenate(watermarked_frames)
  

def fsvc_watermark_detection(watermarked_signal, watermark_length=40, sr=16000):
  '''
  Parameters:
    watermarked_signal - 1D numpy array
    watermark_length - integer representing the length of the embedded watermark
    sr - sampling rate of the signal (default 16 kHz)
  Returns:
    detected watermark - 1D numpy array
  '''
  gamma1 = gamma1 * sr/44100
  gamma2 = gamma2 * sr/44100

  frames = np.array_split(watermarked_signal, watermark_length)
  watermark_bits = []

  for frame in frames:
    x1, x2 = np.array_split(frame, 2)

    X1 = dct(x1, type=2, norm='ortho')
    X2 = dct(x2, type=2, norm='ortho')

    low = int(gamma1*len(frame))
    high = int(gamma2*len(frame)+1)

    X1p = np.expand_dims(X1[low:high], axis=-1)
    X2p = np.expand_dims(X2[low:high], axis=-1)
    
    u1, s1, v1 = svd(X1p, full_matrices=False)
    u2, s2, v2 = svd(X2p, full_matrices=False)

    l1 = s1[0]
    l2 = s2[0]

    if l1/l2 < 1:
      watermark_bits.append(0)
    else:
      watermark_bits.append(1)

  return np.array(watermark_bits)
