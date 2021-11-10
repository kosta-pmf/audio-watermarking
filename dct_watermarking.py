import numpy as np
from scipy.fftpack import dct, idct
import copy

def get_energy(C):
  return np.sum(np.square(C))

def get_band_representative_frequency(band_index, band_size, num_coeffs):
  start_freq = band_index*band_size*sr/(2*num_coeffs)
  end_freq = (band_index+1)*band_size*sr/(2*num_coeffs)

  return (start_freq+end_freq)/2

def get_band_masking_energy(C, band_index, band_size, num_coeffs):

  freq = get_band_representative_frequency(band_index, band_size, num_coeffs)

  bark_scale_freq = 13*np.arctan(0.00076*freq)+3.5*np.arctan((freq/7500)**2)

  a_tmn = -0.275*bark_scale_freq-15.025

  return 10**(a_tmn/10)*get_energy(C)

def divide_band_into_groups(C, lG1):

  choice = np.random.choice(range(C.shape[0]), size=(lG1,), replace=False)
  rest = np.array([i for i in range(C.shape[0]) if i not in choice])

  return np.sort(choice), np.sort(rest)

def energy_compensation(C, G2_ind, niT):
  C_hat = C[:]

  if niT < 0:
    for ind in G2_ind:
      if C[ind]>=0:
        C_hat[ind] = (C[ind]**2-niT/lG2)**(1/2)
      else:
        C_hat[ind] = -(C[ind]**2-niT/lG2)**(1/2)

  elif niT > 0:
    ni = niT
    G2_ind_sorted = sorted(list(G2_ind), key=lambda x: abs(C[x]))
    for k, ind in enumerate(G2_ind_sorted):
      if C[ind]>=0:
        C_hat[ind] = (max(0, C[ind]**2-ni/(lG2-(k+1))))**(1/2)
      else:
        C_hat[ind] = -(max(0, C[ind]**2-ni/(lG2-(k+1))))**(1/2)

      ni = ni - (C[ind]**2 - C_hat[ind]**2)

  return C_hat

def embed_bits_in_band(C, watermark_bits, lv, band_index, band_size, lG1, num_coeffs):
  delta = np.sqrt(get_band_masking_energy(C, band_index, band_size, num_coeffs))
  delta_sigma = np.sqrt(lv)*delta

  G1_ind, G2_ind = divide_band_into_groups(C, lG1)
  C_hat = C[:]
  for i, ind in enumerate(G1_ind):
    wb = watermark_bits[i]
    if wb==0:
        C_hat[ind] = np.floor(C[ind]/delta+0.5)*delta
    else:
        C_hat[ind] = np.floor(C[ind]/delta)*delta+delta/2

  niT = np.sum([C_hat[k] for k in G1_ind]) - np.sum([C[k] for k in G1_ind])

  C_hat = energy_compensation(C, G2_ind, niT)
  return C_hat, G1_ind

def embed_bits_in_frame(C, watermark_bits, band_size, lG1):
  band1 = C[:band_size]
  C_hat = C[:]
  C_hat[:band_size], G1_ind1 = embed_bits_in_band(band1, watermark_bits, 1, 0, band_size, lG1, len(C))

  return C_hat, G1_ind1

def smooth_transitions(original_frames, watermarked_frames, lf, lt):

  alphas = np.zeros(len(original_frames))
  bethas = np.zeros(len(original_frames))

  for n in range(len(original_frames)):
    alphas[n] = watermarked_frames[n][lt] - original_frames[n][lt] # the first modified sample
    bethas[n] = watermarked_frames[n][lf-1] - original_frames[n][lf-1] # the last modified sample
    for k in range(lt):
      if k==0:
        watermarked_frames[n][k] = original_frames[n][k] + alphas[n]*(k+1)/(lt+1)
      else:
        watermarked_frames[n][k] = original_frames[n][k] + bethas[n-1] + (alphas[n]-bethas[n-1])*(k+1)/(lt+1)

  return watermarked_frames

def dctb1_watermark_embedding(signal, watermark, sr=16000, lt=23, lw=1486, band_size=30, lG1=24, lG2=6):
  '''
  Parameters:
    signal - 1D numpy array
    watermark - 1D numpy array of ones and zeros of length band_size*lG1
    sr - sampling rate
    lt - number of transition samples in a frame
    lw - number of samples for embedding in a frame
    band_size - number of DCT coefficients in a band. It must be equal to lG1+lG2
    lG1 - number of DCT coefficients in group G1
    lG2 - number of DCT coefficients in group G2
  Returns:
    watermarked signal - 1D numpy array
    indices of G1 coefficients for each frame (a secret key for watermark detection) - list of 1D numpy arrays
  '''
  lf = lt+lw
  signal_length = len(signal)
  num_frames = signal_length//lf

  bits_per_frame = lG1
  if len(watermark) != num_frames*bits_per_frame:
    print("Watermark length should be: ", num_frames*bits_per_frame)
    return None, None

  if band_size != lG1+lG2:
    print("Band size should be: ", lG1+lG2)
    return None, None

  frames = np.array_split(signal[:(num_frames*lf)], num_frames)
  rframes = []

  G1_inds = []

  watermarked_signal = copy.deepcopy(signal)
  for ind, frame in enumerate(frames):
    C = dct(frame[lt:], norm="ortho")
    C_hat, G1_ind = embed_bits_in_frame(C, watermark[(ind*bits_per_frame):((ind+1)*bits_per_frame)], band_size, lG1)
    G1_inds.append(G1_ind)

    rframe = np.zeros(frame.shape)
    rframe[:lt] = frame[:lt]
    rframe[lt:] = idct(C_hat, norm="ortho")
    rframes.append(rframe)

  rframes = smooth_transitions(frames, rframes, lf, lt)
  watermarked_signal[:(num_frames*lf)] = np.concatenate(rframes)
  return watermarked_signal, G1_inds


def dctb1_watermark_detection(watermarked_signal, G1_inds, lt=23, lw=1486, band_size=30):
  '''
  Parameters:
    watermarked_signal - 1D numpy array
    G1_inds - indices of G1 coefficients in each frame
    lt - number of transition samples in a frame
    lw - number of samples for embedding in a frame
    band_size - number of DCT coefficients in a band
  Returns:
    detected watermark - 1D numpy array
  '''
  lf = lt+lw
  signal_length = len(signal)
  num_frames = signal_length//lf
  frames = np.array_split(watermarked_signal[:(num_frames*lf)], num_frames)
  watermark_bits = []

  for ind, frame in enumerate(frames):
    C = dct(frame[lt:], norm="ortho")
    band1 = C[:band_size]

    delta = np.sqrt(get_band_masking_energy(band1, 0, band_size, lw))

    for k in G1_inds[ind]:
      if abs(C[k]/delta-np.floor(C[k]/delta)-0.5) < 0.25:
        watermark_bits.append(1)
      else:
        watermark_bits.append(0)

  return np.array(watermark_bits)
