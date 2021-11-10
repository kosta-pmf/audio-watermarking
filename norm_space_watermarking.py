import numpy as np
import pywt
from scipy.fftpack import dct, idct


def norm_space_watermark_embedding(signal, watermark, delta=0.03):
  '''
  Parameters:
    signal - 1D numpy array
    watermark - 1D numpy array of ones and zeros
    delta - parameter of the algorithm controlling watermark imperceptibility and robustness. Higher delta makes algorithm more robust, but lowers audio signal quality.
  Returns:
    watermarked signal - 1D numpy array
  '''

  segments = np.array_split(signal, len(watermark))
  rsegments = []

  for ind, segment in enumerate(segments):

    cA1, cD1 = pywt.dwt(segment, 'db1')

    v = dct(cA1, norm='ortho')

    v1 = v[::2]
    v2 = v[1::2]

    nrmv1 = np.linalg.norm(v1, ord=2)
    nrmv2 = np.linalg.norm(v2, ord=2)

    u1 = v1/nrmv1
    u2 = v2/nrmv2

    watermark_bit = watermark[ind]
    nrm = (nrmv1+nrmv2)/2
    if watermark_bit == 1:
      nrmv1 = nrm + delta
      nrmv2 = nrm - delta
    else:
      nrmv1 = nrm - delta
      nrmv2 = nrm + delta

    rv1 = nrmv1*u1
    rv2 = nrmv2*u2

    rv = np.zeros((len(v),))

    rv[::2] = rv1
    rv[1::2] = rv2

    rcA1 = idct(rv, norm='ortho')

    rseg = pywt.idwt(rcA1, cD1, 'db1')
    rsegments.append(rseg[:])

  return np.concatenate(rsegments)


def norm_space_watermark_detection(watermarked_signal, watermark_length=512, delta=0.03):
  '''
  Parameters:
    watermarked_signal - 1D numpy array
    watermark_length - integer representing the length of the embedded watermark
    delta - value of the delta parameter used for watermark embedding
  Returns:
    detected watermark - 1D numpy array
  '''
  segments = np.array_split(watermarked_signal, watermark_length)
  watermark_bits = []

  for ind, segment in enumerate(segments):
    cA1, cD1 = pywt.dwt(segment, 'db1')

    v = dct(cA1, norm='ortho')

    v1 = v[::2]
    v2 = v[1::2]

    nrmv1 = np.linalg.norm(v1, ord=2)
    nrmv2 = np.linalg.norm(v2, ord=2)

    if nrmv1 > nrmv2:
      watermark_bits.append(1.0)
    else:
      watermark_bits.append(0.0)

  return np.array(watermark_bits)
