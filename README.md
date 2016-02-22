# DEFT
A Discrete-Event Fourier Transform.

Discrete-Event signals are discontinuous and non-uniformly sampled, making them difficult to analyze using the standard Discrete or Fast Fourier Transforms. However, instead of trying to work around these problematic properties of Discrete-Event signals we can use them to our advantage to develop a transform method that is both more accurate and less computationally expensive than an equivalent Fast Fourier Transform of the Discrete-Event signal. Thus, instead of a Discrete-Time Fourier Transform (DTFT) we have a Discrete-Event Fourier Transform (DEFT). 

Included here is a Numpy-based Python implementation of the DEFT, and a brief technical note explaining how the DEFT works and why it might be useful. The implementation contains several different ways of computing an FT over a DE signal. Most are derived from earlier work by Bland *et al.* on FTs for non-uniformly sampled signals, and provide only a approximation to the frequency-domain representation of the signal. However, the semi-analytical *sinc*-based DEFT appears to be novel, and provides an accurate transform of a DE signal (up to quantization effects).
