# gravity-wave-analysis-gui
Python software for analyzing gravity waves in radiosonde profiles.

Wavelet analysis of gravity waves based on Zink and Vincent, 2001 and Serafimovich, 2004. 

The program takes GRAW-DFM09 radiosonde profile data as input, and performs the wavelet transform on the zonal and meridional wind components. The resulting power spectrums are added in quadrature to produce a wavelet power spectrum. The power spectrums are then scanned for local maxima manually by entering values in the altitude/scale range boxes. When a local maxima is centered on, the user can click "display one-quarter max" and see the gravity wave in the power spectrum. After getting the 1/4-max, the associated hodograph can be plotted from the de-transformed wind components.

Parameters are then saved by clicking the "save" button. They are serialized by python's pickle package into a Gravity-Wave-Data object, which can be accessed for further analysis. A GUI to analyze the gravity waves further is forthcoming.

