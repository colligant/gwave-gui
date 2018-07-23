import ellipse_fit_method1 as method_1
import ellipse_fit_method2 as method_2
from wavelet_functions import wavelet, wave_signif
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Data_Methods:

    def __init__(self):
        self.order = 2

    def _remove_background(self, array, time):
        ''' Removes a polynomial fit from whatever time series 
            'array' is.
        '''

        order = self.order
        fit = np.poly1d(np.polyfit(time, array, order))
        return array - fit(time)


    def _get_data(self, data_file):
        self.df = pd.read_table(data_file, sep = '\t', skiprows = 18, encoding = 'latin1', skipfooter = 10, engine = 'python')
        self.df = self.df.drop([0]) # drops the row that holds units of measured quantities
        self.df.columns = self.df.columns.str.replace(' ', '')
        alt = np.asarray(pd.to_numeric(self.df['Alt']))
        max_alt_index = list(alt).index(max(alt))
        alt = alt[:max_alt_index]
        time = np.asarray(pd.to_numeric(self.df['Time'])[:max_alt_index])
        ws = np.asarray(pd.to_numeric(self.df['Ws'])[:max_alt_index])
        wd = np.asarray(pd.to_numeric(self.df['Wd'])[:max_alt_index])
        u = ws*np.cos(np.deg2rad(wd))
        v = ws*np.sin(np.deg2rad(wd))
        temp = np.asarray(pd.to_numeric(self.df['T'])[:max_alt_index])
        t = self._remove_background(temp, time)
        u = self._remove_background(u, time)
        v = self._remove_background(v, time)
        return alt, time, u, v, temp


    @staticmethod
    def _find_nearest_index(array, value):
        idx = (np.abs(array-value)).argmin()
        return idx
    

    @staticmethod
    def eta(u, temp, alt):
        ''' Comutes 'eta' over an altitude range.
            From Marlton, Williams, and Nicoll, 2015 
        '''
        dtdz = np.diff(temp)/np.diff(alt)
        eta = np.mean(u[:-1]*dtdz)
        return eta


    @staticmethod
    def _get_altitude_window(arange, data_array, alt_array):
        data_windowed = [data_array[i] for i, a in enumerate(alt_array) if a > arange[0] and a < arange[1]]
        return np.asarray(data_windowed)


class Gravity_Wave_Data:

    ''' 
    Gravity wave parameters:
    w/f
    propagation direction
    altitude of maximum power in the wavelet power spectrum
    '''
    def __init__(self):
        self.row_index = 0
        self.columns = ['launch', 'wf_stokes', 'wf_hodograph', 'prop_dir_stokes', 'prop_dir_hodograph', 'altitude', 'bottom_scale', 'top_scale', 'bottom_alt', 'top_alt']
        self.data_array = []

    def add_row_of_data(self, launch, wf_stokes, wf_hodograph, prop_dir_stokes, prop_dir_hodograph, altitude, bottom_scale, top_scale, bottom_alt, top_alt):
        data_row = np.array([launch, wf_stokes, wf_hodograph, prop_dir_stokes, prop_dir_hodograph, altitude, bottom_scale, top_scale, bottom_alt, top_alt])
        self.data_array.append(data_row)

    
    def dump_data(self, out_file):
        try:
            df = pd.DataFrame(self.data_array, columns = self.columns)
            df.to_pickle(out_file)
        
        except Exception as e:
            print(e)


class Wavelet(Data_Methods):

    def __init__(self, data_file):
        super(Wavelet, self).__init__()
        self.data_file = data_file
        self.mother = 'MORLET' # wavelet choice to use.
        self.pad = 1
        self.lag1 = 0.72
        self.alt, self.time, u, v, self.temp = self._get_data(data_file)
        self.dj = 0.01
        self.dt = np.diff(self.time)[0]
        self.s0 = 2*self.dt
        self.j1 = np.log2(len(u)*self.dt/self.s0)/self.dj
        self.u = self._remove_background(u, self.time)
        self.v = self._remove_background(v, self.time)
        self.u_wavelet, self.fourier_period, self.scale, self.ucoi = \
        wavelet(u, self.dt, self.pad, self.dj, self.s0, self.j1, self.mother)
        self.v_wavelet, self.fourier_period, self.scale, self.vcoi = \
        wavelet(v, self.dt, self.pad, self.dj, self.s0, self.j1, self.mother)

        self.power_spectrum = abs(self.u_wavelet)**2 + abs(self.v_wavelet)**2


    def invert_wavelet_coefficients(self, wavelet_coeff_array, scale_index_1, scale_index_2, alt_index_1, alt_index_2):
        ''' Takes an MxN array of wavelet coefficients and inverts it
            within the bounding box specified by scale and alt indices.
            Assumes rows are wavelet scale values, and columns the altitudes.
            Algorithm taken from Torrence and Compo (1998).
        '''
        scale_index_1 = int(scale_index_1); scale_index_2 = int(scale_index_2)
        alt_index_1 = int(alt_index_1); alt_index_2 = int(alt_index_2)
        j = np.arange(0, self.j1+1)

        a = self.dj*np.sqrt(self.dt)/(0.776) # magic number from Torrence and Compo
        b = a/(np.pi**(-1/4))

        scales = self.s0 * 2. ** (j * self.dj)
        scales_windowed = scales[scale_index_1:scale_index_2]
        wavelet_coeff_array_windowed = wavelet_coeff_array[scale_index_1:scale_index_2,alt_index_1:alt_index_2]
       
        x = []
        for col in range(wavelet_coeff_array_windowed.shape[1]):
            xn = np.sum(((wavelet_coeff_array_windowed[:, col]))/(np.sqrt(scales_windowed)))
            x.append(xn*b)

        return np.asarray(x)

    def plot_power_surface(self, title = False, levels = False):
        fig, ax = plt.subplots()
        ax.plot(self.alt, self.ucoi[:-1], 'k')
        ax.invert_yaxis()
        ax.set_ylim(ymax = self.s0)
        ax.semilogy(basey = 2)
        if title:
            ax.set_title(self.data_file)
        cf = ax.contourf(self.alt, self.fourier_period[1:], self.power_spectrum)
        fig.colorbar(cf)

 
    def windowed_data(self, arange):
        u_windowed = self._get_altitude_window(arange, self.u, self.alt)
        v_windowed = self._get_altitude_window(arange, self.v, self.alt)
        alt_windowed = self._get_altitude_window(arange, self.alt, self.alt)
        return u_windowed, v_windowed, alt_windowed

    def return_windowed_arrays(self, s1, s2, a1, a2):
        sidx1 = self._find_nearest_index(self.fourier_period, s1)
        sidx2 = self._find_nearest_index(self.fourier_period, s2)
        aidx1 = self._find_nearest_index(self.alt, a1)
        aidx2 = self._find_nearest_index(self.alt, a2)
        return self.alt[aidx1:aidx2], self.fourier_period[sidx1:sidx2], self.power_spectrum[sidx1:sidx2, aidx1:aidx2]

    def return_window_indices(self, s1, s2, a1, a2):
        sidx1 = self._find_nearest_index(self.fourier_period, s1)
        sidx2 = self._find_nearest_index(self.fourier_period, s2)
        aidx1 = self._find_nearest_index(self.alt, a1)
        aidx2 = self._find_nearest_index(self.alt, a2)
        return sidx1, sidx2, aidx1, aidx2

    @staticmethod
    def direction_and_frequency_from_stokes_params(u_wavelet_inverted, v_wavelet_inverted):
        '''Calculates Stokes' parameters from de-transformed wind data
        '''
        u = u_wavelet_inverted; v = v_wavelet_inverted
        
        I = np.mean(u.real**2) + np.mean(u.imag**2) + np.mean(v.real**2) + np.mean(v.imag**2) 
        
        D = np.mean(u.real**2) + np.mean(u.imag**2) - np.mean(v.real**2) - np.mean(v.imag**2) 
        
        P = 2*(np.mean(u.real*v.real) + np.mean(u.imag*v.imag))
        
        Q = 2*(np.mean(u.real*v.imag) - np.mean(u.imag*v.real))
        
        phi = 0.5*np.arctan2(P, D)

        d = np.sqrt(D**2 + P**2 + Q**2)
        
        arg = Q/(d)
        
        xi = np.arcsin(arg)/2
        
        omega_over_f = (np.tan(xi))

        return phi, omega_over_f

pi = np.pi
cos = np.cos
sin = np.sin


class Hodograph(Data_Methods):

    def __init__(self, data_file):
        super(Hodograph, self).__init__()
        self.alt, self.time, self.u, self.v, self.temp = self._get_data(data_file)

    def plot_hodograph(self, arange):
        ''' 
            Plots and shows hodograph in desired altitude range.
        '''
        u_windowed = self._get_altitude_window(arange, self.u, self.alt)
        v_windowed = self._get_altitude_window(arange, self.v, self.alt)
        alt_windowed = self._get_altitude_window(arange, self.alt, self.alt)

        fig, ax = plt.subplots()
        ax.annotate(alt_windowed[0],(u_windowed[0], v_windowed[0]))    
        ax.annotate(alt_windowed[-1],(u_windowed[-1],v_windowed[-1]))
        ax.plot(u_windowed, v_windowed, 'rx', ms = 1)
        plt.show()

    def window_data(self, arange):
        u_windowed = self._get_altitude_window(arange, self.u, self.alt)
        v_windowed = self._get_altitude_window(arange, self.v, self.alt)
        alt_windowed = self._get_altitude_window(arange, self.alt, self.alt)
        return u_windowed, v_windowed, alt_windowed


    def fit_and_plot_ellipses(self, arange):

        u_windowed = self._get_altitude_window(arange, self.u, self.alt)
        v_windowed = self._get_altitude_window(arange, self.v, self.alt)
        alt_windowed = self._get_altitude_window(arange, self.alt, self.alt)
        temp_windowed = self._get_altitude_window(arange, self.temp, self.alt)

        # Method 1

        z, self.a1, self.b1, self.phi1 = method_1.fitellipse(np.mat([u_windowed,v_windowed]))
        theta = np.linspace(0, 2*pi, 100)
        x_ellipse_m1 = self.a1*cos(theta)*cos(self.phi1) - self.b1*sin(theta)*sin(self.phi1) + z[0]
        y_ellipse_m1 = self.a1*cos(theta)*sin(self.phi1) + self.b1*sin(theta)*cos(self.phi1) + z[1]

        fig, ax = plt.subplots()
        ax.set_title(r"Ellipse fit, method 1, \eta =", self.eta(u_windowed, temp_windowed, alt_windowed))
        ax.plot(u_windowed, v_windowed, 'rx', u_windowed, v_windowed, 'r.-', alpha = 0.5)
        ax.plot(x_ellipse_m1, y_ellipse_m1, 'k')

        # Method 2

        self.a2, self.b2, self.phi2, x_ellipse_m2, y_ellipse_m2 = method_2.fitellipse(u_windowed,v_windowed)
        
        fig, ax = plt.subplots()
        ax.set_title(r"Ellipse fit, method 2, \eta =", self.eta(u_windowed, temp_windowed, alt_windowed))
        ax.plot(u_windowed, v_windowed, 'rx', u_windowed, v_windowed, 'r.-', alpha = 0.5)
        ax.plot(x_ellipse_m2, y_ellipse_m2, 'k')

        plt.show()











