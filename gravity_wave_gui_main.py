import pickle
import time
import sys
import os
from gravity_wave_gui_UI import Ui_MainWindow
from PyQt5.QtCore import * 
from PyQt5.QtWidgets import * 
from wavelet_hodo_utils import fitellipse_m1, fitellipse_m2, Gravity_Wave_Data, Wavelet, Hodograph, find_nearest_index
import warnings
import numpy as np

warnings.filterwarnings('ignore')

class MainWindow(Ui_MainWindow):
    
    def __init__(self, dialog):
        super(MainWindow, self).__init__()
        ####################################
        ''' Change this variable to adapt to your filesystem setup '''
        self.data_directory = 'EclipseData/'
        ####################################
        

        self.in_file_name = ''
        self.hodograph = ''
        self.wavelet = ''
        self.u_inverted = ''
        self.v_inverted = ''
        self.phi_stokes = ''
        self.omega_stokes = ''
        self.eta = ''
        self.ellipse_semi_major_axis_method_1 = ''
        self.ellipse_semi_minor_axis_method_1 = ''
        self.ellipse_angle_method_1 = ''
        self.ellipse_semi_major_axis_method_2 = ''
        self.ellipse_semi_minor_axis_method_2 = ''
        self.ellipse_angle_method_2 = ''
        self.raw_hodo_eta = ''
        self.raw_hodo_ellipse_semi_major_axis_method_1 = ''
        self.raw_hodo_ellipse_angle_method_1 = ''
        self.raw_hodo_ellipse_semi_major_axis_method_2 = ''
        self.raw_hodo_ellipse_semi_minor_axis_method_2 = ''
        self.raw_hodo_ellipse_angle_method_2 = ''
        self.g_wave_location = ''
        self.windowed_power_spectrum_for_stokes = ''
        self.bottom_scale_index = ''
        self.top_scale_index = ''
        self.bottom_alt_index = ''
        self.top_alt_index = ''
        self.windowed_spectrum_lower_scale_bound = ''
        self.windowed_spectrum_upper_scale_bound = ''
        self.windowed_spectrum_lower_altitude = ''
        self.windowed_spectrum_upper_altitude = ''
        self.windowed_alt = ''
        self.windowed_scales = ''
        self.windowed_power_spectrum = ''
        self.setupUi(dialog)
        self.start_analysis_button.clicked.connect(self.start_analysis)
        self.plot_altitude_window_button.clicked.connect(self.plot_windowed_wavelet_power_spectrum)
        self.display_quarter_max_power_surface_button.clicked.connect(self.display_three_quarter_max)
        self.get_gravity_wave_params_button.clicked.connect(self.get_gravity_wave_params_stokes_and_hodo)
        self.plot_hodograph_from_detransformed_data_button.clicked.connect(self.plot_detransformed_hodograph)
        self.plot_hodograph_altitude_window_button.clicked.connect(self.plot_raw_hodograph)
        self.fit_ellipses_to_wind_data_button.clicked.connect(self.fit_and_plot_raw_hodo)
        self.save_gravity_wave_params_button.clicked.connect(self.save_gravity_wave_params)

    def start_analysis(self):
        ''' Loads from the data file specified, creates hodograph and wavelet objects, plots wavelet power spectrum
        ''' 
        try:
            launch = self.file_to_load_from_line_edit.text()
            if launch != '':
                launch = launch.capitalize()
                for file in os.listdir(self.data_directory):
                    if launch in file:
                        self.in_file_name = file
                        break
            
            self.hodograph = Hodograph(self.data_directory + self.in_file_name)
            self.wavelet = Wavelet(self.data_directory + self.in_file_name)
            self.plot_wavelet_power_spectrum()

        except (IsADirectoryError, FileNotFoundError) as e:
            print("Please enter the correct filepath and filename")
 
    def plot_wavelet_power_spectrum(self):
        ''' Plots a power spectrum from the wavelet object on the wavelet canvas. '''
        self.wavelet_canvas.canvas.ax.clear()
        self.wavelet_canvas.canvas.ax.plot(self.wavelet.alt, self.wavelet.ucoi[:-1], 'k')
        self.wavelet_canvas.canvas.ax.set_ylim(ymin = self.wavelet.s0)
        self.wavelet_canvas.canvas.ax.set_title(self.in_file_name)
        self.wavelet_canvas.canvas.ax.contourf(self.wavelet.alt, self.wavelet.fourier_period[1:], self.wavelet.power_spectrum)
        self.wavelet_canvas.canvas.ax.set_ylabel("Scale (s)")
        self.wavelet_canvas.canvas.ax.set_xlabel("Altitude (m)")
        self.wavelet_canvas.canvas.draw()

    def get_gravity_wave_params_stokes_and_hodo(self):
        '''
        Prints gravity wave parameters 
        '''
        self.u_inverted = self.wavelet.invert_wavelet_coefficients(self.wavelet.u_wavelet, self.bottom_scale_index, self.top_scale_index, self.bottom_alt_index, self.top_alt_index)
        self.v_inverted = self.wavelet.invert_wavelet_coefficients(self.wavelet.v_wavelet, self.bottom_scale_index, self.top_scale_index, self.bottom_alt_index, self.top_alt_index)
        self.phi_stokes, self.omega_stokes = self.wavelet.direction_and_frequency_from_stokes_params(self.u_inverted, self.v_inverted)
        ellipse_angle_method_1 = self.ellipse_angle_method_1
        ellipse_angle_method_2 = self.ellipse_angle_method_2

        print("-----Stokes-----")
        print("AXR:", self.omega_stokes, "PHI:", np.rad2deg(self.phi_stokes))
        print("-----M1-----")
        print("AXR:", self.ellipse_semi_minor_axis_method_1/self.ellipse_semi_major_axis_method_1, 'PHI:', ellipse_angle_method_1, 'ETA:', self.eta)
        print("-----M2-----")
        print("AXR:", self.ellipse_semi_minor_axis_method_2/self.ellipse_semi_major_axis_method_2, 'PHI:', ellipse_angle_method_2, "ETA:", self.eta)
        print("------------")


    def fit_and_plot_ellipses(self, u, v, temp, alt, MplWidget, method = 'STOKES'):
        ''' Plots ellipses on hodograph canvas '''
        u_windowed = u.real
        v_windowed = v.real
        alt_windowed = alt
        temp_windowed = temp
        if method == 'STOKES':
            self.eta = self.hodograph.eta(u_windowed, temp_windowed, alt_windowed)
            z, self.ellipse_semi_major_axis_method_1, self.ellipse_semi_minor_axis_method_1, self.ellipse_angle_method_1 = fitellipse_m1(np.mat([u_windowed,v_windowed]))
            self.ellipse_angle_method_1 = np.rad2deg(self.ellipse_angle_method_1)
            theta = np.linspace(0, 2*np.pi, 100)
            x_ellipse_m1 = self.ellipse_semi_major_axis_method_1*np.cos(theta)*np.cos(self.ellipse_angle_method_1) - self.ellipse_semi_minor_axis_method_1*np.sin(theta)*np.sin(self.ellipse_angle_method_1) + z[0]
            y_ellipse_m1 = self.ellipse_semi_major_axis_method_1*np.cos(theta)*np.sin(self.ellipse_angle_method_1) + self.ellipse_semi_minor_axis_method_1*np.sin(theta)*np.cos(self.ellipse_angle_method_1) + z[1]

            self.ellipse_semi_major_axis_method_2, self.ellipse_semi_minor_axis_method_2, self.ellipse_angle_method_2, x_ellipse_m2, y_ellipse_m2 = fitellipse_m2(u_windowed, v_windowed)
            self.ellipse_angle_method_2 = np.rad2deg(self.ellipse_angle_method_2)
            MplWidget.canvas.ax.plot(x_ellipse_m1, y_ellipse_m1, 'k', label = 'M1')
            MplWidget.canvas.ax.plot(x_ellipse_m2, y_ellipse_m2, 'b', label = 'M2')
            MplWidget.canvas.ax.legend()

        elif method == 'HODOGRAPH':
            try:
                self.raw_hodo_eta = self.hodograph.eta(u_windowed, temp_windowed, alt_windowed)
                z, self.raw_hodo_ellipse_semi_major_axis_method_1, self.raw_hodo_ellipse_semi_minor_axis_method_1, self.raw_hodo_ellipse_angle_method_1 = fitellipse_m1(np.mat([u_windowed,v_windowed]))
                theta = np.linspace(0, 2*np.pi, 100)
                x_ellipse_m1 = self.raw_hodo_ellipse_semi_major_axis_method_1*np.cos(theta)*np.cos(self.raw_hodo_ellipse_angle_method_1) - self.raw_hodo_ellipse_semi_minor_axis_method_1*np.sin(theta)*np.sin(self.raw_hodo_ellipse_angle_method_1) + z[0]
                y_ellipse_m1 = self.raw_hodo_ellipse_semi_major_axis_method_1*np.cos(theta)*np.sin(self.raw_hodo_ellipse_angle_method_1) + self.raw_hodo_ellipse_semi_minor_axis_method_1*np.sin(theta)*np.cos(self.raw_hodo_ellipse_angle_method_1) + z[1]
                # Method 2
                self.raw_hodo_ellipse_semi_major_axis_method_2, self.raw_hodo_ellipse_semi_minor_axis_method_2, self.raw_hodo_ellipse_angle_method_2, x_ellipse_m2, y_ellipse_m2 = fitellipse_m2(u_windowed, v_windowed)
                MplWidget.canvas.ax.plot(x_ellipse_m1, y_ellipse_m1, 'k', label = 'M1')
                MplWidget.canvas.ax.plot(x_ellipse_m2, y_ellipse_m2, 'b', label = 'M2')
                MplWidget.canvas.ax.legend()
            
            except Exception as e:
                print(e)

    def display_three_quarter_max(self):
        ''' Plots 3/4 max from windowed power spectrum after Zink, 2001. Also plots a box around
         the window being viewed on the wavelet canvas '''
        try:
            self.wavelet_canvas.canvas.ax.clear()
            S = self.windowed_power_spectrum
            max_idx = np.where(S == np.max(S))
            scale = max_idx[0][0]; alt = max_idx[1][0]
            self.g_wave_location = alt
            three_quarter = 3*self.windowed_power_spectrum[max_idx]/4
            bottom = find_nearest_index(S[:scale, alt], three_quarter) #returns row index
            top = find_nearest_index(S[scale:, alt], three_quarter) #returns row index
            top += scale
            left = find_nearest_index(S[scale,:alt].T, three_quarter) #returns column index
            right = find_nearest_index(S[scale, alt:].T, three_quarter) #returns column index
            right += alt
            self.wavelet_canvas.canvas.ax.clear()
            self.wavelet_canvas.canvas.ax.plot(self.windowed_alt[alt], self.windowed_scales[scale], 'rx', ms = 3)
            self.wavelet_canvas.canvas.ax.contourf(self.windowed_alt[left:right], self.windowed_scales[bottom:top], self.windowed_power_spectrum[bottom:top, left:right])
            self.windowed_power_spectrum_for_stokes = self.windowed_power_spectrum[bottom:top, left:right]
            self.wavelet_canvas.canvas.ax.set_ylabel("Scale (s)")
            self.wavelet_canvas.canvas.ax.set_xlabel("Altitude (m)")
            s1, s2, a1, a2 = self.wavelet.return_window_indices(self.windowed_spectrum_lower_scale_bound, self.windowed_spectrum_upper_scale_bound, self.windowed_spectrum_lower_altitude, self.windowed_spectrum_upper_altitude)
            self.bottom_scale_index = bottom + s1; self.top_scale_index = top + s1 #for detransforming
            self.bottom_alt_index = left + a1; self.top_alt_index = right + a1
            self.wavelet_canvas.canvas.draw()

            if self.show_me_where_i_am.isChecked():
                self.hodograph_canvas.canvas.ax.clear()
                self.hodograph_canvas.canvas.ax.plot(self.wavelet.alt, self.wavelet.ucoi[:-1], 'k')
                self.hodograph_canvas.canvas.ax.set_ylim(ymin = self.wavelet.s0)
                self.hodograph_canvas.canvas.ax.set_title(self.in_file_name)
                self.hodograph_canvas.canvas.ax.contourf(self.wavelet.alt, self.wavelet.fourier_period[1:], self.wavelet.power_spectrum)
                bottom_alt_index = self.windowed_alt[left]; top_alt_index = self.windowed_alt[right]
                bottom_scale_index = self.windowed_scales[bottom]; top_scale_index = self.windowed_scales[top]
                ls1 = [bottom_alt_index, bottom_alt_index]
                ls2 = [bottom_scale_index, top_scale_index]
                self.hodograph_canvas.canvas.ax.plot(ls1, ls2, 'r-')
                ls1 = [top_alt_index, top_alt_index]
                ls2 = [bottom_scale_index,top_scale_index]
                self.hodograph_canvas.canvas.ax.plot(ls1, ls2, 'r-')
                ls1 = [bottom_alt_index, top_alt_index]
                ls2 = [top_scale_index, top_scale_index]
                self.hodograph_canvas.canvas.ax.plot(ls1, ls2, 'r-')
                ls1 = [bottom_alt_index, top_alt_index]
                ls2 = [bottom_scale_index, bottom_scale_index]
                self.hodograph_canvas.canvas.ax.plot(ls1, ls2, 'r-')
                self.hodograph_canvas.canvas.draw()
        except (TypeError) as e:
            print("Enter a valid window to analyze")



    def plot_raw_hodograph(self):
        try:
            if self.lower_altitude_bound_hodo_line_edit.text() is not '':
                lower_bound = float(self.lower_altitude_bound_hodo_line_edit.text())*1000
                upper_bound = float(self.upper_altitude_bound_hodo_line_edit.text())*1000
                arange = [lower_bound, upper_bound]
                u = self.wavelet.u
                v = self.wavelet.v
                alt = self.wavelet.alt
                temp = self.wavelet.temp
                u = self.get_altitude_window(arange, u, alt)
                v = self.get_altitude_window(arange, v, alt)
                temp = self.get_altitude_window(arange, temp, alt)
                alt = self.get_altitude_window(arange, alt, alt)
                self.hodograph_canvas.canvas.ax.clear()
                self.hodograph_canvas.canvas.ax.plot(u, v, 'rx', ms = 2)
                self.hodograph_canvas.canvas.ax.annotate(alt[-1],(u[-1], v[-1]))
                self.hodograph_canvas.canvas.ax.annotate(alt[0],(u[0], v[0]))
                self.hodograph_canvas.canvas.ax.set_title("Hodograph, raw data")
                self.hodograph_canvas.canvas.draw()
            else:
                u = self.wavelet.u[self.bottom_alt_index:self.top_alt_index]
                v = self.wavelet.v[self.bottom_alt_index:self.top_alt_index]
                alt = self.wavelet.alt[self.bottom_alt_index:self.top_alt_index]
                temp = self.wavelet.temp[self.bottom_alt_index:self.top_alt_index]
                self.hodograph_canvas.canvas.ax.clear()
                self.hodograph_canvas.canvas.ax.plot(u, v, 'rx', ms = 2)
                self.hodograph_canvas.canvas.ax.annotate(alt[-1],(u[-1], v[-1]))
                self.hodograph_canvas.canvas.ax.annotate(alt[0],(u[0], v[0]))
                self.hodograph_canvas.canvas.ax.set_title("Hodograph, raw data")
                self.hodograph_canvas.canvas.draw()

        except Exception as e:
            print(e)
    
    def plot_windowed_wavelet_power_spectrum(self):
        self.wavelet_canvas.canvas.ax.clear()
        try:
            self.windowed_spectrum_lower_scale_bound = float(self.lower_scale_bound_line_edit.text())
            self.windowed_spectrum_upper_scale_bound = float(self.upper_scale_bound_line_edit.text())
            self.windowed_spectrum_lower_altitude = float(self.lower_altitude_bound_line_edit.text())*1000 
            self.windowed_spectrum_upper_altitude = float(self.upper_altitude_bound_line_edit.text())*1000
            if self.windowed_spectrum_lower_altitude < min(self.wavelet.alt):
                self.windowed_spectrum_lower_altitude = min(self.wavelet.alt) 
            if self.windowed_spectrum_upper_altitude > max(self.wavelet.alt):
                self.windowed_spectrum_upper_altitude = max(self.wavelet.alt)
            self.windowed_alt, self.windowed_scales, self.windowed_power_spectrum = self.wavelet.return_windowed_arrays(self.windowed_spectrum_lower_scale_bound, self.windowed_spectrum_upper_scale_bound, self.windowed_spectrum_lower_altitude, self.windowed_spectrum_upper_altitude)
            self.wavelet_canvas.canvas.ax.set_title(self.in_file_name)
            self.wavelet_canvas.canvas.ax.contourf(self.windowed_alt, self.windowed_scales, self.windowed_power_spectrum)
            self.wavelet_canvas.canvas.draw()
            if self.show_me_where_i_am.isChecked():
                self.hodograph_canvas.canvas.ax.clear()
                self.hodograph_canvas.canvas.ax.plot(self.wavelet.alt, self.wavelet.ucoi[:-1], 'k')
                self.hodograph_canvas.canvas.ax.set_ylim(ymin = self.wavelet.s0)
                self.hodograph_canvas.canvas.ax.set_title(self.in_file_name)
                self.hodograph_canvas.canvas.ax.contourf(self.wavelet.alt, self.wavelet.fourier_period[1:], self.wavelet.power_spectrum)
                ls1 = [self.windowed_spectrum_lower_altitude, self.windowed_spectrum_lower_altitude]
                ls2 = [self.windowed_spectrum_lower_scale_bound, self.windowed_spectrum_upper_scale_bound]
                self.hodograph_canvas.canvas.ax.plot(ls1, ls2, 'r-')
                ls1 = [self.windowed_spectrum_upper_altitude, self.windowed_spectrum_upper_altitude]
                ls2 = [self.windowed_spectrum_lower_scale_bound, self.windowed_spectrum_upper_scale_bound]
                self.hodograph_canvas.canvas.ax.plot(ls1, ls2, 'r-')
                ls1 = [self.windowed_spectrum_lower_altitude, self.windowed_spectrum_upper_altitude]
                ls2 = [self.windowed_spectrum_lower_scale_bound, self.windowed_spectrum_lower_scale_bound]
                self.hodograph_canvas.canvas.ax.plot(ls1, ls2, 'r-')
                ls1 = [self.windowed_spectrum_lower_altitude, self.windowed_spectrum_upper_altitude]
                ls2 = [self.windowed_spectrum_upper_scale_bound, self.windowed_spectrum_upper_scale_bound]
                self.hodograph_canvas.canvas.ax.plot(ls1, ls2, 'r-')
                self.hodograph_canvas.canvas.draw()
        except (ValueError, TypeError, AttributeError) as e:
            print("Please enter a valid altitude window")

    def print_raw_hodo_params(self):
        print('---Raw Hodo Params---')
        print("-----M1-----")
        print("AXR:", self.raw_hodo_ellipse_semi_minor_axis_method_1/self.raw_hodo_ellipse_semi_major_axis_method_1, 'PHI:', 180*self.raw_hodo_ellipse_angle_method_1/np.pi, 'ETA:', self.raw_hodo_eta)
        print("-----M2-----")
        print("AXR:", self.raw_hodo_ellipse_semi_minor_axis_method_2/self.raw_hodo_ellipse_semi_major_axis_method_2, 'PHI:', 180*self.raw_hodo_ellipse_angle_method_2/np.pi, "ETA:", self.raw_hodo_eta)

    def fit_and_plot_raw_hodo(self):
        try:
            if self.lower_altitude_bound_hodo_line_edit.text() != '':
                lower_bound = float(self.lower_altitude_bound_hodo_line_edit.text())*1000
                upper_bound = float(self.upper_altitude_bound_hodo_line_edit.text())*1000
                arange = [lower_bound, upper_bound]
                u = self.wavelet.u
                v = self.wavelet.v
                alt = self.wavelet.alt
                temp = self.wavelet.temp
                u = self.get_altitude_window(arange, u, alt)
                v = self.get_altitude_window(arange, v, alt)
                temp = self.get_altitude_window(arange, temp, alt)
                alt = self.get_altitude_window(arange, alt, alt)
                self.hodograph_canvas.canvas.ax.clear()
                self.hodograph_canvas.canvas.ax.plot(u, v, 'rx', ms = 2)
                self.hodograph_canvas.canvas.ax.annotate(alt[-1],(u[-1],v[-1]))
                self.hodograph_canvas.canvas.ax.annotate(alt[0],(u[0],v[0]))
                self.hodograph_canvas.canvas.ax.set_title("Hodograph, raw data")
                self._fit_and_plot_ellipses(u, v, temp, alt, self.hodograph_canvas, method = 'HODOGRAPH')
                self.hodograph_canvas.canvas.draw()
                self._print_raw_hodo_params()
            else:
                self.hodograph_canvas.canvas.ax.clear()
                u = self.wavelet.u[self.bottom_alt_index:self.top_alt_index]
                v = self.wavelet.v[self.bottom_alt_index:self.top_alt_index]
                alt = self.wavelet.alt[self.bottom_alt_index:self.top_alt_index]
                temp = self.wavelet.temp[self.bottom_alt_index:self.top_alt_index]
                self._fit_and_plot_ellipses(u, v, temp, alt, self.hodograph_canvas, method = 'HODOGRAPH')
                self.hodograph_canvas.canvas.ax.plot(u, v, 'rx', ms = 2)
                self.hodograph_canvas.canvas.ax.annotate(alt[-1],(u[-1],v[-1]))
                self.hodograph_canvas.canvas.ax.annotate(alt[0],(u[0],v[0]))
                self.hodograph_canvas.canvas.ax.set_title("Hodograph, raw data")
                self.hodograph_canvas.canvas.draw()
                self._print_raw_hodo_params()

        except Exception as e:
            print(e)

    def plot_detransformed_hodograph(self):
        self.hodograph_canvas.canvas.ax.clear()
        uinv = self.wavelet.invert_wavelet_coefficients(self.wavelet.u_wavelet, self.bottom_scale_index, self.top_scale_index, self.bottom_alt_index, self.top_alt_index)
        vinv = self.wavelet.invert_wavelet_coefficients(self.wavelet.v_wavelet, self.bottom_scale_index, self.top_scale_index, self.bottom_alt_index, self.top_alt_index)
        alt = self.wavelet.alt[self.bottom_alt_index:self.top_alt_index]
        temp = self.wavelet.temp[self.bottom_alt_index:self.top_alt_index]
        self.fit_and_plot_ellipses(uinv, vinv, temp, alt, self.hodograph_canvas)
        self.hodograph_canvas.canvas.ax.plot(uinv, vinv, 'rx', ms = 2)
        self.hodograph_canvas.canvas.ax.annotate(alt[-1], (uinv[-1], vinv[-1]))
        self.hodograph_canvas.canvas.ax.annotate(alt[0], (uinv[0], vinv[0]))
        self.hodograph_canvas.canvas.ax.set_title("Hodograph, extent of wavepacket")
        self.hodograph_canvas.canvas.draw()

    def get_altitude_window(self, arange, data_array, alt_array):
        data_windowed = [data_array[i] for i, a in enumerate(alt_array) if a > arange[0] and a < arange[1]]
        return np.asarray(data_windowed)

    def save_gravity_wave_params(self):
        if self.file_to_save_to_line_edit.text() is not '':
            fname = self.file_to_save_to_line_edit.text()
            if self.eta < 0: 
                self.phi_stokes += 180
                self.ellipse_angle_method_1 += 180
                self.ellipse_angle_method_2 += 180
            if self.save_method_1_cbox.isChecked():
                axr_hodo = self.ellipse_semi_minor_axis_method_1/self.ellipse_semi_major_axis_method_1
                phi_hodo = self.ellipse_angle_method_1
            if self.save_method_2_cbox.isChecked():
                axr_hodo = self.ellipse_semi_minor_axis_method_2/self.ellipse_semi_major_axis_method_2
                phi_hodo = self.ellipse_angle_method_2
            if self.save_method_2_cbox.isChecked() or self.save_method_1_cbox.isChecked():
                loc = self.g_wave_location
                gwave = Gravity_Wave_Data()
                gwave.add_row_of_data(self.in_file_name[8:10], self.omega_stokes, axr_hodo, self.phi_stokes, phi_hodo, loc, 
                    self.bottom_scale_index, self.top_scale_index, self.bottom_alt_index, self.top_alt_index)
                gwave.dump_data(self.file_to_save_to_line_edit.text())
            elif not self.save_method_2_cbox.isChecked() or not self.save_method_1_cbox.isChecked():
                print("Please select an ellipse fit method to save.")
        else:
            print("Please enter a filename.")


if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    form = QMainWindow()
    m_gui = MainWindow(form)
    form.show()
    app.exec_()



