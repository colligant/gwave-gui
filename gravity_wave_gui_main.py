from gravity_wave_gui_UI import Ui_MainWindow
import sys
from PyQt5.QtCore import * 
from PyQt5.QtWidgets import * 
from wavelet_hodo_utils import *
import warnings
warnings.filterwarnings('ignore')
import numpy as np



class MainWindow(Ui_MainWindow):
    
    def __init__(self, dialog):

        super(MainWindow, self).__init__()
        self.setupUi(dialog)
        self.start_analysis_button.clicked.connect(self._start_analysis)
        self.plot_altitude_window_button.clicked.connect(self._plot_windowed_wavelet_power_spectrum)
        self.display_quarter_max_power_surface_button.clicked.connect(self._display_quarter_max)
        self.get_gravity_wave_params_button.clicked.connect(self._get_gravity_wave_params_stokes_and_hodo)
        self.plot_hodograph_from_detransformed_data_button.clicked.connect(self._plot_detransformed_hodograph)
        self.plot_hodograph_altitude_window_button.clicked.connect(self._plot_raw_hodograph)
        self.fit_ellipses_to_wind_data_button.clicked.connect(self._fit_and_plot_raw_hodo)
        self.save_gravity_wave_params_button.clicked.connect(self._save_gravity_wave_params)


    def _start_analysis(self):
        self.in_file_name = self.file_to_load_from_line_edit.text()
        try:
            if self.in_file_name is not '':
                self.hodograph = Hodograph('EclipseData/' + self.in_file_name)
                self.wavelet = Wavelet('EclipseData/' + self.in_file_name)
                self._plot_wavelet_power_spectrum()

        except Exception as e:
            print(e)

    def _plot_wavelet_power_spectrum(self):
        ''' Plots a power spectrum from the wavelet object. '''
        self.wavelet_canvas.canvas.ax.clear()
        self.wavelet_canvas.canvas.ax.plot(self.wavelet.alt, self.wavelet.ucoi[:-1], 'k')
        self.wavelet_canvas.canvas.ax.set_ylim(ymin = self.wavelet.s0)
        self.wavelet_canvas.canvas.ax.set_title(self.in_file_name)
        self.wavelet_canvas.canvas.ax.contourf(self.wavelet.alt, self.wavelet.fourier_period[1:], self.wavelet.power_spectrum)
        self.wavelet_canvas.canvas.draw()

    def _get_gravity_wave_params_stokes_and_hodo(self):
        try:
            self.uinv = self.wavelet.invert_wavelet_coefficients(self.wavelet.u_wavelet, self.bottom_scale, self.top_scale, self.bottom_alt, self.top_alt)
            self.vinv = self.wavelet.invert_wavelet_coefficients(self.wavelet.v_wavelet, self.bottom_scale, self.top_scale, self.bottom_alt, self.top_alt)
            self.phi_stokes, self.omega_stokes = self.wavelet.direction_and_frequency_from_stokes_params(self.uinv, self.vinv)
            phi1 = self.phi1
            phi2 = self.phi2
            print("-----Stokes-----")
            print("AXR:", self.omega_stokes, "PHI:", np.rad2deg(self.phi_stokes))
            print("-----M1-----")
            print("AXR:", self.b1/self.a1, 'PHI:', phi1, 'ETA:', self.eta)
            print("-----M2-----")
            print("AXR:", self.b2/self.a2, 'PHI:', phi2, "ETA:", self.eta)
            print("------------")

        except Exception as e:
            print(e)

    def _fit_and_plot_ellipses(self, u, v, temp, alt, MplWidget, method = 'STOKES'):
        u_windowed = u.real
        v_windowed = v.real
        alt_windowed = alt
        temp_windowed = temp
        # Method 1
        if method == 'STOKES':
            try:
                self.eta = self.hodograph.eta(u_windowed, temp_windowed, alt_windowed)
                z, self.a1, self.b1, self.phi1 = method_1.fitellipse(np.mat([u_windowed,v_windowed]))
                self.phi1 = np.rad2deg(self.phi1)
                theta = np.linspace(0, 2*np.pi, 100)
                x_ellipse_m1 = self.a1*np.cos(theta)*np.cos(self.phi1) - self.b1*np.sin(theta)*np.sin(self.phi1) + z[0]
                y_ellipse_m1 = self.a1*np.cos(theta)*np.sin(self.phi1) + self.b1*np.sin(theta)*np.cos(self.phi1) + z[1]
                # Method 2
                self.a2, self.b2, self.phi2, x_ellipse_m2, y_ellipse_m2 = method_2.fitellipse(u_windowed, v_windowed)
                self.phi2 = np.rad2deg(self.phi2)
                MplWidget.canvas.ax.plot(x_ellipse_m1, y_ellipse_m1, 'k', label = 'M1')
                MplWidget.canvas.ax.plot(x_ellipse_m2, y_ellipse_m2, 'b', label = 'M2')
                MplWidget.canvas.ax.legend()
            except Exception as e:
                print(e)

        elif method == 'HODOGRAPH':
            try:
                self.raw_hodo_eta = self.hodograph.eta(u_windowed, temp_windowed, alt_windowed)
                z, self.raw_hodo_a1, self.raw_hodo_b1, self.raw_hodo_phi1 = method_1.fitellipse(np.mat([u_windowed,v_windowed]))
                theta = np.linspace(0, 2*np.pi, 100)
                x_ellipse_m1 = self.raw_hodo_a1*np.cos(theta)*np.cos(self.raw_hodo_phi1) - self.raw_hodo_b1*np.sin(theta)*np.sin(self.raw_hodo_phi1) + z[0]
                y_ellipse_m1 = self.raw_hodo_a1*np.cos(theta)*np.sin(self.raw_hodo_phi1) + self.raw_hodo_b1*np.sin(theta)*np.cos(self.raw_hodo_phi1) + z[1]
                # Method 2
                self.raw_hodo_a2, self.raw_hodo_b2, self.raw_hodo_phi2, x_ellipse_m2, y_ellipse_m2 = method_2.fitellipse(u_windowed, v_windowed)
                MplWidget.canvas.ax.plot(x_ellipse_m1, y_ellipse_m1, 'k', label = 'M1')
                MplWidget.canvas.ax.plot(x_ellipse_m2, y_ellipse_m2, 'b', label = 'M2')
                MplWidget.canvas.ax.legend()
            except Exception as e:
                print(e)

    def _display_quarter_max(self):
        try:
            self.wavelet_canvas.canvas.ax.clear()
            S = self.windowed_power_spectrum
            max_idx = np.where(S == np.max(S))
            scale = max_idx[0][0]; alt = max_idx[1][0]
            self.g_wave_location = alt
            one_quarter = self.windowed_power_spectrum[max_idx]/4
            one_quarter *= 3
            bottom = self.wavelet._find_nearest_index(S[:scale, alt], one_quarter) #returns row index
            top = self.wavelet._find_nearest_index(S[scale:, alt], one_quarter) #returns row index
            top += scale
            left = self.wavelet._find_nearest_index(S[scale,:alt].T, one_quarter) #returns column index
            right = self.wavelet._find_nearest_index(S[scale, alt:].T, one_quarter) #returns column index
            right += alt
            self.wavelet_canvas.canvas.ax.clear()
            self.wavelet_canvas.canvas.ax.plot(self.windowed_alt[alt], self.windowed_scales[scale], 'rx', ms = 3)
            self.wavelet_canvas.canvas.ax.contourf(self.windowed_alt[left:right], self.windowed_scales[bottom:top], self.windowed_power_spectrum[bottom:top, left:right])
            self.windowed_power_spectrum_for_stokes = self.windowed_power_spectrum[bottom:top, left:right]
            
            s1, s2, a1, a2 = self.wavelet.return_window_indices(self.scale_1, self.scale_2, self.alt_1, self.alt_2)
            self.bottom_scale = bottom + s1; self.top_scale = top + s1 #for detransforming
            self.bottom_alt = left + a1; self.top_alt = right + a1
            self.wavelet_canvas.canvas.draw()

            if self.show_me_where_i_am.isChecked():
                self.hodograph_canvas.canvas.ax.clear()
                self.hodograph_canvas.canvas.ax.plot(self.wavelet.alt, self.wavelet.ucoi[:-1], 'k')
                self.hodograph_canvas.canvas.ax.set_ylim(ymin = self.wavelet.s0)
                self.hodograph_canvas.canvas.ax.set_title(self.in_file_name)
                self.hodograph_canvas.canvas.ax.contourf(self.wavelet.alt, self.wavelet.fourier_period[1:], self.wavelet.power_spectrum)
                bottom_alt = self.windowed_alt[left]; top_alt = self.windowed_alt[right]
                bottom_scale = self.windowed_scales[bottom]; top_scale = self.windowed_scales[top]
                ls1 = [bottom_alt, bottom_alt]
                ls2 = [bottom_scale, top_scale]
                self.hodograph_canvas.canvas.ax.plot(ls1, ls2, 'r-')
                ls1 = [top_alt, top_alt]
                ls2 = [bottom_scale,top_scale]
                self.hodograph_canvas.canvas.ax.plot(ls1, ls2, 'r-')
                ls1 = [bottom_alt, top_alt]
                ls2 = [top_scale, top_scale]
                self.hodograph_canvas.canvas.ax.plot(ls1, ls2, 'r-')
                ls1 = [bottom_alt, top_alt]
                ls2 = [bottom_scale, bottom_scale]
                self.hodograph_canvas.canvas.ax.plot(ls1, ls2, 'r-')
                self.hodograph_canvas.canvas.draw()
            
        except Exception as e:
            print(e)

    def _plot_raw_hodograph(self):
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
                u = self.wavelet.u[self.bottom_alt:self.top_alt]
                v = self.wavelet.v[self.bottom_alt:self.top_alt]
                alt = self.wavelet.alt[self.bottom_alt:self.top_alt]
                temp = self.wavelet.temp[self.bottom_alt:self.top_alt]
                self.hodograph_canvas.canvas.ax.clear()
                self.hodograph_canvas.canvas.ax.plot(u, v, 'rx', ms = 2)
                self.hodograph_canvas.canvas.ax.annotate(alt[-1],(u[-1], v[-1]))
                self.hodograph_canvas.canvas.ax.annotate(alt[0],(u[0], v[0]))
                self.hodograph_canvas.canvas.ax.set_title("Hodograph, raw data")
                self.hodograph_canvas.canvas.draw()

        except Exception as e:
            print(e)
    
    def _plot_windowed_wavelet_power_spectrum(self):
        self.wavelet_canvas.canvas.ax.clear()
        try:
            self.scale_1 = float(self.lower_scale_bound_line_edit.text())
            self.scale_2 = float(self.upper_scale_bound_line_edit.text())
            self.alt_1 = float(self.lower_altitude_bound_line_edit.text())*1000 
            self.alt_2 = float(self.upper_altitude_bound_line_edit.text())*1000
            print(max(self.wavelet.alt))
            if self.alt_1 < min(self.wavelet.alt):
                self.alt_1 = min(self.wavelet.alt) 
            if self.alt_2 > max(self.wavelet.alt):
                self.alt_2 = max(self.wavelet.alt)
            self.windowed_alt, self.windowed_scales, self.windowed_power_spectrum = self.wavelet.return_windowed_arrays(self.scale_1, self.scale_2, self.alt_1, self.alt_2)
            self.wavelet_canvas.canvas.ax.set_title(self.in_file_name)
            self.wavelet_canvas.canvas.ax.contourf(self.windowed_alt, self.windowed_scales, self.windowed_power_spectrum)
            self.wavelet_canvas.canvas.draw()
            if self.show_me_where_i_am.isChecked():
                self.hodograph_canvas.canvas.ax.clear()
                self.hodograph_canvas.canvas.ax.plot(self.wavelet.alt, self.wavelet.ucoi[:-1], 'k')
                self.hodograph_canvas.canvas.ax.set_ylim(ymin = self.wavelet.s0)
                self.hodograph_canvas.canvas.ax.set_title(self.in_file_name)
                self.hodograph_canvas.canvas.ax.contourf(self.wavelet.alt, self.wavelet.fourier_period[1:], self.wavelet.power_spectrum)
                ls1 = [self.alt_1, self.alt_1]
                ls2 = [self.scale_1, self.scale_2]
                self.hodograph_canvas.canvas.ax.plot(ls1, ls2, 'r-')
                ls1 = [self.alt_2, self.alt_2]
                ls2 = [self.scale_1, self.scale_2]
                self.hodograph_canvas.canvas.ax.plot(ls1, ls2, 'r-')
                ls1 = [self.alt_1, self.alt_2]
                ls2 = [self.scale_1, self.scale_1]
                self.hodograph_canvas.canvas.ax.plot(ls1, ls2, 'r-')
                ls1 = [self.alt_1, self.alt_2]
                ls2 = [self.scale_2, self.scale_2]
                self.hodograph_canvas.canvas.ax.plot(ls1, ls2, 'r-')
                self.hodograph_canvas.canvas.draw()
        except Exception as e:
            print(e)

    def _print_raw_hodo_params(self):
        print('---Raw Hodo Params---')
        print("-----M1-----")
        print("AXR:", self.raw_hodo_b1/self.raw_hodo_a1, 'PHI:', 180*self.raw_hodo_phi1/np.pi, 'ETA:', self.raw_hodo_eta)
        print("-----M2-----")
        print("AXR:", self.raw_hodo_b2/self.raw_hodo_a2, 'PHI:', 180*self.raw_hodo_phi2/np.pi, "ETA:", self.raw_hodo_eta)

    def _fit_and_plot_raw_hodo(self):
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
                self.hodograph_canvas.canvas.ax.annotate(alt[-1],(u[-1],v[-1]))
                self.hodograph_canvas.canvas.ax.annotate(alt[0],(u[0],v[0]))
                self.hodograph_canvas.canvas.ax.set_title("Hodograph, raw data")
                self._fit_and_plot_ellipses(u, v, temp, alt, self.hodograph_canvas, method = 'HODOGRAPH')
                self.hodograph_canvas.canvas.draw()
                self._print_raw_hodo_params()
            else:
                self.hodograph_canvas.canvas.ax.clear()
                u = self.wavelet.u[self.bottom_alt:self.top_alt]
                v = self.wavelet.v[self.bottom_alt:self.top_alt]
                alt = self.wavelet.alt[self.bottom_alt:self.top_alt]
                temp = self.wavelet.temp[self.bottom_alt:self.top_alt]
                self._fit_and_plot_ellipses(u, v, temp, alt, self.hodograph_canvas, method = 'HODOGRAPH')
                self.hodograph_canvas.canvas.ax.plot(u, v, 'rx', ms = 2)
                self.hodograph_canvas.canvas.ax.annotate(alt[-1],(u[-1],v[-1]))
                self.hodograph_canvas.canvas.ax.annotate(alt[0],(u[0],v[0]))
                self.hodograph_canvas.canvas.ax.set_title("Hodograph, raw data")
                self.hodograph_canvas.canvas.draw()
                self._print_raw_hodo_params()

        except Exception as e:
            print(e)

    def _plot_detransformed_hodograph(self):
        self.hodograph_canvas.canvas.ax.clear()
        uinv = self.wavelet.invert_wavelet_coefficients(self.wavelet.u_wavelet, self.bottom_scale, self.top_scale, self.bottom_alt, self.top_alt)
        vinv = self.wavelet.invert_wavelet_coefficients(self.wavelet.v_wavelet, self.bottom_scale, self.top_scale, self.bottom_alt, self.top_alt)
        alt = self.wavelet.alt[self.bottom_alt:self.top_alt]
        temp = self.wavelet.temp[self.bottom_alt:self.top_alt]
        self._fit_and_plot_ellipses(uinv, vinv, temp, alt, self.hodograph_canvas)
        self.hodograph_canvas.canvas.ax.plot(uinv, vinv, 'rx', ms = 2)
        self.hodograph_canvas.canvas.ax.annotate(alt[-1],(uinv[-1],vinv[-1]))
        self.hodograph_canvas.canvas.ax.annotate(alt[0],(uinv[0],vinv[0]))
        self.hodograph_canvas.canvas.ax.set_title("Hodograph, extent of wavepacket")
        self.hodograph_canvas.canvas.draw()

    def get_altitude_window(self, arange, data_array, alt_array):
        data_windowed = [data_array[i] for i, a in enumerate(alt_array) if a > arange[0] and a < arange[1]]
        return np.asarray(data_windowed)

    def _save_gravity_wave_params(self):
        if self.file_to_save_to_line_edit.text() is not '':
            fname = self.file_to_save_to_line_edit.text()
            if self.eta < 0: 
                self.phi_stokes += 180
                self.phi1 += 180
                self.phi2 += 180
            if self.save_method_1_cbox.isChecked():
                axr_hodo = self.b1/self.a1
                phi_hodo = self.phi1
            if self.save_method_2_cbox.isChecked():
                axr_hodo = self.b2/self.a2
                phi_hodo = self.phi2

            loc = self.g_wave_location
            gwave = Gravity_Wave_Data()
            gwave.add_row_of_data(self.in_file_name[8:10], self.omega_stokes, axr_hodo, self.phi_stokes, phi_hodo, loc, 
                self.bottom_scale, self.top_scale, self.bottom_alt, self.top_alt)
            gwave.dump_data(self.file_to_save_to_line_edit.text())










if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = QMainWindow()
    m_gui = MainWindow(form)
    form.show()

    app.exec_()



