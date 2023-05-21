# Load the needed package
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import cmath
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import find_peaks
from scipy.integrate import solve_ivp
from decimal import *
from tqdm import tqdm
from JohnsonNyquistNoise import noise_on_trap


class Sinlge_Electron_Cooling(object):
    def __init__(self, Vec0, ParticleParameters = {}, TrapParameters = {}, CircuitParameters = {}, SimulationParameters = {}):
        
         # initial position & velocity
        self.Vec0 = Vec0

        # particle parameters: mass & charge
        
        self.ParticleParameters = {}
        if ('mass') in ParticleParameters:
            self.ParticleParameters['mass'] = ParticleParameters['mass']
        else:
            # by default: mass of electron
            self.ParticleParameters['mass'] = 9.10938297e-31 # 9.10938297e-31 kg 

        if ('charge') in ParticleParameters:
            self.ParticleParameters['charge'] = ParticleParameters['charge']
        else:
            # by default: charge of electron
            self.ParticleParameters['charge'] = 1.6e-19

        # trap parameters: wrf & wradical & waxial & deff

        self.TrapParameters = {}
        if ('wrf') in TrapParameters:
            self.TrapParameters['wrf'] = TrapParameters['wrf']
        else:
            # by default: 10 GHz
            self.TrapParameters['wrf'] = 2 * np.pi * 10e9 # 2pi for angular frequency
        
        if ('wradical') in TrapParameters:
            self.TrapParameters['wradical'] = TrapParameters['wradical']
        else:
            # by default: 1GHz
            self.TrapParameters['wradical'] = 2 * np.pi * 1e9 # 2pi for angular frequency
        
        if ('waxial') in TrapParameters:
            self.TrapParameters['waxial'] = TrapParameters['waxial']
        else:
            # by default: 300 MHz
            self.TrapParameters['waxial'] = 2 * np.pi * 300e6
        
        if ('deff') in TrapParameters:
            self.TrapParameters['deff'] = TrapParameters['deff']
        else:
            # by default: 200 micro
            self.TrapParameters['deff'] = 200e-6

        # define cooling circuit parameters: Rp & QualityFactor & Temperature

        self.CircuitParameters = {}
        if ('Rp') in CircuitParameters:
            self.CircuitParameters['Rp'] = CircuitParameters['Rp']
        else:
            # by default: 1e7 Ohm
            self.CircuitParameters['Rp'] = 1e7
            
        if ('QualityFactor') in CircuitParameters:
            self.CircuitParameters['QualityFactor'] = CircuitParameters['QualityFactor']
        else:
            # by default: 2000
            self.CircuitParameters['QualityFactor'] = 2000
        
        if ('Temperature') in CircuitParameters:
            self.CircuitParameters['Temperature'] = CircuitParameters['Temperature']
        else:
            # by default: 0.4 Kalvin
            self.CircuitParameters['Temperature'] = 0.4
        
       
        # simulation_parameters: dt & CoolingMode & TotalTime
        self.SimulationParameters = {}
        if ('dt') in SimulationParameters:
            self.SimulationParameters['dt'] = SimulationParameters['dt']
        else:
            # by default: dt = 1e-12 sec
            self.SimulationParameters['dt'] = 1e-12
        
        # define cooling mode: secular, red, blue, or any number; This variable determines resonator frequency
        if ('CoolingMode') in SimulationParameters:
            self.SimulationParameters['CoolingMode'] = SimulationParameters['CoolingMode']
        else:
            # by default: secular
            self.SimulationParameters['CoolingMode'] = 'secular'
        
        if ('TotalTime') in SimulationParameters:
            self.SimulationParameters['TotalTime'] = SimulationParameters['TotalTime']
        else:
            # by default: 1 micro sec
            self.SimulationParameters['TotalTime'] = 1e-6
        

    # Dc potential at time t, with phase space vector Vec
    def Vdc(self, Vec, t):
        q = self.ParticleParameters['charge']
        m = self.ParticleParameters['mass']
        waxial = self.TrapParameters['waxial']
        x, y, z, vx, vy, vz = Vec
        return 1/4/q * m * waxial ** 2* (2 * z ** 2 - x ** 2 - y ** 2)

    # Rf potential at time t, with phase space vector Vec
    def Vrf(self, Vec, t):
        q = self.ParticleParameters['charge']
        m = self.ParticleParameters['mass']
        wradical = self.TrapParameters['wradical']
        wrf = self.TrapParameters['wrf']
        x, y, z, vx, vy, vz = Vec
        return m * wradical * wrf * np.sqrt(2) / q * np.cos(wrf * t) * (x ** 2- y ** 2) /2

    def Edc(self, Vec, t):
        q = self.ParticleParameters['charge']
        m = self.ParticleParameters['mass']
        waxial = self.TrapParameters['waxial']
        x, y, z, vx, vy, vz = Vec
        A = 1/2/q * m * waxial ** 2
        return A * x, A * y, -2 * A * z

    def Erf(self, Vec, t):
        q = self.ParticleParameters['charge']
        m = self.ParticleParameters['mass']
        wradical = self.TrapParameters['wradical']
        wrf = self.TrapParameters['wrf']
        x, y, z, vx, vy, vz = Vec
        A = m * wradical * wrf * np.cos(wrf * t) * np.sqrt(2) / q
        return A * (- x), A * y, 0.

    # Calculate motional deviation according to Edc and Erf for initial Run
    def DevMotion_init(self, t, Vec, progress_bar):
        dt = self.SimulationParameters['dt']
        q = self.ParticleParameters['charge']
        m = self.ParticleParameters['mass']
        x, y, z, vx, vy, vz = Vec
        Ex, Ey, Ez = tuple(map(sum, zip(self.Edc(Vec, t),
                                        self.Erf(Vec, t))))
        ax = q * Ex / m
        ay = q * Ey / m
        az = q * Ez / m
        progress_bar.update(dt / 6)
        return vx, vy, vz, ax, ay, az
    
    def InitialRun(self, T_init, dt, DrawPosition = False, DrawSpectrum = False):
        T_init = self.SimulationParameters['TotalTime']
        N = round(T_init / dt)
        progress_bar = tqdm(total=T_init, desc='Initial Simulation for {:.2f} us, time:'.format(T_init * 1e6), position=0, leave=True)
        
        # Using RK45 to solve the ODE
        
        t_eval_init = np.linspace(0, T_init, (round(T_init/dt) + 1))
        solution = solve_ivp(fun=lambda t, Vec: self.DevMotion_init(t, Vec, progress_bar), 
                            t_span = (0, T_init), 
                            y0 = self.Vec0, 
                            t_eval = t_eval_init, 
                            first_step = dt,
                            max_step = dt,
                            atol = 1e-1,
                            rtol = 1e-1,
                            method = 'RK45')
        t_init = solution.t
        VecResult_init = solution.y
        x_init, y_init, z_init = VecResult_init[0, :], VecResult_init[1, :], VecResult_init[2, :]
        vx_init, vy_init, vz_init = VecResult_init[3, :], VecResult_init[4, :], VecResult_init[5, :]
        progress_bar.close()

        if DrawPosition:
            plt.clf()
            # Plot the x-axis motion
            fig1, ax = plt.subplots(1,2)

            # Plot the data on the axes
            ax[0].plot(t_init*1e6, x_init)
            # ax[0].plot(t_init*1e6, vx_init)

            # Add labels to the axes
            ax[0].set_xlabel('t(\mu s)')
            ax[0].set_ylabel('x(t)')

            # Plot the data on the axes
            Zoomnum = round(0.6e-7/dt)
            ax[1].plot(t_init[:Zoomnum]*1e6, x_init[:Zoomnum])

            # Add labels to the axes
            ax[1].set_xlabel('t(\mu s)')
            ax[1].set_ylabel('x(t)')

            # Show the plot
            plt.show(block=False)
        
        # Get the fourier spectrum of velocity
        vxf_init = fft(vx_init)
        vyf_init = fft(vy_init)
        vzf_init = fft(vz_init)
        tf = fftfreq(len(vx_init), dt)

        Front = 0
        # Cutoff for better spectrum plot, eliminate most useless zeros
        Cutoff = 1
        vf = vxf_init[Front:Front + (vxf_init.size//Cutoff//2)]
        ttf = tf[:len(vx_init)//2]
        ttf = ttf[Front:Front + (ttf.size//Cutoff)]

        # Get normalized magnitude of specturm
        N_vf_init = 2.0/(N+1) * np.abs(vf[0:len(vx_init)//2])


        # use find_peaks to get all the peaks(with height = 0)
        peaks, _ = find_peaks(N_vf_init, height=0)


        # we only need the three largest peaks, (secular, red and blue) therefore we find a border and find peaks again
        temp = N_vf_init[peaks]
        temp.sort()
        temp = temp[::-1]
        border = ( temp[3] + temp[2])/2
        #border = 0
        peaks, heights = find_peaks(N_vf_init, height= border)
        # We expect to get three peaks here, with first secular, second red, third blue
        # Regenerate the resonator's frequency by the third peak position
        CoolingMode = self.SimulationParameters['CoolingMode']
        if CoolingMode == 'secular':
            fres = ttf[peaks[0]]
        elif CoolingMode == 'red':
            fres = ttf[peaks[1]]
        elif CoolingMode == 'blue':
            fres = ttf[peaks[2]]
        else:
            fres = float(fres)

        # Calculate the average of the velocity in the first micromotion peroid
        #Tradical = 1 / ttf[peaks[0]]

        print("The location of the three largest peaks is {}, {}, {} Hz".format(tf[peaks[0]], tf[peaks[1]], tf[peaks[2]]))
        if DrawSpectrum:
            plt.clf()
            # Plot the result
            plt.plot(ttf, N_vf_init)
            plt.plot(ttf[peaks], N_vf_init[peaks], "x")

            for i in range(0,3):                                      
                #plt.text(ttf[peaks[i]], N_vf_init[peaks[i]], "%f" % N_vf[peaks[i]])
                plt.text(ttf[peaks[i]], N_vf_init[peaks[i]], "%f" % ttf[peaks[i]])

            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')
            plt.title('Frequency spectrum')
            plt.grid()
            plt.show(block=False)
        
        tf[0] = 1e-5

        Rp = self.CircuitParameters['Rp']
        QualityFactor = self.CircuitParameters['QualityFactor']
        q = self.ParticleParameters['charge']
        deff = self.TrapParameters['deff']

        Rezf = Rp / (1 + QualityFactor ** 2 * (tf / fres - fres / tf) ** 2)
        Imzf = Rp * QualityFactor  * (tf / fres - fres / tf) / (1 + QualityFactor ** 2 * (tf / fres - fres / tf) ** 2)
        Rezf = np.array(Rezf)
        Imzf = np.array(Imzf)
        zf = Rezf + 1j * Imzf

        # This is for convolution of the spectrum
        #Damping_Exf =np.convolve(vxf_init, zf)

        # This is for multiply of the spectrum
        Damping_Exf = np.multiply(vxf_init, zf)

        Damping_Ex_t = np.real(ifft(Damping_Exf))
        Damping_Ex = list(Damping_Ex_t.real)
        # An additional general -1 should be added, because this is damping force
        Damping_Ex =[- q / deff ** 2 * a for a in Damping_Ex]
        Damping_Ex_Ampl = (abs(max(Damping_Ex)) + abs(min(Damping_Ex))) / 2
        phase = np.arcsin(Damping_Ex[0] / Damping_Ex_Ampl)
        #phase = 0.
        return Damping_Ex_Ampl, fres, phase
    
    def DampingForce(self, Damping_Ex_Ampl, fres, phase, Vec_init, Vec, t):
        x_init, y_init, z_init, vx_init, vy_init, vz_init = Vec_init
        x, y, z, vx, vy, vz = Vec
        #DampingFactor_tlist.append(t)
        
        if abs(vx / vx_init) > 10 or abs(vx_init) < 1e-6 or vx_init == 0:
            #DampingFactor_list.append(10)
            #return 0., 0.,0.
            if vx_init == 0:
                return 0., 0., 0.
            if vx / vx_init > 0:
                return Damping_Ex_Ampl * 10 * np.sin(2 * np.pi * fres * t), 0., 0.
            if vx / vx_init < 0:
                return - Damping_Ex_Ampl * 10 * np.sin(2 * np.pi * fres * t), 0., 0.

        #DampingFactor_list.append(vx / vx_init)
        return Damping_Ex_Ampl * vx / vx_init * np.sin(2 * np.pi * fres * t + phase), 0., 0.
    
    def JNNoise(self, Vec, t, fres, NoiseList):
        dt = self.SimulationParameters['dt']

        posi = int(t/dt)

        return NoiseList[posi] + (NoiseList[posi + 1] - NoiseList[posi]) * (t/dt - posi), 0., 0.

    # Calculate motional deviation according to Edc and Erf
    def DevMotion_para(self, t, Vec_para, Damping_Ex_Ampl, fres, phase, progress_bar, NoiseList, WithNoise = True):
        dt = self.SimulationParameters['dt']
        q = self.ParticleParameters['charge']
        m = self.ParticleParameters['mass']

        x_init, y_init, z_init, vx_init, vy_init, vz_init, x, y, z, vx, vy, vz = Vec_para
        Vec_init = x_init, y_init, z_init, vx_init, vy_init, vz_init
        Vec = x, y, z, vx, vy, vz
        if WithNoise:
            Ex, Ey, Ez = tuple(map(sum, zip(self.Edc(Vec, t),
                                            self.Erf(Vec, t),
                                            self.DampingForce(Damping_Ex_Ampl, fres, phase, Vec_init, Vec,  t),
                                            self.JNNoise(Vec, t, fres, NoiseList)
                                            )))
        else:
            Ex, Ey, Ez = tuple(map(sum, zip(self.Edc(Vec, t),
                                            self.Erf(Vec, t),
                                            self.DampingForce(Damping_Ex_Ampl, fres, phase, Vec_init, Vec, t)
                                            )))
        ax = q * Ex / m
        ay = q * Ey / m
        az = q * Ez / m

        Ex_init, Ey_init, Ez_init = tuple(map(sum, zip(self.Edc(Vec_init, t),
                                                       self.Erf(Vec_init, t),
                                                        )))
        ax_init = q * Ex_init / m
        ay_init = q * Ey_init / m
        az_init = q * Ez_init / m
        progress_bar.update(dt / 6)
        return vx_init, vy_init, vz_init, ax_init, ay_init, az_init, vx, vy, vz, ax, ay, az
    
    def SecondRun(self, Damping_Ex_Ampl, fres, phase, DrawPosition = True, DrawVelocity = True, SaveData = False, SaveFig = True):
        T = self.SimulationParameters['TotalTime']
        dt = self.SimulationParameters['dt']
        N = int(np.round(T / dt))
        Vec0 = self.Vec0 + self.Vec0
        progress_bar = tqdm(total=T, desc='Simulation for Second Run with {:.2f} us'.format(T * 1e6), position=0, leave=True)
        # Using RK45 to solve the ODE
        #print(N)
        t_eval = np.linspace(0, T, N + 1)

        # Generate the noise List in advance
        deff = self.TrapParameters['deff']
        Temperature = self.CircuitParameters['Temperature']
        Rp = self.CircuitParameters['Rp']
        QualityFactor = self.CircuitParameters['QualityFactor']

        JNNoise_Ex = noise_on_trap(deff, Temperature, Rp, fres, QualityFactor, T, dt, False)

        # This is for damping factor and time domain method

        solution = solve_ivp(fun=lambda t, Vec: self.DevMotion_para(t, Vec, Damping_Ex_Ampl, fres, phase, progress_bar, JNNoise_Ex, True), 
                            t_span = (0, T), 
                            y0 = Vec0, 
                            t_eval = t_eval, 
                            first_step = dt,
                            max_step = dt,
                            atol = 1e-1,
                            rtol = 1e-1,
                            method = 'RK45')

        t_damp = solution.t
        VecResult = solution.y

        x_damp, y_damp, z_damp = VecResult[6, :], VecResult[7, :], VecResult[8, :]
        vx_damp, vy_damp, vz_damp = VecResult[9, :], VecResult[10, :], VecResult[11, :]


        progress_bar.close()
        if DrawPosition:
            plt.clf()
            # Plot the x-axis motion
            fig1, ax = plt.subplots(1,2)

            # Plot the data on the axes
            ax[0].plot(np.array(t_eval[:len(x_damp) // 500 * 500])*1e6, x_damp[:len(x_damp) // 500 * 500])
            #ax[0].plot(np.array(t_eval[:len(x_damp)])*1e6, x_damp[:len(x_damp)])

            # Add labels to the axes
            ax[0].set_xlabel('t($\mu$s)')
            ax[0].set_ylabel('x(t)')

            # Plot the data on the axes
            Zoomnum = round(0.02e-6/dt)
            ax[1].plot(np.array(t_eval[:Zoomnum // 1 * 1])*1e6, x_damp[:Zoomnum])

            # Add labels to the axes
            ax[1].set_xlabel('t($\mu$s)')
            ax[1].set_ylabel('x(t)')

            plt.grid()
            # Show the plot
            plt.show(block=False)
        
        # Based on the velocity plot, find the cooling time
        vx_abs_init = (np.abs(np.max(vx_damp)) + np.abs(np.min(vx_damp))) / 2
        e = 2.71828182845904523536

        peaks, heights = find_peaks(vx_damp, height= vx_abs_init/ np.sqrt(e))

        CoolingMode = self.SimulationParameters['CoolingMode']
        wrf = self.TrapParameters['wrf']
        waxial = self.TrapParameters['waxial']
        wradical = self.TrapParameters['wradical']


        if DrawVelocity:
            plt.clf()
            plt.plot(np.array(t_damp[:len(x_damp) // 500 * 500])*1e6, vx_damp[:len(x_damp) // 500 * 500],\
                    label = '$\omega_r$:{}GHz, $\omega_m$:{}GHz, $\omega_z$:{}MHz'.format(wrf/2/np.pi/1e9, wradical/2/np.pi/1e9, waxial/2/np.pi/1e6))
            plt.plot(np.array(t_damp[:len(x_damp) // 500 * 500])*1e6, vx_abs_init / np.sqrt(e) * np.ones(len(x_damp))[:len(x_damp) // 500 * 500], 'r--', label = '$1/\sqrt{e}$')
            plt.plot(np.array(t_damp[:len(x_damp) // 500 * 500])*1e6, -vx_abs_init / np.sqrt(e) * np.ones(len(x_damp))[:len(x_damp) // 500 * 500], 'r--')
            plt.plot(np.array(t_damp[peaks[-1]])*1e6, vx_damp[peaks[-1]], "x")
            plt.text(np.array(t_damp[peaks[-1]])*1e6, vx_damp[peaks[-1]], "%.2f $\mu$s" % (float(t_damp[peaks[-1]]) * 1e6))
            #ax[0].plot(np.array(t_eval[:len(x_damp)])*1e6, x_damp[:len(x_damp)])

            # Add labels to the axes
            plt.xlabel('Time($\mu$s)')
            plt.ylabel('$v_x$(t)')
            plt.legend()
            plt.grid()
            plt.ylim([-2e5, 2e5])
            plt.title('{:.2f} Hz, {:.1f} M$\Omega$'.format(fres, Rp/1e6))
            Zoomnum = round(0.2e-6/dt)
            x_zoom = np.array(t_eval[:Zoomnum // 1 * 1])*1e6
            y_zoom = vx_damp[:Zoomnum]
            axes = plt.axes([.50, .2, .23, .2])
            axes.plot(x_zoom, y_zoom, c='blue', lw=1, label='Zoomed curve')
            axes.set_xlabel('$\mu$s')
            #axes.legend(‘right’)

            plt.grid()
            FileName = '{}_Wrf = {:.2f}_Wradical = {:.2f}_Waxial = {:.2f}_T = {:.2f}us'.format(CoolingMode, wrf, wradical, waxial, T * 1e6)
            if SaveFig:
                plt.savefig('figures/' + FileName + '.png')
            plt.show(block=False)
            
        # Decide whether save data
        

        if SaveData:
            FileName = '{}_Wrf = {:.2f}_Wradical = {:.2f}_Waxial = {:.2f}_T = {:.2f}us'.format(CoolingMode, wrf, wradical, waxial, T * 1e6)
            np.save('data/t_' + FileName, t_damp)
            np.save('data/x_' + FileName, x_damp)
            np.save('data/y_' + FileName, y_damp)
            np.save('data/z_' + FileName, z_damp)
            np.save('data/vx_' + FileName, vx_damp)
            np.save('data/vy_' + FileName, vy_damp)
            np.save('data/vz_' + FileName, vz_damp)
        
        if t_damp[peaks[-1]] / T > 0.9:
            print('Cooling Failed')
        else:
            print('Cooling Time is: {:.3f} us'.format(t_damp[peaks[-1]] * 1e6))

        return t_damp[peaks[-1]]
    
    def Run(self):
        TotalTime = self.SimulationParameters['TotalTime']
        dt = self.SimulationParameters['dt']

        Damping_Ex_Ampl, fres, phase = self.InitialRun(TotalTime, dt, False, False)
        Cooling_Time = self.SecondRun(Damping_Ex_Ampl, fres, phase, 
                                      DrawPosition = False,
                                      DrawVelocity = True, 
                                      SaveData = False,
                                      SaveFig = True)

        return Cooling_Time

