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
from scipy.stats import norm
import matplotlib.mlab as mlab
from scipy.optimize import leastsq


class Sinlge_Electron_Cooling(object):
    def __init__(self, Vec0, ParticleParameters = {}, TrapParameters = {}, CircuitParameters = {}, SimulationParameters = {}):
        """
        Definition of Parameters in the single electron cooling process, 

        Params:
            Vec0: float list/array 6*1
                The initial vector in the 3D phase space, including position and velocity, 
                [initial position in x, initial position in y, initial position in z, 
                 initial velocity in x, initial velocity in y, initial velocity in z]
            ParticleParameters: dictionary
                quantities of simulated particles, including (mass)"mass":float and (charge)"charge":float
            TrapParameters: dictionary
                meaningful Trap parameters for the trap potential
                V_{dc} = \kappa U (\frac{2z^2 - x^2 - y^2}{2 r_0^2})
                V_{rf} = V \cos(\Omega_{rf} t) \frac{x^2 - y^2}{2r_0^2}
                After the parameterization by solving motion equation in the first order, V_{dc} and V_{rf} can be reexpressed as:
                V_{dc} = \frac{1}{2} m \omega_z^2 \frac{2z^2 - x^2 - y^2}{2}
                V_{rf} = \sqrt{2}m\omega_r \Omega_{rf}\cos(\Omega_{rf} t)\frac{x^2 - y^2}{2}
                There are three parameters, including (\omega_z)"waxial":float, (\omega_r)"wradical":float, (\Omega_{rf})"wrf":float
                Note that all these are angular velocity frequency
            CircuitPrameters: dictionary
                circuit parameters for the cooling process, the tank circuit is simplified as a RLC parallel circuit
                including: (resistence/peak impedance)"Rp":float, (quality factor)"QualityFactor":float,  (temperature related to Johnson-Nyquist Nose)"Temperature":float
                resonator frequency is also a important parameters for cooling, but it is determined by different cooling mode, which is later defined in the SimulationParameters
                resonator frequency is later got by InitialRun
            SimulationParameters: dictionary
                simulation parameters for the coolign process, including: total simulation time "TotalTime":float, 
                simulation time step "dt":float, cooling mode "CoolingMode": "secular"/"blue"/"red"/float --> meaning a artifically chosen frequency
        """

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
        """
        DC potential

        Params:
            Vec: 6*1 array
                the vector in the phase space, [x, y, z, vx, vy, vz]
            t: float
        Output:
            Vdc: float

        """
        q = self.ParticleParameters['charge']
        m = self.ParticleParameters['mass']
        waxial = self.TrapParameters['waxial']
        x, y, z, vx, vy, vz = Vec
        return 1/4/q * m * waxial ** 2* (2 * z ** 2 - x ** 2 - y ** 2)

    # Rf potential at time t, with phase space vector Vec
    def Vrf(self, Vec, t):
        """
        RF potential

        Params:
            Vec: 6*1 array
                the vector in the phase space, [x, y, z, vx, vy, vz]
            t: float
        Output:
            Vrf: float

        """
        q = self.ParticleParameters['charge']
        m = self.ParticleParameters['mass']
        wradical = self.TrapParameters['wradical']
        wrf = self.TrapParameters['wrf']
        x, y, z, vx, vy, vz = Vec
        return m * wradical * wrf * np.sqrt(2) / q * np.cos(wrf * t) * (x ** 2- y ** 2) /2

    def Edc(self, Vec, t):
        """
        Electric field by DC electrode, the position gradient of Vdc

        Params:
            Vec: 6*1 array
                the vector in the phase space, [x, y, z, vx, vy, vz]
            t: float
        Output:
            Edc: list, size = (3)

        """
        q = self.ParticleParameters['charge']
        m = self.ParticleParameters['mass']
        waxial = self.TrapParameters['waxial']
        x, y, z, vx, vy, vz = Vec
        A = 1/2/q * m * waxial ** 2
        return A * x, A * y, -2 * A * z

    def Erf(self, Vec, t):
        """
        Electric field by RF electrode, the position gradient of Vrf

        Params:
            Vec: 6*1 array
                the vector in the phase space, [x, y, z, vx, vy, vz]
            t: float
        Output:
            Erf: list, size = (3)

        """
        q = self.ParticleParameters['charge']
        m = self.ParticleParameters['mass']
        wradical = self.TrapParameters['wradical']
        wrf = self.TrapParameters['wrf']
        x, y, z, vx, vy, vz = Vec
        A = m * wradical * wrf * np.cos(wrf * t) * np.sqrt(2) / q
        return A * (- x), A * y, 0.

    # Calculate motional deviation according to Edc and Erf for initial Run
    def DevMotion_init(self, t, Vec, progress_bar):
        """
        Deviation of motion, the dimension is 6, representing x, y, z, vx, vy, vz

        Params:
            t: float
                Time
            Vec: 6*1 array
                the vector in the phase space, [x, y, z, vx, vy, vz]
            
        Output:
            Time deviation of phase vector.
            vx:float , vy: float, vz: float, ax: float, ay: float, az: float
            
        """
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
        T_init = 2 * self.SimulationParameters['TotalTime']
        if T_init < 10e-6:
            T_init = 10e-6
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

        # Get the time/time steps in a secular motion period
        NumSecularPeriod = round(2 * np.pi / ttf[peaks[0]] / dt)

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
        return Damping_Ex_Ampl, fres, phase, NumSecularPeriod
    
    def DampingForce(self, Damping_Ex_Ampl, fres, phase, Vec_init, Vec, t):
        x_init, y_init, z_init, vx_init, vy_init, vz_init = Vec_init
        x, y, z, vx, vy, vz = Vec
        
        if abs(vx / vx_init) > 10 or abs(vx_init) < 1e-6 or vx_init == 0:
            
            if vx_init == 0:
                return 0., 0., 0.
            if vx / vx_init > 0:
                return Damping_Ex_Ampl * 10 * np.sin(2 * np.pi * fres * t), 0., 0.
            if vx / vx_init < 0:
                return - Damping_Ex_Ampl * 10 * np.sin(2 * np.pi * fres * t), 0., 0.

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
    

    # input a list of position, time and velocity

    def Vps(self, Vec, t):
        q = self.ParticleParameters['charge']
        m = self.ParticleParameters['mass']
        wrf = self.TrapParameters['wrf']

        Ex, Ey, Ez = self.Erf(Vec, t)
        return q ** 2 / 4 / m / wrf ** 2 * (Ex ** 2 + Ey ** 2 + Ez ** 2)

    # Calculte for the kinetic energy

    def Ekin(self, Vec, t):
        m = self.ParticleParameters['mass']
        x, y, z, vx, vy, vz = Vec
        return 0.5 * m * (vx ** 2 + vy ** 2 + vz ** 2)

    def SecondRun(self, 
                  Damping_Ex_Ampl, 
                  fres, 
                  phase, 
                  NumSecularPeriod,
                  DrawPosition = True, 
                  DrawVelocity = True, 
                  SaveData = False, 
                  SaveFig = True,
                  #FinalTemperature = True
                  DrawVelocityHist = True,
                  DrawEnergyHist = True
                  ):
        
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

        solution = solve_ivp(fun=lambda t, Vec: self.DevMotion_para(t, 
                                                                    Vec, 
                                                                    Damping_Ex_Ampl, 
                                                                    fres, 
                                                                    phase, 
                                                                    progress_bar, 
                                                                    JNNoise_Ex, 
                                                                    WithNoise = False), 
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

        m = self.ParticleParameters['mass']
        q = self.ParticleParameters['charge']
        CoolingMode = self.SimulationParameters['CoolingMode']
        wrf = self.TrapParameters['wrf']
        waxial = self.TrapParameters['waxial']
        wradical = self.TrapParameters['wradical']


        if DrawVelocity:
            plt.clf()
            plt.plot(np.array(t_damp[:len(x_damp) // 500 * 500])*1e6, vx_damp[:len(x_damp) // 500 * 500],\
                    label = '$\Omega_rf$:{}GHz, $\omega_r$:{}GHz, $\omega_z$:{}MHz'.format(wrf/2/np.pi/1e9, wradical/2/np.pi/1e9, waxial/2/np.pi/1e6))
            plt.plot(np.array(t_damp[:len(x_damp) // 500 * 500])*1e6, vx_abs_init / np.sqrt(e) * np.ones(len(x_damp))[:len(x_damp) // 500 * 500], 'r--', label = '$1/\sqrt{e}$')
            plt.plot(np.array(t_damp[:len(x_damp) // 500 * 500])*1e6, -vx_abs_init / np.sqrt(e) * np.ones(len(x_damp))[:len(x_damp) // 500 * 500], 'r--')
            plt.plot(np.array(t_damp[peaks[-1]])*1e6, vx_damp[peaks[-1]], "x")
            plt.text(np.array(t_damp[peaks[-1]])*1e6, vx_damp[peaks[-1]], "%.2f $\mu$s" % (float(t_damp[peaks[-1]]) * 1e6))
            #ax[0].plot(np.array(t_eval[:len(x_damp)])*1e6, x_damp[:len(x_damp)])
           
            # Add estimated cooling time for secular cooling
            if CoolingMode == 'secular':
                plt.plot(m * deff ** 2/ q ** 2/ Rp * 1e6 * np.ones(1000), np.linspace(-2e5, 2e5, 1000), '--', color = 'orange', label = 'Theory Secular Cooling Time')

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
            axes.plot(x_zoom, y_zoom, color='blue', lw=1, label='Zoomed curve')
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
        #clear

        CoolingTime = t_damp[peaks[-1]]
        
        Energy = []
        EquilibriumTime = 2
        print(EquilibriumTime * round(CoolingTime / dt))
        
        # we assume that after 4 cooling Time: e^-4, the motion is dominant by the noise
        t_damp = t_damp[EquilibriumTime * round(CoolingTime / dt):]
        x_damp = x_damp[EquilibriumTime * round(CoolingTime / dt):]
        y_damp = y_damp[EquilibriumTime * round(CoolingTime / dt):]
        z_damp = z_damp[EquilibriumTime * round(CoolingTime / dt):]
        vx_damp = vx_damp[EquilibriumTime * round(CoolingTime / dt):]
        vy_damp = vy_damp[EquilibriumTime * round(CoolingTime / dt):]
        vz_damp = vz_damp[EquilibriumTime * round(CoolingTime / dt):]

        N = len(t_damp)
        SampleList = np.arange(0, N, NumSecularPeriod)
        print(N)
        print(NumSecularPeriod)
        print(SampleList)  
        # we didn't use the pesudopotential approximation till now.
        '''
        all direction of x, y, z energy(potential and kinetic energy), average over a waxial period
        '''
        for i in SampleList:
            Energy_temp = 0
            for j in range(NumSecularPeriod):
                Vec = x_damp[i + j], y_damp[i + j], z_damp[i + j], vx_damp[i + j], vy_damp[i + j], vz_damp[i + j]
                Energy_temp = Energy_temp + self.Vdc(Vec, t_damp[i + j]) + self.Vrf(Vec, t_damp[i + j]) + self.Ekin(Vec, t_damp[i + j])

            Energy.append(Energy_temp / NumSecularPeriod)
        print(Energy)        
        # best fit of Energy Hist
        Energy_fitfunc  = lambda p, x: p[0]*np.exp(- x/p[1])+p[2]
        Energy_errfunc  = lambda p, x, y: (y - Energy_fitfunc(p, x))
        plt.clf()
        plt.plot(Energy)
        plt.savefig('figures/' + 'time vs Energy' + '.png')
        plt.show()

        plt.clf()
        Energy_n, Energy_bin_edges,_ = plt.hist(Energy, 1000, density=True, color='green', label = 'Histogram')
        print(Energy_n)
        print(Energy_bin_edges)
        plt.title('Histogram for Energy')
        plt.show()
        xdata = 0.5*(Energy_bin_edges[1:] + Energy_bin_edges[:-1])
        ydata = Energy_n

        init  = [1.0, 0.5, 0.5]

        out   = leastsq( Energy_errfunc, init, args=(xdata, ydata))
        c = out[0]

        print("A exp[-(x-mu)/sigma] + k ")
       
        print("Fit Coefficients:")
        print(c[0],abs(c[1]),c[2])

        
        if DrawEnergyHist:
            #plt.plot(xdata, Energy_fitfunc(c, xdata), label = 'Fitting Curve')
            plt.plot(xdata, ydata, label = 'raw data')
            
            FileName = 'Histogram for Energy'
            plt.title(r'$A = %.3f\ \sigma = %.3f\ k = %.3f $' %(c[0],abs(c[1]),c[2]))
            
            plt.savefig('figures/' + FileName + '.png')
            plt.show()
        else:
            plt.plot(xdata, Energy_fitfunc(c, xdata))
            plt.plot(xdata, ydata)
            plt.title(r'$A = %.3f\ \sigma = %.3f\ k = %.3f $' %(c[0],abs(c[1]),c[2]))
            plt.show(block = False)
        
        '''
        just by velocity distribution
        '''
        Velocity_fitfunc  = lambda p, x: p[0]*np.exp(- 0.5 * ((x-p[1])/p[2]) ** 2)+p[3]
        Velocity_errfunc  = lambda p, x, y: (y - Velocity_fitfunc(p, x))
       
        plt.clf()
        Velocity_n, Velocity_bin_edges,_ = plt.hist(vx_damp, 60, density=True, color='green', alpha=0.75)
        plt.title('Histogram for velocity')
        plt.show()
        xdata = 0.5*(Velocity_bin_edges[1:] + Velocity_bin_edges[:-1])
        ydata = Velocity_n

        init  = [1.0, 0.5, 0.5, 0.5]

        out   = leastsq( Velocity_errfunc, init, args=(xdata, ydata))
        c     = out[0]

        print("A exp[-0.5((x-mu)/sigma)^2] + k ")
       
        print("Fit Coefficients:")
        print(c[0],c[1],abs(c[2]),c[3])
        if DrawVelocityHist:
            #plt.plot(xdata, Velocity_fitfunc(c, xdata))
            plt.plot(xdata, ydata)
            plt.title(r'$A = %.3f\  \mu = %.3f\  \sigma = %.3f\ k = %.3f $' %(c[0],c[1],abs(c[2]),c[3]))
            
            FileName = 'Histogram for Velocity'
            plt.savefig('figures/' + FileName + '.png')
            plt.show()
        else:
            plt.show(block = False)
        
        FinalTemperature = c[2]
        
       
        return CoolingTime, FinalTemperature
    

    def Run(self):
        #sys.modules[__name__].__dict__.clear()
        #get_ipython().magic('reset -sf')
        TotalTime = self.SimulationParameters['TotalTime']
        dt = self.SimulationParameters['dt']

        Damping_Ex_Ampl, fres, phase, NumSecularPeriod = self.InitialRun(TotalTime, dt, False, False)
        Cooling_Time, FinalTemperature = self.SecondRun(Damping_Ex_Ampl, fres, phase, NumSecularPeriod,
                                                        DrawPosition = False,
                                                        DrawVelocity = True, 
                                                        SaveData = False,
                                                        SaveFig = True
                                                        )

        return Cooling_Time, FinalTemperature

