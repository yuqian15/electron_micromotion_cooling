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
from JohnsonNyquistNoise import noise_on_trap, noise_on_trap_V2, noise_on_trap_FDT
from scipy.stats import norm
import matplotlib.mlab as mlab
from scipy.optimize import leastsq
import os


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
                There are three parameters, including (\omega_z)"waxial":float, (\omega_r)"wradial":float, (\Omega_{rf})"wrf":float
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

        # trap parameters: wrf & wradial & waxial & deff

        self.TrapParameters = {}
        if ('wrf') in TrapParameters:
            self.TrapParameters['wrf'] = TrapParameters['wrf']
        else:
            # by default: 10 GHz
            self.TrapParameters['wrf'] = 2 * np.pi * 10e9 # 2pi for angular frequency
        
        if ('wradial') in TrapParameters:
            self.TrapParameters['wradial'] = TrapParameters['wradial']
        else:
            # by default: 1GHz
            self.TrapParameters['wradial'] = 2 * np.pi * 1e9 # 2pi for angular frequency
        
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
        if ('C') in CircuitParameters:
            self.CircuitParameters['C'] = CircuitParameters['C']
        else:
            # by default: 1e7 Ohm
            self.CircuitParameters['C'] = 1e-12
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
        
        if ('Aim') in SimulationParameters:
            self.SimulationParameters['Aim'] = SimulationParameters['Aim']
        else:
            # by default: CoolingTime + FinalTemperature
            self.SimulationParameters['Aim'] = 'CoolingTime+FinalTemperature'
        

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
        wradial = self.TrapParameters['wradial']
        wrf = self.TrapParameters['wrf']
        x, y, z, vx, vy, vz = Vec
        return m * wradial * wrf * np.sqrt(2) / q * np.cos(wrf * t) * (x ** 2- y ** 2) /2

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
        wradial = self.TrapParameters['wradial']
        wrf = self.TrapParameters['wrf']
        x, y, z, vx, vy, vz = Vec
        A = m * wradial * wrf * np.cos(wrf * t) * np.sqrt(2) / q
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

    def InitialSimulation(self, DrawPosition = False, DrawSpectrum = False):
        """
        """
        T_init = self.SimulationParameters['TotalTime']
        dt = self.SimulationParameters['dt']
        
        N = round(T_init / dt)
        progress_bar = tqdm(total=T_init, desc='Simulation for {:.2f} us, time:'.format(T_init * 1e6), position=0, leave=True)
        
        # Using RK45 to solve the ODE
        
        # generate the time step list
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

        # The three largetest peaks correspond to: secular motion frequency, red sideband frequency, blue sideband frequency
        print("The location of the three largest peaks is {}, {}, {} Hz".format(tf[peaks[0]], tf[peaks[1]], tf[peaks[2]]))
        return fres, NumSecularPeriod

    def InitialRun(self, DrawPosition = False, DrawSpectrum = False):
        """
        Initial Run @ No noise, No damping circuit; The aim of the initial run is to get the 
            1. accurate motion spectrum;
            2. the damping force amplitude;
            3. phase of the damping force;
                * 2. & 3. are because of the assumption that the damping force takes the form: A sin(w_res t + phi)
            4. the number of the time steps in a radial direction secular motion period, which is used for calculating average energy and fitting to get the cooling time
        
        
        Param
        -------
            - DrawPosition: bool
                determine whether drawing the time VS position plot after the initial simulation finishes
            - DrawSpectrum: bool
                determine whether drawing the velocity spectrum plot after the initial simulation finishes
        
        Output
        -------
            - Damping_Ex_Ampl: float
                Damping Force Amplitude(The maximum damping force)
            - fres: float
                resonator frequency, which is determined by cooling mode and the motional spectrum
            - phase: float, in [0, 2Pi]
                initial phase in the damping force list
            - NumSecularPeriod: integer
                the number of the time steps in a radial direction secular motion period
        """
        T_init = 2 * self.SimulationParameters['TotalTime']
        dt = self.SimulationParameters['dt']
        
        N = round(T_init / dt)
        progress_bar = tqdm(total=T_init, desc='Simulation for {:.2f} us, time:'.format(T_init * 1e6), position=0, leave=True)
        
        # Using RK45 to solve the ODE
        
        # generate the time step list
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

        # The three largetest peaks correspond to: secular motion frequency, red sideband frequency, blue sideband frequency
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

        # Gnerating the resonator frequency response, used for a "filter" in the damping force
        Rezf = Rp / (1 + QualityFactor ** 2 * (tf / fres - fres / tf) ** 2)
        Imzf = Rp * QualityFactor  * (tf / fres - fres / tf) / (1 + QualityFactor ** 2 * (tf / fres - fres / tf) ** 2)
        Rezf = np.array(Rezf)
        Imzf = np.array(Imzf)
        zf = Rezf + 1j * Imzf

        # multiply of the spectrum, because Z(w) = U(w) / I(w)
        Damping_Exf = np.multiply(vxf_init, zf)

        # inverse fourier transformation to get a damping force list in the time domain
        Damping_Ex_t = np.real(ifft(Damping_Exf))
        Damping_Ex = list(Damping_Ex_t.real)
        # An additional general -1 should be added, because this is damping force, and q(charge) is defined by absolute value
        Damping_Ex =[- q / deff ** 2 * a for a in Damping_Ex]

        # Getting the damping force electric field 
        Damping_Ex_Ampl = (abs(max(Damping_Ex)) + abs(min(Damping_Ex))) / 2
        # Getting the damping force initial pahse
        phase = np.arcsin(Damping_Ex[0] / Damping_Ex_Ampl)

        return Damping_Ex_Ampl, fres, phase, NumSecularPeriod

    def DampingForce(self, Damping_Ex_Ampl, fres, phase, Vec_init, Vec, t):
        """
        Generating the real Damping Force VS time for the second simulation, an additional "damping factor" = vx/vx_init should times with the original damping force list
        because the real velocity of the electron keeps decreasing; the damping force should also damp its amplitude together with velocity damping

        Param:
            Damping_Ex_Ampl: float
                damping force electric field amplitude, got from initial simulation
            fres: float
                resonator frequncy
            phase: float
                initial phase in the damping force list, got from initial simulation
            Vec_init: 6 tuple
                the Vector in the phase space if the motion of the electron has no damping force or noise, got during the second simulation
            Vec: 6 tuple
                the Vector in the phase space if the motion of the electron has damping force or noise, got during the second simulation
            t: float
                time
        
        Output:
            Tuple 3
                the damping force electric field in all x,y,z directions
        """
        
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
        """
        Generating the real time Johnson-Nyquist noise electric field 

        Param:
            fres: float
                resonator frequncy
            Vec: 6 tuple
                the Vector in the phase space if the motion of the electron has damping force or noise, got during the second simulation
            t: float
                time
            NoiseList: list
                a list generated before the second simulation starts, by using the JohnsonNyquistNoise.py

        Output:
            Tuple: 3
                the Johnson nosie electric field in all x,y,z directions
        """
        dt = self.SimulationParameters['dt']

        posi = int(t/dt)

        return NoiseList[posi] + (NoiseList[posi + 1] - NoiseList[posi]) * (t/dt - posi), 0., 0.

    # Calculate motional deviation according to Edc and Erf
    def DevMotion_para(self, t, Vec_para, Damping_Ex_Ampl, fres, phase, progress_bar, NoiseList, WithNoise = True):
        """
        Deviation of Motion in the phase space(12 dimensions, including 
            1. Vec_init = x_init, y_init, z_init, vx_init, vy_init, vz_init.
                this is the phase space vector in the simulation where there is no damping force or Johnson-Nyquist noise appearing in the electron's motion
    
            2. Vec = x, y, z, vx, vy, vz
                this is the phase space vector in the simulation where there is damping force or Johnson-Nyquist noise appearing in the electron's motion)
        So there are two simulation in parallel, altogher the Vector has 12 dimensions

        Param:
            t: float
                time
            Vec_para: tuple 12
                parallel vector in the phase space
            Damping_Ex_Ampl: float
                damping force electric field amplitude, got from initial simulation, used for damping force
            fres: float
                resonator frequncy, used for damping force
            phase: float
                initial phase in the damping force list, got from initial simulation, used for damping force
            progress_bar
                tqdm library
            NoiseList: list
        """
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
        """
        pseudo-potential at the time = t
        Param:
            Vec: tuple 6
                Vector in the phase space at time = t
            t: float
                time
        Output:
            float
                the pesudo potential
        """
        q = self.ParticleParameters['charge']
        m = self.ParticleParameters['mass']
        wrf = self.TrapParameters['wrf']

        Ex, Ey, Ez = self.Erf(Vec, t)
        return q ** 2 / 4 / m / wrf ** 2 * (Ex ** 2 + Ey ** 2 + Ez ** 2)

    # Calculte for the kinetic energy

    def Ekin(self, Vec, t):
        """
        Total kinetic energy when time = t
        Param:
            Vec: tuple 6
                Vector in the phase space at time = t
            t: float
                time
        Output: float
                the total kinetic energy
        """
        m = self.ParticleParameters['mass']
        x, y, z, vx, vy, vz = Vec
        return 0.5 * m * (vx ** 2 + vy ** 2 + vz ** 2)
    
    def Vdc_without_axial(self, Vec, t):
        """
        DC potental without the axial direction at time = t
        Param:
            Vec: tuple 6
                Vector in the phase space at time = t
            t: float
                time
        Output: float
                DC potental without the axial direction at time = t
        """
        q = self.ParticleParameters['charge']
        m = self.ParticleParameters['mass']
        waxial = self.TrapParameters['waxial']
        x, y, z, vx, vy, vz = Vec
        return 1 /4/ q * m * waxial ** 2* ( - x ** 2 - y ** 2)

    def Vrf_without_axial(self, Vec, t):
        """
        RF potental without the axial direction at time = t
        Param:
            Vec: tuple 6
                Vector in the phase space at time = t
            t: float
                time
        Output: float
                RF potental without the axial direction at time = t
        """
        m = self.ParticleParameters['mass']
        wradial = self.TrapParameters['wradial']
        wrf = self.TrapParameters['wrf']
        q = self.ParticleParameters['charge']
        x, y, z, vx, vy, vz = Vec
        return m * wradial * wrf * np.sqrt(2) / q * np.cos(wrf * t) * (x ** 2- y ** 2) /2
    
    def Vps_without_axial(self, Vec, t):
        """
        pesudo potental energy without the axial direction at time = t
        Param:
            Vec: tuple 6
                Vector in the phase space at time = t
            t: float
                time
        Output: float
                pesudo potental energy without the axial direction at time = t
        """
        q = self.ParticleParameters['charge']
        m = self.ParticleParameters['mass']
        wrf = self.TrapParameters['wrf']
        x, y, z, vx, vy, vz = Vec
        Ex, Ey, Ez = self.Erf(Vec, t)
        return q ** 2 / 4 / m / wrf ** 2 * (Ex ** 2 + Ey ** 2)

    # Calculte for the kinetic energy

    def Ekin_without_axial(self, Vec, t):
        """
        kinetic energy without the axial direction at time = t
        Param:
            Vec: tuple 6
                Vector in the phase space at time = t
            t: float
                time
        Output: float
               kinetic energy without the axial direction at time = t
        """
        m = self.ParticleParameters['mass']
        x, y, z, vx, vy, vz = Vec
        return 0.5 * m * (vx ** 2 + vy ** 2)
    
    def DevMotion_Motion_Circuit(self, t, Vec_Motion_Circuit, L, r, progress_bar, WithNoise = True):
        """
        """
        dt = self.SimulationParameters['dt']
        q = self.ParticleParameters['charge']
        m = self.ParticleParameters['mass']
        deff = self.TrapParameters['deff']
        C = self.CircuitParameters['C']
        

        x, y, z, vx, vy, vz, Qc, Il = Vec_Motion_Circuit
        Vec_Circuit = Qc, Il
        Vec = x, y, z, vx, vy, vz
        Ex, Ey, Ez = tuple(map(sum, zip(self.Edc(Vec, t),
                                        self.Erf(Vec, t),
                                        (Qc / C / deff, 0, 0)
                                        )))
        
        ax = q * Ex / m
        ay = q * Ey / m
        az = q * Ez / m

        dQc_dt = q * vx / deff - Il
        dIl_dt = 1/L * (Qc / C - Il)

        progress_bar.update(dt / 6)
        return vx, vy, vz, ax, ay, az, dQc_dt, dIl_dt

    # Simulation is the parallel simulation for both tank circuit and the motion of the electron
    def Simulation(self, 
                  fres,
                  NumSecularPeriod,
                  DrawPosition = True, 
                  DrawVelocity = True, 
                  SaveData = False, 
                  DrawVelocityHist = True,
                  DrawEnergyHist = True):
        """
        Another method to simulate, by solving the equation of motion and the circuit 
        """
        # loading the Parameters from self
        q = self.ParticleParameters['charge']
        deff = self.TrapParameters['deff']
        dt = self.SimulationParameters['dt']
        T = self.SimulationParameters['TotalTime']
        Q = self.CircuitParameters['QualityFactor']
        C = self.CircuitParameters['C']
        N = int(np.round(T / dt))
        Temperature = self.CircuitParameters['Temperature']
        t_eval = np.linspace(0, T, N + 1)
        # calculate the inductance and r based on resonance frequency and quality factor respectively.
        L = 1 / (C * 4 * np.pi ** 2 * fres ** 2)
        r = 1 / (Q * 2 * np.pi * fres * C)
        # initial condition for the phase vector of the electron
        
        # initial condition for both the position & velocity & circuit
        # Version 1
        #Vec0 = self.Vec0 + (0, - vx0 * q / deff)
        Vec0_Motion_Circuit = self.Vec0 + (0, 0)
        progress_bar = tqdm(total=T, desc='Simulation with {:.2f} us'.format(T * 1e6), position=0, leave=True)
        solution = solve_ivp(fun=lambda t, Vec: self.DevMotion_Motion_Circuit(
                                                                    t, 
                                                                    Vec, 
                                                                    L,
                                                                    r, 
                                                                    progress_bar,
                                                                    True
                                                                    ), 
                                                                    t_span = (0, T), 
                                                                    y0 = Vec0_Motion_Circuit, 
                                                                    t_eval = t_eval, 
                                                                    first_step = dt,
                                                                    max_step = dt,
                                                                    atol = 1e-1,
                                                                    rtol = 1e-1,
                                                                    method = 'RK45')
        t_damp = solution.t
        VecResult = solution.y
        x_damp, y_damp, z_damp = VecResult[0, :], VecResult[1, :], VecResult[2, :]
        vx_damp, vy_damp, vz_damp = VecResult[3, :], VecResult[4, :], VecResult[5, :]
        Qc_damp, Il_damp = VecResult[6, :], VecResult[7, :]
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
        wradial = self.TrapParameters['wradial']


        if DrawVelocity:
            plt.clf()
            plt.plot(np.array(t_damp[:len(x_damp) // 500 * 500])*1e6, vx_damp[:len(x_damp) // 500 * 500],\
                    label = '$\Omega_rf$:{}GHz, $\omega_r$:{}GHz, $\omega_z$:{}MHz'.format(wrf/2/np.pi/1e9, wradial/2/np.pi/1e9, waxial/2/np.pi/1e6))
            plt.plot(np.array(t_damp[:len(x_damp) // 500 * 500])*1e6, vx_abs_init / np.sqrt(e) * np.ones(len(x_damp))[:len(x_damp) // 500 * 500], 'r--', label = '$1/\sqrt{e}$')
            plt.plot(np.array(t_damp[:len(x_damp) // 500 * 500])*1e6, -vx_abs_init / np.sqrt(e) * np.ones(len(x_damp))[:len(x_damp) // 500 * 500], 'r--')
            plt.plot(np.array(t_damp[peaks[-1]])*1e6, vx_damp[peaks[-1]], "x")
            plt.text(np.array(t_damp[peaks[-1]])*1e6, vx_damp[peaks[-1]], "%.2f $\mu$s" % (float(t_damp[peaks[-1]]) * 1e6))
            #ax[0].plot(np.array(t_eval[:len(x_damp)])*1e6, x_damp[:len(x_damp)])
           
            # Add estimated cooling time for secular cooling
            #if CoolingMode == 'secular':
                #plt.plot(m * deff ** 2 / q ** 2 / Rp * 1e6 *np.ones(1000), np.linspace(-2e5, 2e5, 1000), '--', color = 'orange', label = 'Theory Secular Cooling Time')

            # Add labels to the axes
            plt.xlabel('Time/$\mu$s')
            plt.ylabel('$v_x$')
            plt.legend()
            plt.grid()
            plt.ylim([-2e5, 2e5])
            #plt.title('fres = {:.2f} GHz, Rp {:.1f} M$\Omega$'.format(fres / 1e9, Rp/1e6))
            Zoomnum = round(0.2e-6/dt)
            x_zoom = np.array(t_eval[:Zoomnum // 1 * 1])*1e6
            y_zoom = vx_damp[:Zoomnum]
            axes = plt.axes([.50, .2, .23, .2])
            axes.plot(x_zoom, y_zoom, color='blue', lw=1, label='Zoomed curve')
            axes.set_xlabel('$\mu$s')
            #axes.legend(‘right’)

            plt.grid()

            FileName = '{}_Wrf=2PI{:.2f}GHz_Wradial=2PI{:.2f}GHz_Waxial=2PI{:.2f}MHz_T={:.2f}us'.format(CoolingMode, wrf / 2 / np.pi / 1e9, wradial / 2 / np.pi / 1e9, waxial / 2 / np.pi / 1e6, T * 1e6)
            
            plt.savefig('figures/' + FileName + '.png')
            plt.show(block=False)
            
        # Decide whether save data
        if SaveData:
            FileName = '{}_Wrf=2PI{:.2f}GHz_Wradial=2PI{:.2f}GHz_Waxial=2PI{:.2f}MHz_T={:.2f}us'.format(CoolingMode, wrf / 2 / np.pi / 1e9, wradial / 2 / np.pi / 1e9, waxial / 2 / np.pi / 1e6, T * 1e6)
            np.savez('data/damp_' + FileName, t = t_damp, x = x_damp, vx = vx_damp)
        
        if t_damp[peaks[-1]] / T > 0.9:
            print('Cooling Failed')
        else:
            print('By velocity plot, the cooling Time is: {:.3f} us'.format(t_damp[peaks[-1]] * 1e6))
        
        # Using the velocity plot to get a raw cooling time
        CoolingTime_rough = t_damp[peaks[-1]]
        
        # based on some tests, if the it's secular cooling, then after
        if CoolingMode == 'secular':
            EquilibriumTime = 5
        elif CoolingMode == 'blue':
            EquilibriumTime = 7
        else:
            EquilibriumTime = 5
        
        # we assume that after 5 cooling Time: e^-5, the motion is dominant by the noise
        t_damping = t_damp[:EquilibriumTime * round(CoolingTime_rough / dt)]
        x_damping = x_damp[:EquilibriumTime * round(CoolingTime_rough / dt)]
        y_damping = y_damp[:EquilibriumTime * round(CoolingTime_rough / dt)]
        z_damping = z_damp[:EquilibriumTime * round(CoolingTime_rough / dt)]
        vx_damping = vx_damp[:EquilibriumTime * round(CoolingTime_rough / dt)]
        vy_damping = vy_damp[:EquilibriumTime * round(CoolingTime_rough / dt)]
        vz_damping = vz_damp[:EquilibriumTime * round(CoolingTime_rough / dt)]

        if SaveData:
            FileName = '{}_Wrf=2PI{:.2f}GHz_Wradial=2PI{:.2f}GHz_Waxial=2PI{:.2f}MHz_T={:.2f}us'.format(CoolingMode, wrf / 2 / np.pi / 1e9, wradial / 2 / np.pi / 1e9, waxial / 2 / np.pi / 1e6, T * 1e6)
            np.savez('data/t_damping_' + FileName, t = t_damping, x = x_damping, vx = vx_damping)
            

        # Using Curve fitting to get the exact cooling Time
        
        Energy = []
        N = len(t_damping)
        SampleList = np.arange(0, N, 1 * NumSecularPeriod)
        SampleList = SampleList[:-1]
        # we didn't use the pesudopotential approximation till now.

        # all direction of x, y energy(potential and kinetic energy), average over a waxial period

        for i in SampleList:
            Energy_temp = 0
            for j in range(NumSecularPeriod):
                Vec = x_damping[i + j], y_damping[i + j], z_damping[i + j], vx_damping[i + j], vy_damping[i + j], vz_damping[i + j]
                Energy_temp = Energy_temp + \
                              self.Vdc_without_axial(Vec, t_damping[i + j]) + \
                              self.Vrf_without_axial(Vec, t_damping[i + j]) + \
                              self.Ekin_without_axial(Vec, t_damping[i + j])

            Energy.append(Energy_temp / NumSecularPeriod)

        SampleList = SampleList * dt

        Energy_fitfunc  = lambda p, x: p[0]*np.exp(- x/p[1])+p[2]
        Energy_errfunc  = lambda p, x, y: (y - Energy_fitfunc(p, x))
        init  = [1e1, CoolingTime_rough, 0.5]
        xdata = SampleList
        ydata = Energy
        out   = leastsq( Energy_errfunc, init, args=(xdata, ydata))
        c = out[0]
        plt.clf()
        plt.plot(xdata, ydata, label = 'raw data', alpha = 0.5)
        plt.plot(xdata, Energy_fitfunc(c, xdata), label = 'Fitting Curve')
        
        plt.title(r'$A = %.3f\ \sigma = %.8f\ k = %.3f $' %(c[0],abs(c[1]) * 1e6,c[2]))
        plt.xlabel('Time(s)')
        plt.legend('Energy(J)')
        FileName = 'Cooling_TIME_ENERGY_{}_Wrf=2PI{:.2f}GHz_Wradial=2PI{:.2f}GHz_Waxial=2PI{:.2f}MHz_T={:.2f}us'.format(CoolingMode, wrf / 2 / np.pi / 1e9, wradial / 2 / np.pi / 1e9, waxial / 2 / np.pi / 1e6, T * 1e6)
        plt.savefig('figures/' + FileName + '.png')
        plt.show(block = False)
        
        kB = 1.380649e-23 # Boltzman constant, given by Wikipedia
        print('By energy fitting, Cooling Time is: {:.3f} us'.format(abs(c[1]) * 1e6))
        print('By energy fitting, the final temperature is: {:.3f}K'.format(c[2] / kB))
        
        # Exact Cooling Time and rough
        CoolingTime = abs(c[1])
        FinalTemperature_rough = c[2] / kB

        Energy = []
        # based on some tests, if the it's secular cooling, then after
        if CoolingMode == 'secular':
            EquilibriumTime = 5
        elif CoolingMode == 'blue':
            EquilibriumTime = 7
        else:
            EquilibriumTime = 5
        #print(EquilibriumTime * round(CoolingTime_raw / dt))
        
        # we assume that after 5 cooling Time: e^-5, the motion is dominant by the noise
        t_damped = t_damp[EquilibriumTime * round(CoolingTime_rough / dt):]
        x_damped = x_damp[EquilibriumTime * round(CoolingTime_rough / dt):]
        y_damped = y_damp[EquilibriumTime * round(CoolingTime_rough / dt):]
        z_damped = z_damp[EquilibriumTime * round(CoolingTime_rough / dt):]
        vx_damped = vx_damp[EquilibriumTime * round(CoolingTime_rough / dt):]
        vy_damped = vy_damp[EquilibriumTime * round(CoolingTime_rough / dt):]
        vz_damped = vz_damp[EquilibriumTime * round(CoolingTime_rough / dt):]
        
        if SaveData:
            FileName = '{}_Wrf=2PI{:.2f}GHz_Wradial=2PI{:.2f}GHz_Waxial=2PI{:.2f}MHz_T={:.2f}us'.format(CoolingMode, wrf / 2 / np.pi / 1e9, wradial / 2 / np.pi / 1e9, waxial / 2 / np.pi / 1e6, T * 1e6)
            np.savez('data/t_damped_' + FileName, t = t_damped, x = x_damped, vx = vx_damped)

        
        '''
        by velocity square
        '''
        vx_damped_square = [a ** 2  * 0.5 * m / q * 1000 for a in vx_damped]
        # turn the unit to be the kinetic energy (eV)

        Velocity_Square_fitfunc  = lambda p, x: p[0]/np.sqrt(x-p[1])*np.exp(- (x-p[1])/p[2]) +p[3]
        Velocity_Square_errfunc  = lambda p, x, y: (y - Velocity_Square_fitfunc(p, x))
       
        plt.clf()
        Velocity_Square_n, Velocity_Square_bin_edges,_ = plt.hist(vx_damped_square, 200, density=True, color='green', alpha=0.75)
        #plt.title('Histogram for velocity')
        #plt.show()
        xdata = 0.5*(Velocity_Square_bin_edges[1:] + Velocity_Square_bin_edges[:-1])
        ydata = Velocity_Square_n
 

        init  = [0.5, 0., Temperature / q * kB * 1000, 0. ]

        out   = leastsq( Velocity_Square_errfunc, init, args=(xdata, ydata))
        c     = out[0]

        print("A exp[- (x-mu)/sigma)] + k ")
       
        print("Fit Coefficients:")
        print(c[0],c[1],abs(c[2]),c[3])
        if DrawVelocityHist:
            plt.plot(xdata, Velocity_Square_fitfunc(c, xdata), label = 'Fitting Curve')
            plt.plot(xdata, ydata, label = 'raw data')
            plt.legend()
            plt.xlabel('Kinetic Energy/meV')
            plt.ylabel('Probability')
            plt.title(r'$A = %.3f\  \mu = %.3f\  \sigma = %.3f\ k = %.3f $' %(c[0],c[1],abs(c[2]),c[3]))
            
            FileName = 'Histogram_for_Velocity_Square'
            plt.savefig('figures/' + FileName + '.png')
            plt.show(block = False)
            np.savez('data/xdata_' + FileName + '_' + CoolingMode, xdata)
            np.savez('data/ydata_' + FileName + '_' + CoolingMode, ydata)
        else:
            plt.show(block = False)
        
        
        print('By kinetic enrgy distribution, the final Temperature is {:.4f} K'.format(c[2] * q / kB / 1000))
        
        FinalTemperature = c[2] * q / kB / 1000

        return CoolingTime, FinalTemperature

    def SecondRun(self, 
                  Damping_Ex_Ampl, 
                  fres, 
                  phase, 
                  NumSecularPeriod,
                  DrawPosition = True, 
                  DrawVelocity = True, 
                  SaveData = False, 
                  DrawVelocityHist = True,
                  DrawEnergyHist = True
                  ):
        """
        Second Run with noise & damping circuit; The aim of the second simulation is to get the 

            1. cooling time;
            2. final temperature;
            There are generally two ways to get the cooling time:
                1.1: looking at the velocity plot and finding the place where v_damped = v_init / sqrt(e); the critical time is roughly the Cooling time(CoolingTime_raw)
                1.2: calculate the energy without axial direciton(kinetic energy + potential energy) at doing a curve fitting
            There are generally two ways to get the final temperature:
                2.1: when fitting the enrgy curve(time VS energy), the energy offset in the y direction(energy) corresponds to the final temperature
                2.2: after enough time when the cooling ends, the rest motion is dominated by the noise only; By doing satistics and histogram fitting, the enrgy(kinetic energy) obeys 1D Maxwell-Boltzmann distribution

        Param
        -------
            - Damping_Ex_Ampl 'float'
                got from the initial simulation, damping force electric field amplitude, got from initial simulation, used for damping force
            - fres: 'float'
                got from the initial simulation, resonator frequncy, used for damping force
            - phase: float
                got from the initial simulation, initial phase in the damping force list, got from initial simulation, used for damping force
            - NumSecularPeriod: integer
                got from the first simulation, used for calculation for the average energy during one secular period
            - DrawPosition: boolean
                determine whether draw & save the time VS position plot or not
            - DrawVelocity: boolean
                determine whether draw & save the time VS velocity plot or not  
            - SaveData: boolean
                determine whether save the data(x, y, z, vx, vy, vz) during the simulation or not
            - DrawVelocityHist: boolean
                determine whether save the Velocity distribution plot and data in x direction
            - DrawEnergyHist: boolean
                determine whether save the energy distribution plot and data
        
        Output
        -----
            - Damping_Ex_Ampl: float
                Damping Force Amplitude(The maximum damping force)
            - fres: float
                resonator frequency, which is determined by cooling mode and the motional spectrum
            - phase: float, in [0, 2Pi]
                initial phase in the damping force list
            - NumSecularPeriod: integer
                the number of the time steps in a radial direction secular motion period
        """
        Aim = self.SimulationParameters['Aim']

        T = self.SimulationParameters['TotalTime']
        dt = self.SimulationParameters['dt']
        CoolingMode = self.SimulationParameters['CoolingMode']
        N = int(np.round(T / dt))
        Vec0 = self.Vec0 + self.Vec0
        progress_bar = tqdm(total=T, desc='Simulation for Second Run with {:.2f} us'.format(T * 1e6), position=0, leave=True)
        
        # generating the time list
        t_eval = np.linspace(0, T, N + 1)

        # Generate the noise List in advance
        deff = self.TrapParameters['deff']
        Temperature = self.CircuitParameters['Temperature']
        Rp = self.CircuitParameters['Rp']
        QualityFactor = self.CircuitParameters['QualityFactor']
        
        kB = 1.380649e-23 # Boltzman constant, given by Wikipedias
        # Call the function "noise_on_trap" to get a Johnson Nyquist noise electric field list
        # Noise Model Version 1: White Noise with Filter, RLC parallel cirucit
        JNNoise_Ex = noise_on_trap_FDT(deff, Temperature, Rp, fres, QualityFactor, T, dt)
        
        # Noise Model Version 2: Purely White Noise, but with spectrum amplitude suppressed
        if CoolingMode == 'secular':
            ratio = 0.4
        elif CoolingMode == 'blue':
            ratio = 0.3
        else:
            ratio = 0.3
        #JNNoise_Ex = noise_on_trap_V2(deff, Temperature, Rp, fres, QualityFactor, T, dt, ratio)

        # Using RK45 to solve the ODE
        solution = solve_ivp(fun=lambda t, Vec: self.DevMotion_para(t, 
                                                                    Vec, 
                                                                    Damping_Ex_Ampl, 
                                                                    fres, 
                                                                    phase, 
                                                                    progress_bar, 
                                                                    JNNoise_Ex, 
                                                                    WithNoise = True), 
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
        wradial = self.TrapParameters['wradial']


        if DrawVelocity:
            plt.clf()
            plt.plot(np.array(t_damp[:len(x_damp) // 500 * 500])*1e6, vx_damp[:len(x_damp) // 500 * 500],\
                    label = '$\Omega_rf$:{}GHz, $\omega_r$:{}GHz, $\omega_z$:{}MHz'.format(wrf/2/np.pi/1e9, wradial/2/np.pi/1e9, waxial/2/np.pi/1e6))
            plt.plot(np.array(t_damp[:len(x_damp) // 500 * 500])*1e6, vx_abs_init / np.sqrt(e) * np.ones(len(x_damp))[:len(x_damp) // 500 * 500], 'r--', label = '$1/\sqrt{e}$')
            plt.plot(np.array(t_damp[:len(x_damp) // 500 * 500])*1e6, -vx_abs_init / np.sqrt(e) * np.ones(len(x_damp))[:len(x_damp) // 500 * 500], 'r--')
            plt.plot(np.array(t_damp[peaks[-1]])*1e6, vx_damp[peaks[-1]], "x")
            plt.text(np.array(t_damp[peaks[-1]])*1e6, vx_damp[peaks[-1]], "%.2f $\mu$s" % (float(t_damp[peaks[-1]]) * 1e6))
            #ax[0].plot(np.array(t_eval[:len(x_damp)])*1e6, x_damp[:len(x_damp)])
           
            # Add estimated cooling time for secular cooling
            if CoolingMode == 'secular':
                plt.plot(m * deff ** 2 / q ** 2 / Rp * 1e6 *np.ones(1000), np.linspace(-2e5, 2e5, 1000), '--', color = 'orange', label = 'Theory Secular Cooling Time')

            # Add labels to the axes
            plt.xlabel('Time/$\mu$s')
            plt.ylabel('$v_x$')
            plt.legend()
            plt.grid()
            plt.ylim([-2e5, 2e5])
            plt.title('fres = {:.2f} GHz, Rp {:.1f} M$\Omega$'.format(fres / 1e9, Rp/1e6))
            Zoomnum = round(0.2e-6/dt)
            x_zoom = np.array(t_eval[:Zoomnum // 1 * 1])*1e6
            y_zoom = vx_damp[:Zoomnum]
            axes = plt.axes([.50, .2, .23, .2])
            axes.plot(x_zoom, y_zoom, color='blue', lw=1, label='Zoomed curve')
            axes.set_xlabel('$\mu$s')
            #axes.legend(‘right’)

            plt.grid()

            FileName = '{}_Wrf=2PI{:.2f}GHz_Wradial=2PI{:.2f}GHz_Waxial=2PI{:.2f}MHz_T={:.2f}us'.format(CoolingMode, wrf / 2 / np.pi / 1e9, wradial / 2 / np.pi / 1e9, waxial / 2 / np.pi / 1e6, T * 1e6)
            
            plt.savefig('figures/' + FileName + '.png')
            plt.show(block=False)
            
        # Decide whether save data
        if SaveData:
            FileName = '{}_Wrf=2PI{:.2f}GHz_Wradial=2PI{:.2f}GHz_Waxial=2PI{:.2f}MHz_T={:.2f}us'.format(CoolingMode, wrf / 2 / np.pi / 1e9, wradial / 2 / np.pi / 1e9, waxial / 2 / np.pi / 1e6, T * 1e6)
            np.savez('data/t_damp_' + FileName, t = t_damp, x = x_damp, vx = vx_damp)
            
        if Aim == 'CoolingTime+FinalTemperature':
            if t_damp[peaks[-1]] / T > 0.9:
                print('Cooling Failed')
                if CoolingMode == 'secular':
                    CoolingTime_rough = m * deff ** 2 / q ** 2 / Rp 
                elif CoolingMode == 'blue':
                    CoolingTime_rough = np.sqrt(2) * wrf / wradial * m * deff ** 2 / q ** 2 / Rp
                else:
                    CoolingTime_rough = t_damp[peaks[-1]]
            else:
                print('By velocity plot, the cooling Time is: {:.3f} us'.format(t_damp[peaks[-1]] * 1e6))
                # Using the velocity plot to get a raw cooling time
                CoolingTime_rough = t_damp[peaks[-1]]
        
            # based on some tests, if the it's secular cooling, then after
            if CoolingMode == 'secular':
                EquilibriumTime = 5
            elif CoolingMode == 'blue':
                EquilibriumTime = 7
            else:
                EquilibriumTime = 5
            
            # we assume that after 5 cooling Time: e^-5, the motion is dominant by the noise
            t_damping = t_damp[:EquilibriumTime * round(CoolingTime_rough / dt)]
            x_damping = x_damp[:EquilibriumTime * round(CoolingTime_rough / dt)]
            y_damping = y_damp[:EquilibriumTime * round(CoolingTime_rough / dt)]
            z_damping = z_damp[:EquilibriumTime * round(CoolingTime_rough / dt)]
            vx_damping = vx_damp[:EquilibriumTime * round(CoolingTime_rough / dt)]
            vy_damping = vy_damp[:EquilibriumTime * round(CoolingTime_rough / dt)]
            vz_damping = vz_damp[:EquilibriumTime * round(CoolingTime_rough / dt)]
        
        elif Aim == 'CoolingTime':
            t_damping = t_damp
            x_damping = x_damp
            y_damping = y_damp
            z_damping = z_damp
            vx_damping = vx_damp
            vy_damping = vy_damp
            vz_damping = vz_damp
        
        if Aim != 'FinalTemperature':

            if SaveData:
                FileName = '{}_Wrf=2PI{:.2f}GHz_Wradial=2PI{:.2f}GHz_Waxial=2PI{:.2f}MHz_T={:.2f}us'.format(CoolingMode, wrf / 2 / np.pi / 1e9, wradial / 2 / np.pi / 1e9, waxial / 2 / np.pi / 1e6, T * 1e6)
                np.savez('data/damping_' + FileName, t = t_damping, x = x_damping, vx = vx_damping)
                
            # Using Curve fitting to get the exact cooling Time
            
            Energy = []
            N = len(t_damping)
            SampleList = np.arange(0, N, 1 * NumSecularPeriod)
            SampleList = SampleList[:-1]
            # we didn't use the pesudopotential approximation till now.

            # all direction of x, y energy(potential and kinetic energy), average over a waxial period

            for i in SampleList:
                Energy_temp = 0
                for j in range(NumSecularPeriod):
                    Vec = x_damping[i + j], y_damping[i + j], z_damping[i + j], vx_damping[i + j], vy_damping[i + j], vz_damping[i + j]
                    Energy_temp = Energy_temp + \
                                self.Vdc_without_axial(Vec, t_damping[i + j]) + \
                                self.Vrf_without_axial(Vec, t_damping[i + j]) + \
                                self.Ekin_without_axial(Vec, t_damping[i + j])

                Energy.append(Energy_temp / NumSecularPeriod)

            SampleList = SampleList * dt

            Energy_fitfunc  = lambda p, x: p[0]*np.exp(- x/p[1])+p[2]
            Energy_errfunc  = lambda p, x, y: (y - Energy_fitfunc(p, x))
            init  = [1e1, CoolingTime_rough, 0.5]
            xdata = SampleList
            ydata = Energy
            out   = leastsq( Energy_errfunc, init, args=(xdata, ydata))
            c = out[0]
            plt.clf()
            plt.plot(xdata, ydata, label = 'raw data', alpha = 0.5)
            plt.plot(xdata, Energy_fitfunc(c, xdata), label = 'Fitting Curve')
            
            plt.title(r'$A = %.3f\ \sigma = %.8f\ k = %.3f $' %(c[0],abs(c[1]) * 1e6,c[2]))
            plt.xlabel('Time(s)')
            plt.legend('Energy(J)')
            FileName = 'Cooling_TIME_ENERGY_{}_Wrf=2PI{:.2f}GHz_Wradial=2PI{:.2f}GHz_Waxial=2PI{:.2f}MHz_T={:.2f}us'.format(CoolingMode, wrf / 2 / np.pi / 1e9, wradial / 2 / np.pi / 1e9, waxial / 2 / np.pi / 1e6, T * 1e6)
            plt.savefig('figures/' + FileName + '.png')
            plt.show(block = False)
            
            
            print('By energy fitting, Cooling Time is: {:.3f} us'.format(abs(c[1]) * 1e6))
            print('By energy fitting, the final temperature is: {:.3f}K'.format(c[2] / kB))
            
            # Exact Cooling Time and rough
            CoolingTime = abs(c[1])
            FinalTemperature_rough = c[2] / kB
        
        if Aim == 'CoolingTime':
            return CoolingTime

        Energy = []

        if Aim == 'CoolingTime+FinalTemperature':
            # based on some tests, if the it's secular cooling, then after
            if CoolingMode == 'secular':
                EquilibriumTime = 5
            elif CoolingMode == 'blue':
                EquilibriumTime = 7
            else:
                EquilibriumTime = 5
            #print(EquilibriumTime * round(CoolingTime_raw / dt))
            
            # we assume that after 5 cooling Time: e^-5, the motion is dominant by the noise
            t_damped = t_damp[EquilibriumTime * round(CoolingTime_rough / dt):]
            x_damped = x_damp[EquilibriumTime * round(CoolingTime_rough / dt):]
            y_damped = y_damp[EquilibriumTime * round(CoolingTime_rough / dt):]
            z_damped = z_damp[EquilibriumTime * round(CoolingTime_rough / dt):]
            vx_damped = vx_damp[EquilibriumTime * round(CoolingTime_rough / dt):]
            vy_damped = vy_damp[EquilibriumTime * round(CoolingTime_rough / dt):]
            vz_damped = vz_damp[EquilibriumTime * round(CoolingTime_rough / dt):]
            
            if SaveData:
                FileName = '{}_Wrf=2PI{:.2f}GHz_Wradial=2PI{:.2f}GHz_Waxial=2PI{:.2f}MHz_T={:.2f}us'.format(CoolingMode, wrf / 2 / np.pi / 1e9, wradial / 2 / np.pi / 1e9, waxial / 2 / np.pi / 1e6, T * 1e6)
                np.savez('data/damped_' + FileName, t = t_damped, x = x_damped, vx = vx_damped)
        
        elif Aim == 'FinalTemperature':
            t_damped = t_damp
            x_damped = x_damp
            y_damped = y_damp
            z_damped = z_damp
            vx_damped = vx_damp
            vy_damped = vy_damp
            vz_damped = vz_damp
        
        else:
            print('AIM ERROR!')

        '''
        by velocity square
        '''
        vx_damped_square = [a ** 2  * 0.5 * m / q * 1000 for a in vx_damped]
        # turn the unit to be the kinetic energy (eV)

        Velocity_Square_fitfunc  = lambda p, x: p[0]/np.sqrt(x-p[1])*np.exp(- (x-p[1])/p[2]) +p[3]
        Velocity_Square_errfunc  = lambda p, x, y: (y - Velocity_Square_fitfunc(p, x))
       
        plt.clf()
        Velocity_Square_n, Velocity_Square_bin_edges,_ = plt.hist(vx_damped_square, 200, density=True, color='green', alpha=0.75)
        #plt.title('Histogram for velocity')
        #plt.show()
        xdata = 0.5*(Velocity_Square_bin_edges[1:] + Velocity_Square_bin_edges[:-1])
        ydata = Velocity_Square_n
 

        init  = [0.5, 0., Temperature / q * kB * 1000, 0. ]

        out   = leastsq( Velocity_Square_errfunc, init, args=(xdata, ydata))
        c     = out[0]

        print("A exp[- (x-mu)/sigma)] + k ")
       
        print("Fit Coefficients:")
        print(c[0],c[1],abs(c[2]),c[3])
        if DrawVelocityHist:
            plt.plot(xdata, Velocity_Square_fitfunc(c, xdata), label = 'Fitting Curve')
            plt.plot(xdata, ydata, label = 'raw data')
            plt.legend()
            plt.xlabel('Kinetic Energy/meV')
            plt.ylabel('Probability')
            plt.title(r'$A = %.3f\  \mu = %.3f\  \sigma = %.3f\ k = %.3f $' %(c[0],c[1],abs(c[2]),c[3]))
            
            FileName = 'Histogram_for_Velocity_Square'
            plt.savefig('figures/' + FileName + '.png')
            plt.show(block = False)
            np.save('data/xdata_' + FileName + '_' + CoolingMode, xdata)
            np.save('data/ydata_' + FileName + '_' + CoolingMode, ydata)
        else:
            plt.show(block = False)
        
        
        print('By kinetic enrgy distribution, the final Temperature is {:.4f} K'.format(c[2] * q / kB / 1000))
        
        FinalTemperature = c[2] * q / kB / 1000

        if Aim == 'CoolingTime+FinalTemperature':
            return CoolingTime, FinalTemperature
        
        elif Aim == 'FinalTemperature':
            return FinalTemperature

        else:
            print('AIM ERROR!')
    

    def Run(self):
        #sys.modules[__name__].__dict__.clear()
        #get_ipython().magic('reset -sf')
        '''
        main
        '''
        T = self.SimulationParameters['TotalTime']
        dt = self.SimulationParameters['dt']
        CoolingMode = self.SimulationParameters['CoolingMode']
        wrf = self.TrapParameters['wrf']
        wradial = self.TrapParameters['wradial']
        waxial = self.TrapParameters['waxial']
        Rp = self.CircuitParameters['Rp']
        Aim = self.SimulationParameters['Aim']

        print(Aim)

        FileName = '{}_{}_Wrf=2PI{:.2f}GHz_Wradial=2PI{:.2f}GHz_Waxial=2PI{:.2f}MHz_T={:.2f}us_Rp={:.2f}kOhm'.format(Aim, CoolingMode, wrf / 2 / np.pi / 1e9, wradial / 2 / np.pi / 1e9, waxial / 2 / np.pi / 1e6, T * 1e6, Rp / 1e3)


        # Version 1: Simplified circuit model, just simulation of electron motion, but using damping factor
        if Aim == 'CoolingTime+FinalTemperature':
            OtherFileName = 'CoolingTime_{}_Wrf=2PI{:.2f}GHz_Wradial=2PI{:.2f}GHz_Waxial=2PI{:.2f}MHz_T={:.2f}us_Rp={:.2f}kOhm'.format(CoolingMode, wrf / 2 / np.pi / 1e9, wradial / 2 / np.pi / 1e9, waxial / 2 / np.pi / 1e6, T * 1e6, Rp / 1e3)
        
            if os.path.exists('data/' + FileName + '_Initial_Simulation_Results.npz'):
                print('Loading Previous Initial Simulation Results...')
                Damping_Ex_Ampl = np.load('data/' + FileName + '_Initial_Simulation_Results.npz')['Damping_Ex_Ampl']
                fres = np.load('data/' + FileName + '_Initial_Simulation_Results.npz')['fres']
                phase = np.load('data/' + FileName + '_Initial_Simulation_Results.npz')['phase']
                NumSecularPeriod = np.load('data/' + FileName + '_Initial_Simulation_Results.npz')['NumSecularPeriod']
            
            elif os.path.exists('data/' + OtherFileName + '_Initial_Simulation_Results.npz'):
                print('Loading Previous Initial Simulation Results...')
                Damping_Ex_Ampl = np.load('data/' + OtherFileName + '_Initial_Simulation_Results.npz')['Damping_Ex_Ampl']
                fres = np.load('data/' + OtherFileName + '_Initial_Simulation_Results.npz')['fres']
                phase = np.load('data/' + OtherFileName + '_Initial_Simulation_Results.npz')['phase']
                NumSecularPeriod = np.load('data/' + OtherFileName + '_Initial_Simulation_Results.npz')['NumSecularPeriod']

            else:
                Damping_Ex_Ampl, fres, phase, NumSecularPeriod = self.InitialRun(False, False)
                np.savez('data/' + FileName + '_Initial_Simulation_Results', Damping_Ex_Ampl = Damping_Ex_Ampl, fres = fres, phase = phase, NumSecularPeriod = NumSecularPeriod)
            
            Cooling_Time, FinalTemperature = self.SecondRun(Damping_Ex_Ampl, fres, phase, NumSecularPeriod,
                                                        DrawPosition = False,
                                                        DrawVelocity = True, 
                                                        SaveData = True
                                                        )
            
            return Cooling_Time, FinalTemperature
        
        elif Aim == 'CoolingTime':
            OtherFileName = 'CoolingTime+FinalTemperature_{}_Wrf=2PI{:.2f}GHz_Wradial=2PI{:.2f}GHz_Waxial=2PI{:.2f}MHz_T={:.2f}us_Rp={:.2f}kOhm'.format(CoolingMode, wrf / 2 / np.pi / 1e9, wradial / 2 / np.pi / 1e9, waxial / 2 / np.pi / 1e6, T * 1e6, Rp / 1e3)
        
            if os.path.exists('data/' + FileName + '_Initial_Simulation_Results.npz'):
                print('Loading Previous Initial Simulation Results...')
                Damping_Ex_Ampl = np.load('data/' + FileName + '_Initial_Simulation_Results.npz')['Damping_Ex_Ampl']
                fres = np.load('data/' + FileName + '_Initial_Simulation_Results.npz')['fres']
                phase = np.load('data/' + FileName + '_Initial_Simulation_Results.npz')['phase']
                NumSecularPeriod = np.load('data/' + FileName + '_Initial_Simulation_Results.npz')['NumSecularPeriod']
            
            elif os.path.exists('data/' + OtherFileName + '_Initial_Simulation_Results.npz'):
                print('Loading Previous Initial Simulation Results...')
                Damping_Ex_Ampl = np.load('data/' + OtherFileName + '_Initial_Simulation_Results.npz')['Damping_Ex_Ampl']
                fres = np.load('data/' + OtherFileName + '_Initial_Simulation_Results.npz')['fres']
                phase = np.load('data/' + OtherFileName + '_Initial_Simulation_Results.npz')['phase']
                NumSecularPeriod = np.load('data/' + OtherFileName + '_Initial_Simulation_Results.npz')['NumSecularPeriod']

            else:
                Damping_Ex_Ampl, fres, phase, NumSecularPeriod = self.InitialRun(False, False)
                np.savez('data/' + FileName + '_Initial_Simulation_Results', Damping_Ex_Ampl = Damping_Ex_Ampl, fres = fres, phase = phase, NumSecularPeriod = NumSecularPeriod)
            
            Cooling_Time = self.SecondRun(Damping_Ex_Ampl, fres, phase, NumSecularPeriod,
                                                        DrawPosition = False,
                                                        DrawVelocity = True, 
                                                        SaveData = True
                                                        )
            return Cooling_Time

        elif Aim == 'FinalTemperature':
            if os.path.exists('data/' + FileName + '_Initial_Simulation_Results.npz'):
                print('Loading Previous Initial Simulation Results...')
                Damping_Ex_Ampl = np.load('data/' + FileName + '_Initial_Simulation_Results.npz')['Damping_Ex_Ampl']
                fres = np.load('data/' + FileName + '_Initial_Simulation_Results.npz')['fres']
                phase = np.load('data/' + FileName + '_Initial_Simulation_Results.npz')['phase']
                NumSecularPeriod = np.load('data/' + FileName + '_Initial_Simulation_Results.npz')['NumSecularPeriod']
            
            else:
                Damping_Ex_Ampl, fres, phase, NumSecularPeriod = self.InitialRun(False, False)
                np.savez('data/' + FileName + '_Initial_Simulation_Results', Damping_Ex_Ampl = Damping_Ex_Ampl, fres = fres, phase = phase, NumSecularPeriod = NumSecularPeriod)
            Final_Temperature = self.SecondRun(Damping_Ex_Ampl, fres, phase, NumSecularPeriod,
                                                        DrawPosition = False,
                                                        DrawVelocity = True, 
                                                        SaveData = True
                                                        )
            return Final_Temperature

        else:
            print('AIM ERROR!')
            pass
            
        
        # Version 2: Detailed circuit model, simulation of both electron motion and circuit
        '''
        fres, NumSecularPeriod = self.InitialSimulation(False, False)
        Cooling_Time, FinalTemperature = self.Simulation(
                                                        fres,
                                                        NumSecularPeriod,
                                                        DrawPosition= True,
                                                        DrawVelocity = True,
                                                        SaveData = False,
                                                        DrawVelocityHist= True
        )
        '''
        

