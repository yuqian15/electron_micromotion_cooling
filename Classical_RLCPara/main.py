import os.path
import Particle
import numpy as np
import csv


wrf = 2 * np.pi* 10.6e9 # 10.6 GHz
wradial = 2 * np.pi * 2e9 # 2 GHz  
waxial = 2 * np.pi * 300e6 # 300 MHz
deff = 138e-6 # 138 micron
m = 9.10938297e-31 # 9.10938297e-31 kg 
q = 1.6e-19 # 1.6e-19 C
kB = 1.380649e-23
CircuitTemperature = 4 # Temperature of the tank circuit = 4K
TotalTime = 10e-6
#Rp = 8e4 #8e4
Rp = 8e6 # for faster simulation
Q = 1000
qx = wradial / wrf

# CoolingMode = 'secular'/'blue'/'red'/ float
CoolingMode = input('What cooling mode do you want? 1. "secular"; 2. "blue"; 3. "red"; 4. float\n')
#CoolingMode = 'blue'
#CoolingMode = 'secular'
# Aim = 'CoolingTime' / 'FinalTemperture'/ 'CoolingTime_FinalTemperature'
Aim = 'CoolingTime_FinalTemperature'
#Aim = 'CoolingTime'

# REMINDER
# when Aim == 'CoolingTime', it is suggested to make WithNoise = False in the Particle.py
# but when Aim != 'CoolingTime', it is suggested to make WithNoise = True in the Particle.py

#if __name__ == "__main__":

wradialList = np.linspace(2 * np.pi * 1.5e9, 2 * np.pi * 2.5e9, 10)
#wradialList = wradial * np.ones(1)
wrfList = np.linspace(2 * np.pi * 8e9, 2 * np.pi * 12e9, 10)
#wradialList = wradial * np.ones(10)
#wrfList = np.linspace(2 * np.pi * 5e9, 2 * np.pi * 15e9, 1)
CoolingTime = []
FinalTemperature = []
if Aim == 'CoolingTime' or Aim == 'CoolingTime_FinalTemperature':
    x0, y0, z0, vx0, vy0, vz0 = 10e-6, 10e-6, 5e-6, 1., 1., 1.
    # arbitray choice of initial condition, the initial motional temperture ~ 2000K
    Vec0 = x0, y0, z0, vx0, vy0, vz0
elif Aim == 'FinalTemperature':
    x0, y0, z0, vx0, vy0, vz0 = 0., 0., 0., np.sqrt(kB * CircuitTemperature / m), np.sqrt(kB * CircuitTemperature / m), np.sqrt(kB * CircuitTemperature / m)
    Vec0 = x0, y0, z0, vx0, vy0, vz0
else:
    print('AIM ERROR!')

scantype = input('What trap paramerterization do you want? 1. "wradial"; 2. "wrf"; 3. "qx"\n')

if scantype == 'wradial':
    for wradial in wradialList:
        print(Aim + ' ' + CoolingMode + ' for wrf = 2pi*{:.2f}, wradial = 2pi*{:.2f}, waxial = 2pi*{:.2f} in {:.2f} us'.format(wrf/(2 * np.pi),wradial/(2 * np.pi), waxial/(2 * np.pi), TotalTime * 1e6))
        
        test = Particle.Sinlge_Electron_Cooling(Vec0,
                                                ParticleParameters={
                                                    'mass': m,
                                                    'charge': q,
                                                },
                                                TrapParameters={
                                                    'wrf': wrf,
                                                    'wradial': wradial,
                                                    'waxial': waxial,
                                                    'deff': deff
                                                },
                                                CircuitParameters={
                                                    #'L': 6e-9, # Impedance should be decided based on the cooling mode
                                                    'C': 1e-12,
                                                    #'r': 1e3, # r should be determined by the quality factor and 
                                                    'Rp': Rp, 
                                                    'QualityFactor': Q,
                                                    'Temperature': CircuitTemperature,
                                                },
                                                SimulationParameters={
                                                    'dt': 1e-12,
                                                    'CoolingMode': CoolingMode,
                                                    'TotalTime': TotalTime,
                                                    'Aim': Aim,
                                                })
        if Aim == 'FinalTemperature':
            FinalTemperature_temp = test.Run()
            FinalTemperature.append(FinalTemperature_temp)
        elif Aim == 'CoolingTime':
            CoolingTime_temp = test.Run()
            CoolingTime.append(CoolingTime_temp)
        elif Aim == 'CoolingTime_FinalTemperature':
            CoolingTime_temp, FinalTemperature_temp = test.Run()
            CoolingTime.append(CoolingTime_temp)
            FinalTemperature.append(FinalTemperature_temp)
        else:
            print('AIM ERROR!')
        
        

    FileName = '{}_{}_T={:.2f}us_Rp={:.2f}kOhm.csv'.format(Aim, CoolingMode, TotalTime * 1e6, Rp / 1e3)
    # save data
    #np.save(CoolingMode + 'wrf_changing, wradical=2pi*{:.2f},waxial=2pi*{:.2f} in {:.2f} us.txt'.format(wrf/(2 * np.pi),wradical/(2 * np.pi), waxial/(2 * np.pi), TotalTime * 1e6), CoolingTime)
    with open(FileName, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        if Aim == 'FinalTemperature':
            wr.writerow(wradialList)
            wr.writerow(FinalTemperature)
        elif Aim == 'CoolingTime':
            wr.writerow(wradialList)
            wr.writerow(CoolingTime)
        elif Aim == 'CoolingTime_FinalTemperature':
            wr.writerow(wradialList)
            wr.writerow(CoolingTime)
            wr.writerow(FinalTemperature)

elif scantype == 'wrf':
    for wrf in wrfList:
        print(Aim + ' ' + CoolingMode + ' for wrf = 2pi*{:.2f}, wradial = 2pi*{:.2f}, waxial = 2pi*{:.2f} in {:.2f} us'.format(wrf/(2 * np.pi),wradial/(2 * np.pi), waxial/(2 * np.pi), TotalTime * 1e6))
        
        test = Particle.Sinlge_Electron_Cooling(Vec0,
                                                ParticleParameters={
                                                    'mass': m,
                                                    'charge': q,
                                                },
                                                TrapParameters={
                                                    'wrf': wrf,
                                                    'wradial': wradial,
                                                    'waxial': waxial,
                                                    'deff': deff
                                                },
                                                CircuitParameters={
                                                    #'L': 6e-9, # Impedance should be decided based on the cooling mode
                                                    'C': 1e-12,
                                                    #'r': 1e3, # r should be determined by the quality factor and 
                                                    'Rp': Rp, 
                                                    'QualityFactor': Q,
                                                    'Temperature': CircuitTemperature,
                                                },
                                                SimulationParameters={
                                                    'dt': 1e-12,
                                                    'CoolingMode': CoolingMode,
                                                    'TotalTime': TotalTime,
                                                    'Aim': Aim,
                                                })
        if Aim == 'FinalTemperature':
            FinalTemperature_temp = test.Run()
            FinalTemperature.append(FinalTemperature_temp)
        elif Aim == 'CoolingTime':
            CoolingTime_temp = test.Run()
            CoolingTime.append(CoolingTime_temp)
        elif Aim == 'CoolingTime_FinalTemperature':
            CoolingTime_temp, FinalTemperature = test.Run()
            CoolingTime.append(CoolingTime_temp)
            FinalTemperature.append(FinalTemperature_temp)
        else:
            print('AIM ERROR!')
        
        

    FileName = '{}_{}_T={:.2f}us_Rp={:.2f}kOhm.csv'.format(Aim, CoolingMode, TotalTime * 1e6, Rp / 1e3)
    # save data
    #np.save(CoolingMode + 'wrf_changing, wradical=2pi*{:.2f},waxial=2pi*{:.2f} in {:.2f} us.txt'.format(wrf/(2 * np.pi),wradical/(2 * np.pi), waxial/(2 * np.pi), TotalTime * 1e6), CoolingTime)
    with open(FileName, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        if Aim == 'FinalTemperature':
            wr.writerow(wradialList)
            wr.writerow(FinalTemperature)
        elif Aim == 'CoolingTime':
            wr.writerow(wradialList)
            wr.writerow(CoolingTime)
        elif Aim == 'CoolingTime_FinalTemperature':
            wr.writerow(wradialList)
            wr.writerow(CoolingTime)
            wr.writerow(FinalTemperature)
elif scantype == 'qx':
    for wrf in wrfList:
        wradial = qx * wrf
        print(Aim + ' ' + CoolingMode + ' for wrf = 2pi*{:.2f}, wradial = 2pi*{:.2f}, waxial = 2pi*{:.2f} in {:.2f} us'.format(wrf/(2 * np.pi),wradial/(2 * np.pi), waxial/(2 * np.pi), TotalTime * 1e6))
        
        test = Particle.Sinlge_Electron_Cooling(Vec0,
                                                ParticleParameters={
                                                    'mass': m,
                                                    'charge': q,
                                                },
                                                TrapParameters={
                                                    'wrf': wrf,
                                                    'wradial': wradial,
                                                    'waxial': waxial,
                                                    'deff': deff
                                                },
                                                CircuitParameters={
                                                    #'L': 6e-9, # Impedance should be decided based on the cooling mode
                                                    'C': 1e-12,
                                                    #'r': 1e3, # r should be determined by the quality factor and 
                                                    'Rp': Rp, 
                                                    'QualityFactor': Q,
                                                    'Temperature': CircuitTemperature,
                                                },
                                                SimulationParameters={
                                                    'dt': 1e-12,
                                                    'CoolingMode': CoolingMode,
                                                    'TotalTime': TotalTime,
                                                    'Aim': Aim,
                                                })
        if Aim == 'FinalTemperature':
            FinalTemperature_temp = test.Run()
            FinalTemperature.append(FinalTemperature_temp)
        elif Aim == 'CoolingTime':
            CoolingTime_temp = test.Run()
            CoolingTime.append(CoolingTime_temp)
        elif Aim == 'CoolingTime_FinalTemperature':
            CoolingTime_temp, FinalTemperature = test.Run()
            CoolingTime.append(CoolingTime_temp)
            FinalTemperature.append(FinalTemperature_temp)
        else:
            print('AIM ERROR!')
        
        

    FileName = '{}_{}_T={:.2f}us_Rp={:.2f}kOhm.csv'.format(Aim, CoolingMode, TotalTime * 1e6, Rp / 1e3)
    # save data
    #np.save(CoolingMode + 'wrf_changing, wradical=2pi*{:.2f},waxial=2pi*{:.2f} in {:.2f} us.txt'.format(wrf/(2 * np.pi),wradical/(2 * np.pi), waxial/(2 * np.pi), TotalTime * 1e6), CoolingTime)
    with open(FileName, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        if Aim == 'FinalTemperature':
            wr.writerow(wradialList)
            wr.writerow(FinalTemperature)
        elif Aim == 'CoolingTime':
            wr.writerow(wradialList)
            wr.writerow(CoolingTime)
        elif Aim == 'CoolingTime_FinalTemperature':
            wr.writerow(wradialList)
            wr.writerow(CoolingTime)
            wr.writerow(FinalTemperature)
else:
    print('scann type error')