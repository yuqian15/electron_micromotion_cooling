import os.path
import Particle
import numpy as np
import csv

x0, y0, z0, vx0, vy0, vz0 = 10e-6, 10e-6, 5e-6, 0.1, 0.1, 0.1
Vec0 = x0, y0, z0, vx0, vy0, vz0
wrf = 2 * np.pi* 10e9 # 10 GHz
wradical = 2 * np.pi * 1e9 # 2 GHz  
waxial = 2 * np.pi * 300e6 # 300 MHz
deff = 200e-6 # 200 micron
m = 9.10938297e-31 # 9.10938297e-31 kg 
q = 1.6e-19 # 1.6e-19 C
TotalTime = 40e-6
Rp = 5e6
Q = 2000

CoolingMode = 'secular'
#if __name__ == "__main__":

wrfList = np.linspace(2 * np.pi * 5e9, 2 * np.pi * 15e9, 2)
CoolingTime = []
for wrf in wrfList:
    print(CoolingMode + 'for wrf = 2pi*{:.2f}, wradical = 2pi*{:.2f}, waxial = 2pi*{:.2f} in {:.2f} us'.format(wrf/(2 * np.pi),wradical/(2 * np.pi), waxial/(2 * np.pi), TotalTime * 1e6))
    test = Particle.Sinlge_Electron_Cooling(Vec0,
                                            ParticleParameters={
                                                'mass': m,
                                                'charge': q
                                            },
                                            TrapParameters={
                                                'wrf': wrf,
                                                'wradical': wradical,
                                                'waxial': waxial,
                                                'deff': deff
                                            },
                                            CircuitParameters={
                                                'Rp': Rp,
                                                'QualityFactor': Q,
                                                'Temperature': 0.4,
                                            },
                                            SimulationParameters={
                                                'dt': 1e-12,
                                                'CoolingMode': CoolingMode,
                                                'TotalTime': TotalTime
                                            })
    CoolingTime.append(test.Run())

# save data
#np.save(CoolingMode + 'wrf_changing, wradical=2pi*{:.2f},waxial=2pi*{:.2f} in {:.2f} us.txt'.format(wrf/(2 * np.pi),wradical/(2 * np.pi), waxial/(2 * np.pi), TotalTime * 1e6), CoolingTime)
with open('Results, keep: Wrf.csv', 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     wr.writerow(wrfList)
     wr.writerow(CoolingTime)