import os.path
import Particle
import numpy as np

x0, y0, z0, vx0, vy0, vz0 = 10e-6, 10e-6, 5e-6, 0.1, 0.1, 0.1
Vec0 = x0, y0, z0, vx0, vy0, vz0
wrf = 2 * np.pi* 10e9 # 10 GHz
wradical = 2 * np.pi * 1e9 # 2 GHz  
waxial = 2 * np.pi * 300e6 # 300 MHz
deff = 200e-6 # 200 micron
m = 9.10938297e-31 # 9.10938297e-31 kg 
q = 1.6e-19 # 1.6e-19 C

Rp = 1e7
Q = 2000

CoolingMode = 'blue'

if __name__ == "__main__":
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
                                                'CoolingMode': 'secular',
                                                'TotalTime': 2e-6
                                            })
    test.Run()