#pragma once
#include <ctime>
#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include <cstdlib>


// calculate mean temperature
// start averaging after the cooling steps....then the final temperature should be reached
 
// since the potential energy is difficult to separate into  x,y and z direction, we use kinetic energy in z
// and take into account that kinetic and potential energy are constantly transfered into one another.
// This means that averaged over time the energy is divided into kinetic and potential energy by same parts
// thus we can get the mean total energy by taking twice the mean kinetic energy

double calc_axial_temp(vector<double> ax_energy_kin){
	
	double mean_Etot = 0.;
        //double most_probable_tempZ = 0;
        double mean_temp;
      
	for (int i = 0; i < numSteps - coolSteps; i++){
                mean_Etot += 2.*ax_energy_kin[coolSteps + i];
        }
	
	mean_Etot = mean_Etot / (numSteps - coolSteps);
        mean_temp = M_PI*mean_Etot / kB;
        //most_probable_tempZ = sqrt(M_PI)/2*mean_tempZ;
      
	//cout << "mean temperature after cooling in z-direction: " << mean_tempZ << " / K" << endl;
        //cout << "most probable temperature after cooling in z-direction: " << most_probable_tempZ << " / K" << endl;
        
	return mean_temp;
}



// create histogramm to see if we follow a Maxwell boltzmann energy distribution
// NO NORMALISATION
vector<double> get_ener_prob(vector<double> ax_energy_kin){
	
	vector<double> prob(cutoff_temp*intervals_temp , 0.);
	for (int i = 0; i < numSteps - coolSteps; i++){
		if (2.*ax_energy_kin[coolSteps + i]/kB < cutoff_temp){
			prob[int(2*ax_energy_kin[coolSteps + i]/kB*intervals_temp)] += 1;
		}
	}
        
	return prob;
}


vector<double> get_ener_steps(){
        vector<double> ener_vector(cutoff_temp*intervals_temp, 0.);
        for (int i = 0; i < cutoff_temp*intervals_temp; i++){
        	ener_vector[i] = i*0.01;
                ener_vector[i] *= (0.5*kB);
        }
        return ener_vector;
}


