#pragma once
#include <vector>
#include <cmath>
#include <math.h>
#include <iostream>
#include <random>
#include <cstdlib>
#include <stdlib.h>

using namespace std;

// gives the potential energy of the real trapping potential
double calcEpot(double t, double x, double y, double z, vector<double> fact, vector<double> frf){

	// l=1
	double EpotDC = z*fact[1] - x*fact[2] - y*fact[3];
	double EpotRF = z*frf[1] - x*frf[2] - y*frf[3];

	// l=2
	EpotDC += (-1./2.*pow(x,2)-1./2.*pow(y,2)+pow(z,2))*fact[4] - 3*x*z*fact[5] + (-3*y*z)*fact[6] + (3*pow(x,2)-3*pow(y,2))*fact[7] + (6*x*y)*fact[8];
	EpotRF += (-1./2.*pow(x,2)-1./2.*pow(y,2)+pow(z,2))*frf[4] - 3*x*z*frf[5] + (-3*y*z)*frf[6] + (3*pow(x,2)-3*pow(y,2))*frf[7] + (6*x*y)*frf[8];


	// l=3
	EpotDC += (pow(z,3)-3./2.*z*(pow(x,2)+pow(y,2)))*fact[9] + ((3./2.)*pow(x,3)+(3./2.)*x*pow(y,2)-6*x*pow(z,2))*fact[10] + ((3./2.)*pow(x,2)*y+(3./2.)*pow(y,3)-6*y*pow(z,2))*fact[11] + (15*z*(pow(x,2)-pow(y,2)))*fact[12] + (30*x*y*z)*fact[13] + (-15*pow(x,3)+45*x*pow(y,2))*fact[14] + (-45*pow(x,2)*y+15*pow(y,3))*fact[15];
	EpotRF += (pow(z,3)-3./2.*z*(pow(x,2)+pow(y,2)))*frf[9] + ((3./2.)*pow(x,3)+(3./2.)*x*pow(y,2)-6*x*pow(z,2))*frf[10] + ((3./2.)*pow(x,2)*y+(3./2.)*pow(y,3)-6*y*pow(z,2))*frf[11] + (15*z*(pow(x,2)-pow(y,2)))*frf[12] + (30*x*y*z)*frf[13] + (-15*pow(x,3)+45*x*pow(y,2))*frf[14] + (-45*pow(x,2)*y+15*pow(y,3))*frf[15];

	// l=4
	EpotDC += ((3./8.)*pow(x,4)+(3./4.)*pow(x,2)*pow(y,2)+(3./8.)*pow(y,4)+pow(z,4)-3*pow(z,2)*(pow(x,2)+pow(y,2)))*fact[16] + (-10*x*pow(z,3)+(15./2.)*z*(pow(x,3)+x*pow(y,2)))*fact[17] + (-10*y*pow(z,3)+(15./2.)*z*(pow(x,2)*y+pow(y,3)))*fact[18] + (-15./2.*pow(x,4)+(15./2.)*pow(y,4)+45*pow(z,2)*(pow(x,2)-pow(y,2)))*fact[19] + (-15*pow(x,3)*y-15*x*pow(y,3)+90*x*y*pow(z,2))*fact[20] + (-105*z*(pow(x,3)-3*x*pow(y,2)))*fact[21] + (-105*z*(3*pow(x,2)*y-pow(y,3)))*fact[22] + (105*pow(x,4)-630*pow(x,2)*pow(y,2)+105*pow(y,4))*fact[23] + (420*pow(x,3)*y-420*x*pow(y,3))*fact[24];
	EpotRF += ((3./8.)*pow(x,4)+(3./4.)*pow(x,2)*pow(y,2)+(3./8.)*pow(y,4)+pow(z,4)-3*pow(z,2)*(pow(x,2)+pow(y,2)))*frf[16] + (-10*x*pow(z,3)+(15./2.)*z*(pow(x,3)+x*pow(y,2)))*frf[17] + (-10*y*pow(z,3)+(15./2.)*z*(pow(x,2)*y+pow(y,3)))*frf[18] + (-15./2.*pow(x,4)+(15./2.)*pow(y,4)+45*pow(z,2)*(pow(x,2)-pow(y,2)))*frf[19] + (-15*pow(x,3)*y-15*x*pow(y,3)+90*x*y*pow(z,2))*frf[20] + (-105*z*(pow(x,3)-3*x*pow(y,2)))*frf[21] + (-105*z*(3*pow(x,2)*y-pow(y,3)))*frf[22] + (105*pow(x,4)-630*pow(x,2)*pow(y,2)+105*pow(y,4))*frf[23] + (420*pow(x,3)*y-420*x*pow(y,3))*frf[24];

	// l=5 
	EpotDC += (pow(z,5)-5*pow(z,3)*(pow(x,2)+pow(y,2))+(15./8.)*z*(pow(x,4)+2*pow(x,2)*pow(y,2)+pow(y,4)))*fact[25] + (-15./8.*pow(x,5)-15./4.*pow(x,3)*pow(y,2)-15./8.*x*pow(y,4)-15*x*pow(z,4)+(45./2.)*pow(z,2)*(pow(x,3)+x*pow(y,2)))*fact[26] + (-15./8.*pow(x,4)*y-15./4.*pow(x,2)*pow(y,3)-15./8.*pow(y,5)-15*y*pow(z,4)+(45./2.)*pow(z,2)*(pow(x,2)*y+pow(y,3)))*fact[27] + (105*pow(z,3)*(pow(x,2)-pow(y,2))-105./2.*z*(pow(x,4)-pow(y,4)))*fact[28]+(210*x*y*pow(z,3)-105*z*(pow(x,3)*y+x*pow(y,3)))*fact[29] + ((105./2.)*pow(x,5)- 05*pow(x,3)*pow(y,2)-315./2.*x*pow(y,4)-420*pow(z,2)*(pow(x,3)-3*x*pow(y,2)))*fact[30] + ((315./2.)*pow(x,4)*y+105*pow(x,2)*pow(y,3)-105./2.*pow(y,5)-420*pow(z,2)*(3*pow(x,2)*y-pow(y,3)))*fact[31] + (945*z*(pow(x,4)-6*pow(x,2)*pow(y,2)+pow(y,4)))*fact[32] + (3780*z*(pow(x,3)*y-x*pow(y,3)))*fact[33] + (-945*pow(x,5)+9450*pow(x,3)*pow(y,2)-4725*x*pow(y,4))*fact[34] + (-4725*pow(x,4)*y+9450*pow(x,2)*pow(y,3)-945*pow(y,5))*fact[35];
	EpotRF += (pow(z,5)-5*pow(z,3)*(pow(x,2)+pow(y,2))+(15./8.)*z*(pow(x,4)+2*pow(x,2)*pow(y,2)+pow(y,4)))*frf[25] + (-15./8.*pow(x,5)-15./4.*pow(x,3)*pow(y,2)-15./8.*x*pow(y,4)-15*x*pow(z,4)+(45./2.)*pow(z,2)*(pow(x,3)+x*pow(y,2)))*frf[26] + (-15./8.*pow(x,4)*y-15./4.*pow(x,2)*pow(y,3)-15./8.*pow(y,5)-15*y*pow(z,4)+(45./2.)*pow(z,2)*(pow(x,2)*y+pow(y,3)))*frf[27] + (105*pow(z,3)*(pow(x,2)-pow(y,2))-105./2.*z*(pow(x,4)-pow(y,4)))*frf[28]+(210*x*y*pow(z,3)-105*z*(pow(x,3)*y+x*pow(y,3)))*frf[29] + ((105./2.)*pow(x,5)- 05*pow(x,3)*pow(y,2)-315./2.*x*pow(y,4)-420*pow(z,2)*(pow(x,3)-3*x*pow(y,2)))*frf[30] + ((315./2.)*pow(x,4)*y+105*pow(x,2)*pow(y,3)-105./2.*pow(y,5)-420*pow(z,2)*(3*pow(x,2)*y-pow(y,3)))*frf[31] + (945*z*(pow(x,4)-6*pow(x,2)*pow(y,2)+pow(y,4)))*frf[32] + (3780*z*(pow(x,3)*y-x*pow(y,3)))*frf[33] + (-945*pow(x,5)+9450*pow(x,3)*pow(y,2)-4725*x*pow(y,4))*frf[34] + (-4725*pow(x,4)*y+9450*pow(x,2)*pow(y,3)-945*pow(y,5))*frf[35];

	// l=6
	EpotDC += (-5./16.*pow(x,6)-15./16.*pow(x,4)*pow(y,2)-15./16.*pow(x,2)*pow(y,4)-5./16.*pow(y,6)+pow(z,6)-15./2.*pow(z,4)*(pow(x,2)+pow(y,2))+(45./8.)*pow(z,2)*(pow(x,4)+2*pow(x,2)*pow(y,2)+pow(y,4)))*fact[36] + (-21*x*pow(z,5)+(105./2.)*pow(z,3)*(pow(x,3)+x*pow(y,2))-105./8.*z*(pow(x,5)+2*pow(x,3)*pow(y,2)+x*pow(y,4)))*fact[37] + (-21*y*pow(z,5)+(105./2.)*pow(z,3)*(pow(x,2)*y+pow(y,3))-105./8.*z*(pow(x,4)*y+2*pow(x,2)*pow(y,3)+pow(y,5)))*fact[38] + ((105./8.)*pow(x,6)+(105./8.)*pow(x,4)*pow(y,2)-105./8.*pow(x,2)*pow(y,4)-105./8.*pow(y,6)+210*pow(z,4)*(pow(x,2)-pow(y,2))-210*pow(z,2)*(pow(x,4)-pow(y,4)))*fact[39] + ((105./4.)*pow(x,5)*y+(105./2.)*pow(x,3)*pow(y,3)+(105./4.)*x*pow(y,5)+420*x*y*pow(z,4)-420*pow(z,2)*(pow(x,3)*y+x*pow(y,3)))*fact[40] + (-1260*pow(z,3)*(pow(x,3)-3*x*pow(y,2))+(945./2.)*z*(pow(x,5)-2*pow(x,3)*pow(y,2)-3*x*pow(y,4)))*fact[41] + (-1260*pow(z,3)*(3*pow(x,2)*y-pow(y,3))+(945./2.)*z*(3*pow(x,4)*y+2*pow(x,2)*pow(y,3)-pow(y,5)))*fact[42] + (-945./2.*pow(x,6)+(4725./2.)*pow(x,4)*pow(y,2)+(4725./2.)*pow(x,2)*pow(y,4)-945./2.*pow(y,6)+4725*pow(z,2)*(pow(x,4)-6*pow(x,2)*pow(y,2)+pow(y,4)))*fact[43] + (-1890*pow(x,5)*y+ 890*x*pow(y,5)+18900*pow(z,2)*(pow(x,3)*y-x*pow(y,3)))*fact[44] + (-10395*z*(pow(x,5)-10*pow(x,3)*pow(y,2)+5*x*pow(y,4)))*fact[45] + (-10395*z*(5*pow(x,4)*y-10*pow(x, 2)*pow(y, 3)+pow(y,5)))*fact[46] + (10395*pow(x,6)-155925*pow(x,4)*pow(y,2)+155925*pow(x,2)*pow(y,4)-10395*pow(y,6))*fact[47] + (62370*pow(x,5)*y-207900*pow(x,3)*pow(y,3)+62370*x*pow(y,5))*fact[48];
	EpotRF += (-5./16.*pow(x,6)-15./16.*pow(x,4)*pow(y,2)-15./16.*pow(x,2)*pow(y,4)-5./16.*pow(y,6)+pow(z,6)-15./2.*pow(z,4)*(pow(x,2)+pow(y,2))+(45./8.)*pow(z,2)*(pow(x,4)+2*pow(x,2)*pow(y,2)+pow(y,4)))*frf[36] + (-21*x*pow(z,5)+(105./2.)*pow(z,3)*(pow(x,3)+x*pow(y,2))-105./8.*z*(pow(x,5)+2*pow(x,3)*pow(y,2)+x*pow(y,4)))*frf[37] + (-21*y*pow(z,5)+(105./2.)*pow(z,3)*(pow(x,2)*y+pow(y,3))-105./8.*z*(pow(x,4)*y+2*pow(x,2)*pow(y,3)+pow(y,5)))*frf[38] + ((105./8.)*pow(x,6)+(105./8.)*pow(x,4)*pow(y,2)-105./8.*pow(x,2)*pow(y,4)-105./8.*pow(y,6)+210*pow(z,4)*(pow(x,2)-pow(y,2))-210*pow(z,2)*(pow(x,4)-pow(y,4)))*frf[39] + ((105./4.)*pow(x,5)*y+(105./2.)*pow(x,3)*pow(y,3)+(105./4.)*x*pow(y,5)+420*x*y*pow(z,4)-420*pow(z,2)*(pow(x,3)*y+x*pow(y,3)))*frf[40] + (-1260*pow(z,3)*(pow(x,3)-3*x*pow(y,2))+(945./2.)*z*(pow(x,5)-2*pow(x,3)*pow(y,2)-3*x*pow(y,4)))*frf[41] + (-1260*pow(z,3)*(3*pow(x,2)*y-pow(y,3))+(945./2.)*z*(3*pow(x,4)*y+2*pow(x,2)*pow(y,3)-pow(y,5)))*frf[42] + (-945./2.*pow(x,6)+(4725./2.)*pow(x,4)*pow(y,2)+(4725./2.)*pow(x,2)*pow(y,4)-945./2.*pow(y,6)+4725*pow(z,2)*(pow(x,4)-6*pow(x,2)*pow(y,2)+pow(y,4)))*frf[43] + (-1890*pow(x,5)*y+ 890*x*pow(y,5)+18900*pow(z,2)*(pow(x,3)*y-x*pow(y,3)))*frf[44] + (-10395*z*(pow(x,5)-10*pow(x,3)*pow(y,2)+5*x*pow(y,4)))*frf[45] + (-10395*z*(5*pow(x,4)*y-10*pow(x, 2)*pow(y, 3)+pow(y,5)))*frf[46] + (10395*pow(x,6)-155925*pow(x,4)*pow(y,2)+155925*pow(x,2)*pow(y,4)-10395*pow(y,6))*frf[47] + (62370*pow(x,5)*y-207900*pow(x,3)*pow(y,3)+62370*x*pow(y,5))*frf[48];

	// add everything up to potential energy
	double Epot_full = EpotDC + cos(omegaRF*t)*EpotRF;


	return Epot_full;
}





void energies(double comVelZ, vector<electron> &el, double t, double dX01, double dY01, double dZ01, double dTot, vector<double> fact, vector<double> frf){
	
	// starting with potential energy
	// *************** energy from harmonic trap potential****************************************************************
	
	el[0].Epot = VacQe_ov_r0r0*cos(omegaRF*t)*(el[0].posX*el[0].posX - el[0].posY*el[0].posY) + kappaUdcQe_ov_2z0z0*(-2*el[0].posZ*el[0].posZ - 0.9*el[0].posX*el[0].posX - 1.1*el[0].posY*el[0].posY);
	el[1].Epot = VacQe_ov_r0r0*cos(omegaRF*t)*(el[1].posX*el[1].posX - el[1].posY*el[1].posY) + kappaUdcQe_ov_2z0z0*(-2*el[1].posZ*el[1].posZ - 0.9*el[1].posX*el[1].posX - 1.1*el[1].posY*el[1].posY);
	// *************** OR: energy from realistic trap potential **********************************************************
	/*
	el[0].Epot = calcEpot(t, el[0].posX, el[0].posY, el[0].posZ, fact, frf);
	el[1].Epot = calcEpot(t, el[1].posX, el[1].posY, el[1].posZ, fact, frf);
	*/
	  
	// add interaction energy
	el[0].Epot += 0.5*coulfct/dTot;
	el[1].Epot += 0.5*coulfct/dTot;

	// now kinetic energy
	el[0].Ekin = me_ov_2*(el[0].velX*el[0].velX + el[0].velY*el[0].velY + el[0].velZ*el[0].velZ);
	el[1].Ekin = me_ov_2*(el[1].velX*el[1].velX + el[1].velY*el[1].velY + el[1].velZ*el[1].velZ);

	el[0].EkinX = me_ov_2*el[0].velX*el[0].velX;
	el[0].EkinY = me_ov_2*el[0].velY*el[0].velY;
	el[0].EkinZ = me_ov_2*el[0].velZ*el[0].velZ;

	el[1].EkinX = me_ov_2*el[1].velX*el[1].velX;
	el[1].EkinY = me_ov_2*el[1].velY*el[1].velY;
	el[1].EkinZ = me_ov_2*el[1].velZ*el[1].velZ;

	// now total energy
	el[0].Etot = el[0].Epot + el[0].Ekin;
	el[1].Etot = el[1].Epot + el[1].Ekin;
}
