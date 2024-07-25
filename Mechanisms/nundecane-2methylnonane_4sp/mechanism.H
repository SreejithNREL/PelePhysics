#ifndef MECHANISM_H
#define MECHANISM_H

#include <AMReX_Gpu.H>
#include <AMReX_REAL.H>

/* Elements
0  O
1  H
2  C
3  N
*/

// Species
#define C11H24_ID 0
#define O2_ID 1
#define N2_ID 2
#define IC10H22_ID 3

#define NUM_GAS_ELEMENTS 4  // Elements in the homogeneous phase
#define NUM_GAS_SPECIES 4   // Species in the homogeneous phase
#define NUM_GAS_REACTIONS 0 // Reactions in the homogeneous phase

#define SITE_DENSITY 0.000000E+00 // mol/cm^2

#define NUM_SURFACE_ELEMENTS 0  // Additional elements in heterogeneous phase
#define NUM_SURFACE_SPECIES 0   // Species in the heterogeneous phase
#define NUM_SURFACE_REACTIONS 0 // Reactions in the heterogeneous phase

#define NUM_ELEMENTS (NUM_GAS_ELEMENTS + NUM_SURFACE_ELEMENTS)
#define NUM_SPECIES (NUM_GAS_SPECIES + NUM_SURFACE_SPECIES)
#define NUM_REACTIONS (NUM_GAS_REACTIONS + NUM_SURFACE_REACTIONS)

#define NUM_IONS 0

#define NUM_FIT 4

//  ALWAYS on CPU stuff -- can have different def depending on if we are CPU or
//  GPU based. Defined in mechanism.cpp
void atomicWeight(amrex::Real* awt);
//  MISC
void CKAWT(amrex::Real* awt);
void CKNCF(int* ncf);
void CKSYME_STR(amrex::Vector<std::string>& ename);
void CKSYMS_STR(amrex::Vector<std::string>& kname);
void GET_RMAP(int* _rmap);
void CKINU(const int i, int& nspec, int* ki, int* nu);
void CKKFKR(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real* x,
  amrex::Real* q_f,
  amrex::Real* q_r);
void progressRateFR(
  amrex::Real* q_f, amrex::Real* q_r, amrex::Real* sc, amrex::Real T);
//  SPARSE INFORMATION
void SPARSITY_INFO(int* nJdata, const int* consP, int NCELLS);
void SPARSITY_INFO_SYST(int* nJdata, const int* consP, int NCELLS);
void SPARSITY_INFO_SYST_SIMPLIFIED(int* nJdata, const int* consP);
void
SPARSITY_PREPROC_CSC(int* rowVals, int* colPtrs, const int* consP, int NCELLS);
void SPARSITY_PREPROC_CSR(
  int* colVals, int* rowPtrs, const int* consP, int NCELLS, int base);
void SPARSITY_PREPROC_SYST_CSR(
  int* colVals, int* rowPtrs, const int* consP, int NCELLS, int base);
void SPARSITY_PREPROC_SYST_SIMPLIFIED_CSC(
  int* rowVals, int* colPtrs, int* indx, const int* consP);
void SPARSITY_PREPROC_SYST_SIMPLIFIED_CSR(
  int* colVals, int* rowPtr, const int* consP, int base);

// A few mechanism parameters
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKINDX(int& mm, int& kk, int& ii, int& nfit)
{
  mm = 4;
  kk = 4;
  ii = 0;
  nfit = -1; // Why do you need this anyway ?
}

//  inverse molecular weights
#ifdef AMREX_USE_GPU
AMREX_GPU_CONSTANT const amrex::Real global_imw[4] = {
  0.0063974205600302, // C11H24
  0.0312519532470779, // O2
  0.0356964374955379, // N2
  0.0070280983371519, // IC10H22
};
#endif
const amrex::Real h_global_imw[4] = {
  0.0063974205600302, // C11H24
  0.0312519532470779, // O2
  0.0356964374955379, // N2
  0.0070280983371519, // IC10H22
};

//  molecular weights
#ifdef AMREX_USE_GPU
AMREX_GPU_CONSTANT const amrex::Real global_mw[4] = {
  156.313000, // C11H24
  31.998000,  // O2
  28.014000,  // N2
  142.286000, // IC10H22
};
#endif
const amrex::Real h_global_mw[4] = {
  156.313000, // C11H24
  31.998000,  // O2
  28.014000,  // N2
  142.286000, // IC10H22
};

//  inverse molecular weights
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
get_imw(amrex::Real* imw_new)
{
  imw_new[0] = 0.0063974205600302; // C11H24
  imw_new[1] = 0.0312519532470779; // O2
  imw_new[2] = 0.0356964374955379; // N2
  imw_new[3] = 0.0070280983371519; // IC10H22
}

//  inverse molecular weight
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
imw(const int n)
{
#if AMREX_DEVICE_COMPILE
  return global_imw[n];
#else
  return h_global_imw[n];
#endif
}
//  molecular weights
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
get_mw(amrex::Real* mw_new)
{
  mw_new[0] = 156.313000; // C11H24
  mw_new[1] = 31.998000;  // O2
  mw_new[2] = 28.014000;  // N2
  mw_new[3] = 142.286000; // IC10H22
}

//  molecular weight
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
mw(const int n)
{
#if AMREX_DEVICE_COMPILE
  return global_mw[n];
#else
  return h_global_mw[n];
#endif
}

//  Returns R, Rc, Patm
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKRP(amrex::Real& ru, amrex::Real& ruc, amrex::Real& pa)
{
  ru = 8.31446261815324e+07;
  ruc = 1.98721558317399615845;
  pa = 1.01325e+06;
}

// compute Cv/R at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
cv_R(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 1: O2
    species[1] = +2.78245636e+00 - 2.99673416e-03 * T + 9.84730201e-06 * T2 -
                 9.68129509e-09 * T3 + 3.24372837e-12 * T4;
    // species 2: N2
    species[2] = +2.29867700e+00 + 1.40824040e-03 * T - 3.96322200e-06 * T2 +
                 5.64151500e-09 * T3 - 2.44485400e-12 * T4;
  } else {
    // species 1: O2
    species[1] = +2.28253784e+00 + 1.48308754e-03 * T - 7.57966669e-07 * T2 +
                 2.09470555e-10 * T3 - 2.16717794e-14 * T4;
    // species 2: N2
    species[2] = +1.92664000e+00 + 1.48797680e-03 * T - 5.68476000e-07 * T2 +
                 1.00970380e-10 * T3 - 6.75335100e-15 * T4;
  }

  // species with midpoint at T=1392 kelvin
  if (T < 1392) {
    // species 3: IC10H22
    species[3] = -3.52903989e+00 + 1.24306954e-01 * T - 8.00528889e-05 * T2 +
                 2.64935667e-08 * T3 - 3.58959145e-12 * T4;
  } else {
    // species 3: IC10H22
    species[3] = +3.08896738e+01 + 4.81493426e-02 * T - 1.64455764e-05 * T2 +
                 2.55076863e-09 * T3 - 1.47889199e-13 * T4;
  }

  // species with midpoint at T=1396 kelvin
  if (T < 1396) {
    // species 0: C11H24
    species[0] = -3.73545158e+00 + 1.41200556e-01 * T - 9.94788663e-05 * T2 +
                 3.92009595e-08 * T3 - 6.97268819e-12 * T4;
  } else {
    // species 0: C11H24
    species[0] = +3.48459345e+01 + 5.19677716e-02 * T - 1.77529070e-05 * T2 +
                 2.75374802e-09 * T3 - 1.59662271e-13 * T4;
  }
}

// compute Cp/R at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
cp_R(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 1: O2
    species[1] = +3.78245636e+00 - 2.99673416e-03 * T + 9.84730201e-06 * T2 -
                 9.68129509e-09 * T3 + 3.24372837e-12 * T4;
    // species 2: N2
    species[2] = +3.29867700e+00 + 1.40824040e-03 * T - 3.96322200e-06 * T2 +
                 5.64151500e-09 * T3 - 2.44485400e-12 * T4;
  } else {
    // species 1: O2
    species[1] = +3.28253784e+00 + 1.48308754e-03 * T - 7.57966669e-07 * T2 +
                 2.09470555e-10 * T3 - 2.16717794e-14 * T4;
    // species 2: N2
    species[2] = +2.92664000e+00 + 1.48797680e-03 * T - 5.68476000e-07 * T2 +
                 1.00970380e-10 * T3 - 6.75335100e-15 * T4;
  }

  // species with midpoint at T=1392 kelvin
  if (T < 1392) {
    // species 3: IC10H22
    species[3] = -2.52903989e+00 + 1.24306954e-01 * T - 8.00528889e-05 * T2 +
                 2.64935667e-08 * T3 - 3.58959145e-12 * T4;
  } else {
    // species 3: IC10H22
    species[3] = +3.18896738e+01 + 4.81493426e-02 * T - 1.64455764e-05 * T2 +
                 2.55076863e-09 * T3 - 1.47889199e-13 * T4;
  }

  // species with midpoint at T=1396 kelvin
  if (T < 1396) {
    // species 0: C11H24
    species[0] = -2.73545158e+00 + 1.41200556e-01 * T - 9.94788663e-05 * T2 +
                 3.92009595e-08 * T3 - 6.97268819e-12 * T4;
  } else {
    // species 0: C11H24
    species[0] = +3.58459345e+01 + 5.19677716e-02 * T - 1.77529070e-05 * T2 +
                 2.75374802e-09 * T3 - 1.59662271e-13 * T4;
  }
}

// compute the g/(RT) at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
gibbs(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;
  const amrex::Real invT = 1.0 / T;
  const amrex::Real logT = log(T);

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 1: O2
    species[1] = -1.063943560000000e+03 * invT + 1.247806300000001e-01 -
                 3.782456360000000e+00 * logT + 1.498367080000000e-03 * T -
                 1.641217001666667e-06 * T2 + 8.067745908333334e-10 * T3 -
                 1.621864185000000e-13 * T4;
    // species 2: N2
    species[2] = -1.020899900000000e+03 * invT - 6.516950000000001e-01 -
                 3.298677000000000e+00 * logT - 7.041202000000000e-04 * T +
                 6.605369999999999e-07 * T2 - 4.701262500000001e-10 * T3 +
                 1.222427000000000e-13 * T4;
  } else {
    // species 1: O2
    species[1] = -1.088457720000000e+03 * invT - 2.170693450000000e+00 -
                 3.282537840000000e+00 * logT - 7.415437700000000e-04 * T +
                 1.263277781666667e-07 * T2 - 1.745587958333333e-11 * T3 +
                 1.083588970000000e-15 * T4;
    // species 2: N2
    species[2] = -9.227977000000000e+02 * invT - 3.053888000000000e+00 -
                 2.926640000000000e+00 * logT - 7.439884000000000e-04 * T +
                 9.474600000000001e-08 * T2 - 8.414198333333333e-12 * T3 +
                 3.376675500000000e-16 * T4;
  }

  // species with midpoint at T=1392 kelvin
  if (T < 1392) {
    // species 3: IC10H22
    species[3] = -3.535749980000000e+04 * invT - 4.819491099000000e+01 +
                 2.529039890000000e+00 * logT - 6.215347700000000e-02 * T +
                 1.334214815000000e-05 * T2 - 2.207797225000000e-09 * T3 +
                 1.794795725000000e-13 * T4;
  } else {
    // species 3: IC10H22
    species[3] = -4.779616250000000e+04 * invT + 1.726810638000000e+02 -
                 3.188967380000000e+01 * logT - 2.407467130000000e-02 * T +
                 2.740929400000000e-06 * T2 - 2.125640525000000e-10 * T3 +
                 7.394459949999999e-15 * T4;
  }

  // species with midpoint at T=1396 kelvin
  if (T < 1396) {
    // species 0: C11H24
    species[0] = -3.737813680000000e+04 * invT - 5.162014358000000e+01 +
                 2.735451580000000e+00 * logT - 7.060027800000000e-02 * T +
                 1.657981105000000e-05 * T2 - 3.266746625000000e-09 * T3 +
                 3.486344095000000e-13 * T4;
  } else {
    // species 0: C11H24
    species[0] = -5.102047480000000e+04 * invT + 1.948253035000000e+02 -
                 3.584593450000000e+01 * logT - 2.598388580000000e-02 * T +
                 2.958817833333333e-06 * T2 - 2.294790016666667e-10 * T3 +
                 7.983113550000001e-15 * T4;
  }
}

// compute the a/(RT) at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
helmholtz(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;
  const amrex::Real invT = 1.0 / T;
  const amrex::Real logT = log(T);

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 1: O2
    species[1] = -1.06394356e+03 * invT - 8.75219370e-01 -
                 3.78245636e+00 * logT + 1.49836708e-03 * T -
                 1.64121700e-06 * T2 + 8.06774591e-10 * T3 -
                 1.62186418e-13 * T4;
    // species 2: N2
    species[2] = -1.02089990e+03 * invT - 1.65169500e+00 -
                 3.29867700e+00 * logT - 7.04120200e-04 * T +
                 6.60537000e-07 * T2 - 4.70126250e-10 * T3 +
                 1.22242700e-13 * T4;
  } else {
    // species 1: O2
    species[1] = -1.08845772e+03 * invT - 3.17069345e+00 -
                 3.28253784e+00 * logT - 7.41543770e-04 * T +
                 1.26327778e-07 * T2 - 1.74558796e-11 * T3 +
                 1.08358897e-15 * T4;
    // species 2: N2
    species[2] = -9.22797700e+02 * invT - 4.05388800e+00 -
                 2.92664000e+00 * logT - 7.43988400e-04 * T +
                 9.47460000e-08 * T2 - 8.41419833e-12 * T3 +
                 3.37667550e-16 * T4;
  }

  // species with midpoint at T=1392 kelvin
  if (T < 1392) {
    // species 3: IC10H22
    species[3] = -3.53574998e+04 * invT - 4.91949110e+01 +
                 2.52903989e+00 * logT - 6.21534770e-02 * T +
                 1.33421482e-05 * T2 - 2.20779723e-09 * T3 +
                 1.79479572e-13 * T4;
  } else {
    // species 3: IC10H22
    species[3] = -4.77961625e+04 * invT + 1.71681064e+02 -
                 3.18896738e+01 * logT - 2.40746713e-02 * T +
                 2.74092940e-06 * T2 - 2.12564053e-10 * T3 +
                 7.39445995e-15 * T4;
  }

  // species with midpoint at T=1396 kelvin
  if (T < 1396) {
    // species 0: C11H24
    species[0] = -3.73781368e+04 * invT - 5.26201436e+01 +
                 2.73545158e+00 * logT - 7.06002780e-02 * T +
                 1.65798110e-05 * T2 - 3.26674662e-09 * T3 +
                 3.48634410e-13 * T4;
  } else {
    // species 0: C11H24
    species[0] = -5.10204748e+04 * invT + 1.93825303e+02 -
                 3.58459345e+01 * logT - 2.59838858e-02 * T +
                 2.95881783e-06 * T2 - 2.29479002e-10 * T3 +
                 7.98311355e-15 * T4;
  }
}

// compute the e/(RT) at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
speciesInternalEnergy(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;
  const amrex::Real invT = 1.0 / T;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 1: O2
    species[1] = +2.78245636e+00 - 1.49836708e-03 * T + 3.28243400e-06 * T2 -
                 2.42032377e-09 * T3 + 6.48745674e-13 * T4 -
                 1.06394356e+03 * invT;
    // species 2: N2
    species[2] = +2.29867700e+00 + 7.04120200e-04 * T - 1.32107400e-06 * T2 +
                 1.41037875e-09 * T3 - 4.88970800e-13 * T4 -
                 1.02089990e+03 * invT;
  } else {
    // species 1: O2
    species[1] = +2.28253784e+00 + 7.41543770e-04 * T - 2.52655556e-07 * T2 +
                 5.23676387e-11 * T3 - 4.33435588e-15 * T4 -
                 1.08845772e+03 * invT;
    // species 2: N2
    species[2] = +1.92664000e+00 + 7.43988400e-04 * T - 1.89492000e-07 * T2 +
                 2.52425950e-11 * T3 - 1.35067020e-15 * T4 -
                 9.22797700e+02 * invT;
  }

  // species with midpoint at T=1392 kelvin
  if (T < 1392) {
    // species 3: IC10H22
    species[3] = -3.52903989e+00 + 6.21534770e-02 * T - 2.66842963e-05 * T2 +
                 6.62339168e-09 * T3 - 7.17918290e-13 * T4 -
                 3.53574998e+04 * invT;
  } else {
    // species 3: IC10H22
    species[3] = +3.08896738e+01 + 2.40746713e-02 * T - 5.48185880e-06 * T2 +
                 6.37692158e-10 * T3 - 2.95778398e-14 * T4 -
                 4.77961625e+04 * invT;
  }

  // species with midpoint at T=1396 kelvin
  if (T < 1396) {
    // species 0: C11H24
    species[0] = -3.73545158e+00 + 7.06002780e-02 * T - 3.31596221e-05 * T2 +
                 9.80023988e-09 * T3 - 1.39453764e-12 * T4 -
                 3.73781368e+04 * invT;
  } else {
    // species 0: C11H24
    species[0] = +3.48459345e+01 + 2.59838858e-02 * T - 5.91763567e-06 * T2 +
                 6.88437005e-10 * T3 - 3.19324542e-14 * T4 -
                 5.10204748e+04 * invT;
  }
}

// compute the h/(RT) at the given temperature (Eq 20)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
speciesEnthalpy(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;
  const amrex::Real invT = 1.0 / T;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 1: O2
    species[1] = +3.78245636e+00 - 1.49836708e-03 * T + 3.28243400e-06 * T2 -
                 2.42032377e-09 * T3 + 6.48745674e-13 * T4 -
                 1.06394356e+03 * invT;
    // species 2: N2
    species[2] = +3.29867700e+00 + 7.04120200e-04 * T - 1.32107400e-06 * T2 +
                 1.41037875e-09 * T3 - 4.88970800e-13 * T4 -
                 1.02089990e+03 * invT;
  } else {
    // species 1: O2
    species[1] = +3.28253784e+00 + 7.41543770e-04 * T - 2.52655556e-07 * T2 +
                 5.23676387e-11 * T3 - 4.33435588e-15 * T4 -
                 1.08845772e+03 * invT;
    // species 2: N2
    species[2] = +2.92664000e+00 + 7.43988400e-04 * T - 1.89492000e-07 * T2 +
                 2.52425950e-11 * T3 - 1.35067020e-15 * T4 -
                 9.22797700e+02 * invT;
  }

  // species with midpoint at T=1392 kelvin
  if (T < 1392) {
    // species 3: IC10H22
    species[3] = -2.52903989e+00 + 6.21534770e-02 * T - 2.66842963e-05 * T2 +
                 6.62339168e-09 * T3 - 7.17918290e-13 * T4 -
                 3.53574998e+04 * invT;
  } else {
    // species 3: IC10H22
    species[3] = +3.18896738e+01 + 2.40746713e-02 * T - 5.48185880e-06 * T2 +
                 6.37692158e-10 * T3 - 2.95778398e-14 * T4 -
                 4.77961625e+04 * invT;
  }

  // species with midpoint at T=1396 kelvin
  if (T < 1396) {
    // species 0: C11H24
    species[0] = -2.73545158e+00 + 7.06002780e-02 * T - 3.31596221e-05 * T2 +
                 9.80023988e-09 * T3 - 1.39453764e-12 * T4 -
                 3.73781368e+04 * invT;
  } else {
    // species 0: C11H24
    species[0] = +3.58459345e+01 + 2.59838858e-02 * T - 5.91763567e-06 * T2 +
                 6.88437005e-10 * T3 - 3.19324542e-14 * T4 -
                 5.10204748e+04 * invT;
  }
}

// compute the S/R at the given temperature (Eq 21)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
speciesEntropy(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;
  const amrex::Real logT = log(T);

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 1: O2
    species[1] = +3.78245636e+00 * logT - 2.99673416e-03 * T +
                 4.92365101e-06 * T2 - 3.22709836e-09 * T3 +
                 8.10932092e-13 * T4 + 3.65767573e+00;
    // species 2: N2
    species[2] = +3.29867700e+00 * logT + 1.40824040e-03 * T -
                 1.98161100e-06 * T2 + 1.88050500e-09 * T3 -
                 6.11213500e-13 * T4 + 3.95037200e+00;
  } else {
    // species 1: O2
    species[1] = +3.28253784e+00 * logT + 1.48308754e-03 * T -
                 3.78983334e-07 * T2 + 6.98235183e-11 * T3 -
                 5.41794485e-15 * T4 + 5.45323129e+00;
    // species 2: N2
    species[2] = +2.92664000e+00 * logT + 1.48797680e-03 * T -
                 2.84238000e-07 * T2 + 3.36567933e-11 * T3 -
                 1.68833775e-15 * T4 + 5.98052800e+00;
  }

  // species with midpoint at T=1392 kelvin
  if (T < 1392) {
    // species 3: IC10H22
    species[3] = -2.52903989e+00 * logT + 1.24306954e-01 * T -
                 4.00264445e-05 * T2 + 8.83118890e-09 * T3 -
                 8.97397862e-13 * T4 + 4.56658711e+01;
  } else {
    // species 3: IC10H22
    species[3] = +3.18896738e+01 * logT + 4.81493426e-02 * T -
                 8.22278820e-06 * T2 + 8.50256210e-10 * T3 -
                 3.69722997e-14 * T4 - 1.40791390e+02;
  }

  // species with midpoint at T=1396 kelvin
  if (T < 1396) {
    // species 0: C11H24
    species[0] = -2.73545158e+00 * logT + 1.41200556e-01 * T -
                 4.97394331e-05 * T2 + 1.30669865e-08 * T3 -
                 1.74317205e-12 * T4 + 4.88846920e+01;
  } else {
    // species 0: C11H24
    species[0] = +3.58459345e+01 * logT + 5.19677716e-02 * T -
                 8.87645350e-06 * T2 + 9.17916007e-10 * T3 -
                 3.99155677e-14 * T4 - 1.58979369e+02;
  }
}

// compute d(Cp/R)/dT and d(Cv/R)/dT at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
dcvpRdT(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 1: O2
    species[1] = -2.99673416e-03 + 1.96946040e-05 * T - 2.90438853e-08 * T2 +
                 1.29749135e-11 * T3;
    // species 2: N2
    species[2] = +1.40824040e-03 - 7.92644400e-06 * T + 1.69245450e-08 * T2 -
                 9.77941600e-12 * T3;
  } else {
    // species 1: O2
    species[1] = +1.48308754e-03 - 1.51593334e-06 * T + 6.28411665e-10 * T2 -
                 8.66871176e-14 * T3;
    // species 2: N2
    species[2] = +1.48797680e-03 - 1.13695200e-06 * T + 3.02911140e-10 * T2 -
                 2.70134040e-14 * T3;
  }

  // species with midpoint at T=1392 kelvin
  if (T < 1392) {
    // species 3: IC10H22
    species[3] = +1.24306954e-01 - 1.60105778e-04 * T + 7.94807001e-08 * T2 -
                 1.43583658e-11 * T3;
  } else {
    // species 3: IC10H22
    species[3] = +4.81493426e-02 - 3.28911528e-05 * T + 7.65230589e-09 * T2 -
                 5.91556796e-13 * T3;
  }

  // species with midpoint at T=1396 kelvin
  if (T < 1396) {
    // species 0: C11H24
    species[0] = +1.41200556e-01 - 1.98957733e-04 * T + 1.17602879e-07 * T2 -
                 2.78907528e-11 * T3;
  } else {
    // species 0: C11H24
    species[0] = +5.19677716e-02 - 3.55058140e-05 * T + 8.26124406e-09 * T2 -
                 6.38649084e-13 * T3;
  }
}

// Returns the mean specific heat at CP (Eq. 33)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPBL(const amrex::Real T, const amrex::Real x[], amrex::Real& cpbl)
{
  amrex::Real result = 0;
  amrex::Real cpor[4]; //  temporary storage
  cp_R(cpor, T);

  // perform dot product
  for (int id = 0; id < 4; ++id) {
    result += x[id] * cpor[id];
  }

  cpbl = result * 8.31446261815324e+07;
}

// Returns the mean specific heat at CP (Eq. 34)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPBS(const amrex::Real T, const amrex::Real y[], amrex::Real& cpbs)
{
  amrex::Real result = 0.0;

  // compute Cp/R at the given temperature
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 1: O2
    result += y[1] *
              (+3.78245636e+00 - 2.99673416e-03 * T + 9.84730201e-06 * T2 -
               9.68129509e-09 * T3 + 3.24372837e-12 * T4) *
              0.0312519532470779;
    // species 2: N2
    result += y[2] *
              (+3.29867700e+00 + 1.40824040e-03 * T - 3.96322200e-06 * T2 +
               5.64151500e-09 * T3 - 2.44485400e-12 * T4) *
              0.0356964374955379;
  } else {
    // species 1: O2
    result += y[1] *
              (+3.28253784e+00 + 1.48308754e-03 * T - 7.57966669e-07 * T2 +
               2.09470555e-10 * T3 - 2.16717794e-14 * T4) *
              0.0312519532470779;
    // species 2: N2
    result += y[2] *
              (+2.92664000e+00 + 1.48797680e-03 * T - 5.68476000e-07 * T2 +
               1.00970380e-10 * T3 - 6.75335100e-15 * T4) *
              0.0356964374955379;
  }

  // species with midpoint at T=1392 kelvin
  if (T < 1392) {
    // species 3: IC10H22
    result += y[3] *
              (-2.52903989e+00 + 1.24306954e-01 * T - 8.00528889e-05 * T2 +
               2.64935667e-08 * T3 - 3.58959145e-12 * T4) *
              0.0070280983371519;
  } else {
    // species 3: IC10H22
    result += y[3] *
              (+3.18896738e+01 + 4.81493426e-02 * T - 1.64455764e-05 * T2 +
               2.55076863e-09 * T3 - 1.47889199e-13 * T4) *
              0.0070280983371519;
  }

  // species with midpoint at T=1396 kelvin
  if (T < 1396) {
    // species 0: C11H24
    result += y[0] *
              (-2.73545158e+00 + 1.41200556e-01 * T - 9.94788663e-05 * T2 +
               3.92009595e-08 * T3 - 6.97268819e-12 * T4) *
              0.0063974205600302;
  } else {
    // species 0: C11H24
    result += y[0] *
              (+3.58459345e+01 + 5.19677716e-02 * T - 1.77529070e-05 * T2 +
               2.75374802e-09 * T3 - 1.59662271e-13 * T4) *
              0.0063974205600302;
  }

  cpbs = result * 8.31446261815324e+07;
}

// Returns the mean specific heat at CV (Eq. 35)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCVBL(const amrex::Real T, const amrex::Real x[], amrex::Real& cvbl)
{
  amrex::Real result = 0;
  amrex::Real cvor[4]; //  temporary storage
  cv_R(cvor, T);

  // perform dot product
  for (int id = 0; id < 4; ++id) {
    result += x[id] * cvor[id];
  }

  cvbl = result * 8.31446261815324e+07;
}

// Returns the mean specific heat at CV (Eq. 36)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCVBS(const amrex::Real T, const amrex::Real y[], amrex::Real& cvbs)
{
  amrex::Real result = 0.0;
  // compute Cv/R at the given temperature
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 1: O2
    result += y[1] *
              (+2.78245636e+00 - 2.99673416e-03 * T + 9.84730201e-06 * T2 -
               9.68129509e-09 * T3 + 3.24372837e-12 * T4) *
              0.0312519532470779;
    // species 2: N2
    result += y[2] *
              (+2.29867700e+00 + 1.40824040e-03 * T - 3.96322200e-06 * T2 +
               5.64151500e-09 * T3 - 2.44485400e-12 * T4) *
              0.0356964374955379;
  } else {
    // species 1: O2
    result += y[1] *
              (+2.28253784e+00 + 1.48308754e-03 * T - 7.57966669e-07 * T2 +
               2.09470555e-10 * T3 - 2.16717794e-14 * T4) *
              0.0312519532470779;
    // species 2: N2
    result += y[2] *
              (+1.92664000e+00 + 1.48797680e-03 * T - 5.68476000e-07 * T2 +
               1.00970380e-10 * T3 - 6.75335100e-15 * T4) *
              0.0356964374955379;
  }

  // species with midpoint at T=1392 kelvin
  if (T < 1392) {
    // species 3: IC10H22
    result += y[3] *
              (-3.52903989e+00 + 1.24306954e-01 * T - 8.00528889e-05 * T2 +
               2.64935667e-08 * T3 - 3.58959145e-12 * T4) *
              0.0070280983371519;
  } else {
    // species 3: IC10H22
    result += y[3] *
              (+3.08896738e+01 + 4.81493426e-02 * T - 1.64455764e-05 * T2 +
               2.55076863e-09 * T3 - 1.47889199e-13 * T4) *
              0.0070280983371519;
  }

  // species with midpoint at T=1396 kelvin
  if (T < 1396) {
    // species 0: C11H24
    result += y[0] *
              (-3.73545158e+00 + 1.41200556e-01 * T - 9.94788663e-05 * T2 +
               3.92009595e-08 * T3 - 6.97268819e-12 * T4) *
              0.0063974205600302;
  } else {
    // species 0: C11H24
    result += y[0] *
              (+3.48459345e+01 + 5.19677716e-02 * T - 1.77529070e-05 * T2 +
               2.75374802e-09 * T3 - 1.59662271e-13 * T4) *
              0.0063974205600302;
  }

  cvbs = result * 8.31446261815324e+07;
}

// Returns the mean enthalpy of the mixture in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHBML(const amrex::Real T, const amrex::Real x[], amrex::Real& hbml)
{
  amrex::Real result = 0;
  amrex::Real hml[4];                        //  temporary storage
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T
  speciesEnthalpy(hml, T);

  // perform dot product
  for (int id = 0; id < 4; ++id) {
    result += x[id] * hml[id];
  }

  hbml = result * RT;
}

// Returns mean enthalpy of mixture in mass units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHBMS(const amrex::Real T, const amrex::Real y[], amrex::Real& hbms)
{
  amrex::Real result = 0.0;
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;
  const amrex::Real invT = 1.0 / T;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 1: O2
    result +=
      y[1] *
      (+3.78245636e+00 - 1.49836708e-03 * T + 3.28243400e-06 * T2 -
       2.42032377e-09 * T3 + 6.48745674e-13 * T4 - 1.06394356e+03 * invT) *
      0.0312519532470779;
    // species 2: N2
    result +=
      y[2] *
      (+3.29867700e+00 + 7.04120200e-04 * T - 1.32107400e-06 * T2 +
       1.41037875e-09 * T3 - 4.88970800e-13 * T4 - 1.02089990e+03 * invT) *
      0.0356964374955379;
  } else {
    // species 1: O2
    result +=
      y[1] *
      (+3.28253784e+00 + 7.41543770e-04 * T - 2.52655556e-07 * T2 +
       5.23676387e-11 * T3 - 4.33435588e-15 * T4 - 1.08845772e+03 * invT) *
      0.0312519532470779;
    // species 2: N2
    result +=
      y[2] *
      (+2.92664000e+00 + 7.43988400e-04 * T - 1.89492000e-07 * T2 +
       2.52425950e-11 * T3 - 1.35067020e-15 * T4 - 9.22797700e+02 * invT) *
      0.0356964374955379;
  }

  // species with midpoint at T=1392 kelvin
  if (T < 1392) {
    // species 3: IC10H22
    result +=
      y[3] *
      (-2.52903989e+00 + 6.21534770e-02 * T - 2.66842963e-05 * T2 +
       6.62339168e-09 * T3 - 7.17918290e-13 * T4 - 3.53574998e+04 * invT) *
      0.0070280983371519;
  } else {
    // species 3: IC10H22
    result +=
      y[3] *
      (+3.18896738e+01 + 2.40746713e-02 * T - 5.48185880e-06 * T2 +
       6.37692158e-10 * T3 - 2.95778398e-14 * T4 - 4.77961625e+04 * invT) *
      0.0070280983371519;
  }

  // species with midpoint at T=1396 kelvin
  if (T < 1396) {
    // species 0: C11H24
    result +=
      y[0] *
      (-2.73545158e+00 + 7.06002780e-02 * T - 3.31596221e-05 * T2 +
       9.80023988e-09 * T3 - 1.39453764e-12 * T4 - 3.73781368e+04 * invT) *
      0.0063974205600302;
  } else {
    // species 0: C11H24
    result +=
      y[0] *
      (+3.58459345e+01 + 2.59838858e-02 * T - 5.91763567e-06 * T2 +
       6.88437005e-10 * T3 - 3.19324542e-14 * T4 - 5.10204748e+04 * invT) *
      0.0063974205600302;
  }

  const amrex::Real RT = 8.31446261815324e+07 * T; // R*T

  hbms = result * RT;
}

// get mean internal energy in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKUBML(const amrex::Real T, const amrex::Real x[], amrex::Real& ubml)
{
  amrex::Real result = 0;
  amrex::Real uml[4];                        //  temporary energy array
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T
  speciesInternalEnergy(uml, T);

  // perform dot product
  for (int id = 0; id < 4; ++id) {
    result += x[id] * uml[id];
  }

  ubml = result * RT;
}

// get mean internal energy in mass units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKUBMS(const amrex::Real T, const amrex::Real y[], amrex::Real& ubms)
{
  amrex::Real result = 0.0;
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;
  const amrex::Real invT = 1.0 / T;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 1: O2
    result +=
      y[1] *
      (+2.78245636e+00 - 1.49836708e-03 * T + 3.28243400e-06 * T2 -
       2.42032377e-09 * T3 + 6.48745674e-13 * T4 - 1.06394356e+03 * invT) *
      0.0312519532470779;
    // species 2: N2
    result +=
      y[2] *
      (+2.29867700e+00 + 7.04120200e-04 * T - 1.32107400e-06 * T2 +
       1.41037875e-09 * T3 - 4.88970800e-13 * T4 - 1.02089990e+03 * invT) *
      0.0356964374955379;
  } else {
    // species 1: O2
    result +=
      y[1] *
      (+2.28253784e+00 + 7.41543770e-04 * T - 2.52655556e-07 * T2 +
       5.23676387e-11 * T3 - 4.33435588e-15 * T4 - 1.08845772e+03 * invT) *
      0.0312519532470779;
    // species 2: N2
    result +=
      y[2] *
      (+1.92664000e+00 + 7.43988400e-04 * T - 1.89492000e-07 * T2 +
       2.52425950e-11 * T3 - 1.35067020e-15 * T4 - 9.22797700e+02 * invT) *
      0.0356964374955379;
  }

  // species with midpoint at T=1392 kelvin
  if (T < 1392) {
    // species 3: IC10H22
    result +=
      y[3] *
      (-3.52903989e+00 + 6.21534770e-02 * T - 2.66842963e-05 * T2 +
       6.62339168e-09 * T3 - 7.17918290e-13 * T4 - 3.53574998e+04 * invT) *
      0.0070280983371519;
  } else {
    // species 3: IC10H22
    result +=
      y[3] *
      (+3.08896738e+01 + 2.40746713e-02 * T - 5.48185880e-06 * T2 +
       6.37692158e-10 * T3 - 2.95778398e-14 * T4 - 4.77961625e+04 * invT) *
      0.0070280983371519;
  }

  // species with midpoint at T=1396 kelvin
  if (T < 1396) {
    // species 0: C11H24
    result +=
      y[0] *
      (-3.73545158e+00 + 7.06002780e-02 * T - 3.31596221e-05 * T2 +
       9.80023988e-09 * T3 - 1.39453764e-12 * T4 - 3.73781368e+04 * invT) *
      0.0063974205600302;
  } else {
    // species 0: C11H24
    result +=
      y[0] *
      (+3.48459345e+01 + 2.59838858e-02 * T - 5.91763567e-06 * T2 +
       6.88437005e-10 * T3 - 3.19324542e-14 * T4 - 5.10204748e+04 * invT) *
      0.0063974205600302;
  }

  const amrex::Real RT = 8.31446261815324e+07 * T; // R*T

  ubms = result * RT;
}

// get mixture entropy in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSBML(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real& sbml)
{
  amrex::Real result = 0;
  // Log of normalized pressure in cgs units dynes/cm^2 by Patm
  amrex::Real logPratio = log(P / 1013250.0);
  amrex::Real sor[4]; //  temporary storage
  speciesEntropy(sor, T);

  // Compute Eq 42
  for (int id = 0; id < 4; ++id) {
    result += x[id] * (sor[id] - log((x[id] + 1e-100)) - logPratio);
  }

  sbml = result * 8.31446261815324e+07;
}

// get mixture entropy in mass units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSBMS(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real y[],
  amrex::Real& sbms)
{
  amrex::Real result = 0;
  // Log of normalized pressure in cgs units dynes/cm^2 by Patm
  amrex::Real logPratio = log(P / 1013250.0);
  amrex::Real sor[4];  //  temporary storage
  amrex::Real x[4];    //  need a ytx conversion
  amrex::Real YOW = 0; // See Eq 4, 6 in CK Manual

  // Compute inverse of mean molecular wt first
  for (int i = 0; i < 4; i++) {
    YOW += y[i] * imw(i);
  }
  // Now compute y to x conversion
  x[0] = y[0] / (156.313000 * YOW);
  x[1] = y[1] / (31.998000 * YOW);
  x[2] = y[2] / (28.014000 * YOW);
  x[3] = y[3] / (142.286000 * YOW);
  speciesEntropy(sor, T);
  // Perform computation in Eq 42 and 43
  for (int i = 0; i < 4; i++) {
    result += x[i] * (sor[i] - log((x[i] + 1e-100)) - logPratio);
  }
  // Scale by R/W
  sbms = result * 8.31446261815324e+07 * YOW;
}

//  get temperature given internal energy in mass units and mass fracs
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
GET_T_GIVEN_EY(
  const amrex::Real e, const amrex::Real y[], amrex::Real& t, int& ierr)
{
#ifdef CONVERGENCE
  const int maxiter = 5000;
  const amrex::Real tol = 1.e-12;
#else
  const int maxiter = 200;
  const amrex::Real tol = 1.e-6;
#endif
  amrex::Real tmin = 90;   // max lower bound for thermo def
  amrex::Real tmax = 4000; // min upper bound for thermo def
  amrex::Real e1, emin, emax, cv, t1, dt;
  CKUBMS(tmin, y, emin);
  CKUBMS(tmax, y, emax);
  if (e < emin) {
    // Linear Extrapolation below tmin
    CKCVBS(tmin, y, cv);
    t = tmin - (emin - e) / cv;
    ierr = 1;
    return;
  }
  if (e > emax) {
    // Linear Extrapolation above tmax
    CKCVBS(tmax, y, cv);
    t = tmax - (emax - e) / cv;
    ierr = 1;
    return;
  }
  t1 = t;
  if (t1 < tmin || t1 > tmax) {
    t1 = tmin + (tmax - tmin) / (emax - emin) * (e - emin);
  }
  for (int i = 0; i < maxiter; ++i) {
    CKUBMS(t1, y, e1);
    CKCVBS(t1, y, cv);
    dt = (e - e1) / cv;
    if (dt > 100.) {
      dt = 100.;
    } else if (dt < -100.) {
      dt = -100.;
    } else if (fabs(dt) < tol) {
      break;
    }
    t1 += dt;
  }
  t = t1;
  ierr = 0;
}

//  get temperature given enthalpy in mass units and mass fracs
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
GET_T_GIVEN_HY(
  const amrex::Real h, const amrex::Real y[], amrex::Real& t, int& ierr)
{
#ifdef CONVERGENCE
  const int maxiter = 5000;
  const amrex::Real tol = 1.e-12;
#else
  const int maxiter = 200;
  const amrex::Real tol = 1.e-6;
#endif
  amrex::Real tmin = 90;   // max lower bound for thermo def
  amrex::Real tmax = 4000; // min upper bound for thermo def
  amrex::Real h1, hmin, hmax, cp, t1, dt;
  CKHBMS(tmin, y, hmin);
  CKHBMS(tmax, y, hmax);
  if (h < hmin) {
    // Linear Extrapolation below tmin
    CKCPBS(tmin, y, cp);
    t = tmin - (hmin - h) / cp;
    ierr = 1;
    return;
  }
  if (h > hmax) {
    // Linear Extrapolation above tmax
    CKCPBS(tmax, y, cp);
    t = tmax - (hmax - h) / cp;
    ierr = 1;
    return;
  }
  t1 = t;
  if (t1 < tmin || t1 > tmax) {
    t1 = tmin + (tmax - tmin) / (hmax - hmin) * (h - hmin);
  }
  for (int i = 0; i < maxiter; ++i) {
    CKHBMS(t1, y, h1);
    CKCPBS(t1, y, cp);
    dt = (h - h1) / cp;
    if (dt > 100.) {
      dt = 100.;
    } else if (dt < -100.) {
      dt = -100.;
    } else if (fabs(dt) < tol) {
      break;
    }
    t1 += dt;
  }
  t = t1;
  ierr = 0;
}

// Compute P = rhoRT/W(x)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKPX(
  const amrex::Real rho,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real& P)
{
  amrex::Real XW = 0;                      //  To hold mean molecular wt
  XW += x[0] * 156.313000;                 // C11H24
  XW += x[1] * 31.998000;                  // O2
  XW += x[2] * 28.014000;                  // N2
  XW += x[3] * 142.286000;                 // IC10H22
  P = rho * 8.31446261815324e+07 * T / XW; // P = rho*R*T/W
}

// Compute P = rhoRT/W(y)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKPY(
  const amrex::Real rho,
  const amrex::Real T,
  const amrex::Real y[],
  amrex::Real& P)
{
  amrex::Real YOW = 0; //  for computing mean MW

  for (int i = 0; i < 4; i++) {
    YOW += y[i] * imw(i);
  }
  P = rho * 8.31446261815324e+07 * T * YOW; // P = rho*R*T/W
}

// Compute P = rhoRT/W(c)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKPC(
  const amrex::Real rho,
  const amrex::Real T,
  const amrex::Real c[],
  amrex::Real& P)
{
  // See Eq 5 in CK Manual
  amrex::Real W = 0;
  amrex::Real sumC = 0;
  W += c[0] * 156.313000; // C11H24
  W += c[1] * 31.998000;  // O2
  W += c[2] * 28.014000;  // N2
  W += c[3] * 142.286000; // IC10H22

  for (int id = 0; id < 4; ++id) {
    sumC += c[id];
  }
  P = rho * 8.31446261815324e+07 * T * sumC / W; // P = rho*R*T/W
}

// Compute rho = PW(x)/RT
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKRHOX(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real& rho)
{
  amrex::Real XW = 0;                        //  To hold mean molecular wt
  XW += x[0] * 156.313000;                   // C11H24
  XW += x[1] * 31.998000;                    // O2
  XW += x[2] * 28.014000;                    // N2
  XW += x[3] * 142.286000;                   // IC10H22
  rho = P * XW / (8.31446261815324e+07 * T); // rho = P*W/(R*T)
}

// Compute rho = P*W(y)/RT
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKRHOY(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real y[],
  amrex::Real& rho)
{
  amrex::Real YOW = 0;

  for (int i = 0; i < 4; i++) {
    YOW += y[i] * imw(i);
  }

  rho = P / (8.31446261815324e+07 * T * YOW); // rho = P*W/(R*T)
}

// Compute rho = P*W(c)/(R*T)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKRHOC(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real c[],
  amrex::Real& rho)
{
  // See Eq 5 in CK Manual
  amrex::Real W = 0;
  amrex::Real sumC = 0;
  W += c[0] * 156.313000; // C11H24
  W += c[1] * 31.998000;  // O2
  W += c[2] * 28.014000;  // N2
  W += c[3] * 142.286000; // IC10H22

  for (int id = 0; id < 4; ++id) {
    sumC += c[id];
  }
  rho = P * W / (sumC * T * 8.31446261815324e+07); // rho = PW/(R*T)
}

// get molecular weight for all species
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKWT(amrex::Real wt[])
{
  get_mw(wt);
}

// given y[species]: mass fractions
// s mean molecular weight (gm/mole)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKMMWY(const amrex::Real y[], amrex::Real& wtm)
{
  amrex::Real YOW = 0;

  for (int i = 0; i < 4; i++) {
    YOW += y[i] * imw(i);
  }

  wtm = 1.0 / YOW;
}

// given x[species]: mole fractions
// returns mean molecular weight (gm/mole)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKMMWX(const amrex::Real x[], amrex::Real& wtm)
{
  amrex::Real XW = 0;      //  see Eq 4 in CK Manual
  XW += x[0] * 156.313000; // C11H24
  XW += x[1] * 31.998000;  // O2
  XW += x[2] * 28.014000;  // N2
  XW += x[3] * 142.286000; // IC10H22
  wtm = XW;
}

// given c[species]: molar concentration
// returns mean molecular weight (gm/mole)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKMMWC(const amrex::Real c[], amrex::Real& wtm)
{
  // See Eq 5 in CK Manual
  amrex::Real W = 0;
  amrex::Real sumC = 0;
  W += c[0] * 156.313000; // C11H24
  W += c[1] * 31.998000;  // O2
  W += c[2] * 28.014000;  // N2
  W += c[3] * 142.286000; // IC10H22

  for (int id = 0; id < 4; ++id) {
    sumC += c[id];
  }
  //  CK provides no guard against division by zero
  wtm = W / sumC;
}

// get Cp/R as a function of T
// for all species (Eq 19)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPOR(const amrex::Real T, amrex::Real cpor[])
{
  cp_R(cpor, T);
}

// get H/RT as a function of T
// for all species (Eq 20)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHORT(const amrex::Real T, amrex::Real hort[])
{
  speciesEnthalpy(hort, T);
}

// get S/R as a function of T
// for all species (Eq 21)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSOR(const amrex::Real T, amrex::Real sor[])
{
  speciesEntropy(sor, T);
}

// convert y[species] (mass fracs) to x[species] (mole fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKYTX(const amrex::Real y[], amrex::Real x[])
{
  amrex::Real YOW = 0;

  for (int i = 0; i < 4; i++) {
    YOW += y[i] * imw(i);
  }

  amrex::Real YOWINV = 1.0 / YOW;

  for (int i = 0; i < 4; i++) {
    x[i] = y[i] * imw(i) * YOWINV;
  }
}

// convert y[species] (mass fracs) to c[species] (molar conc)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKYTCP(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real y[],
  amrex::Real c[])
{
  amrex::Real YOW = 0;
  amrex::Real PWORT;

  // Compute inverse of mean molecular wt first
  for (int i = 0; i < 4; i++) {
    c[i] = y[i] * imw(i);
  }
  for (int i = 0; i < 4; i++) {
    YOW += c[i];
  }

  // PW/RT (see Eq. 7)
  PWORT = P / (YOW * 8.31446261815324e+07 * T);
  // Now compute conversion

  for (int i = 0; i < 4; i++) {
    c[i] = PWORT * y[i] * imw(i);
  }
}

// convert y[species] (mass fracs) to c[species] (molar conc)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKYTCR(
  const amrex::Real rho,
  amrex::Real /*T*/,
  const amrex::Real y[],
  amrex::Real c[])
{

  for (int i = 0; i < 4; i++) {
    c[i] = rho * y[i] * imw(i);
  }
}

// convert x[species] (mole fracs) to y[species] (mass fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKXTY(const amrex::Real x[], amrex::Real y[])
{
  amrex::Real XW = 0; // See Eq 4, 9 in CK Manual
  // Compute mean molecular wt first
  XW += x[0] * 156.313000; // C11H24
  XW += x[1] * 31.998000;  // O2
  XW += x[2] * 28.014000;  // N2
  XW += x[3] * 142.286000; // IC10H22
  // Now compute conversion
  amrex::Real XWinv = 1.0 / XW;
  y[0] = x[0] * 156.313000 * XWinv;
  y[1] = x[1] * 31.998000 * XWinv;
  y[2] = x[2] * 28.014000 * XWinv;
  y[3] = x[3] * 142.286000 * XWinv;
}

// convert x[species] (mole fracs) to c[species] (molar conc)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKXTCP(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real c[])
{
  amrex::Real PORT = P / (8.31446261815324e+07 * T); // P/RT

  // Compute conversion, see Eq 10
  for (int id = 0; id < 4; ++id) {
    c[id] = x[id] * PORT;
  }
}

// convert x[species] (mole fracs) to c[species] (molar conc)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKXTCR(
  const amrex::Real rho,
  const amrex::Real /*T*/,
  const amrex::Real x[],
  amrex::Real c[])
{
  amrex::Real XW = 0; // See Eq 4, 11 in CK Manual
  amrex::Real ROW;
  // Compute mean molecular wt first
  XW += x[0] * 156.313000; // C11H24
  XW += x[1] * 31.998000;  // O2
  XW += x[2] * 28.014000;  // N2
  XW += x[3] * 142.286000; // IC10H22
  ROW = rho / XW;

  // Compute conversion, see Eq 11
  for (int id = 0; id < 4; ++id) {
    c[id] = x[id] * ROW;
  }
}

// convert c[species] (molar conc) to x[species] (mole fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCTX(const amrex::Real c[], amrex::Real x[])
{
  amrex::Real sumC = 0;

  // compute sum of c
  for (int id = 0; id < 4; ++id) {
    sumC += c[id];
  }

  //  See Eq 13
  amrex::Real sumCinv = 1.0 / sumC;
  for (int id = 0; id < 4; ++id) {
    x[id] = c[id] * sumCinv;
  }
}

// convert c[species] (molar conc) to y[species] (mass fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCTY(const amrex::Real c[], amrex::Real y[])
{
  amrex::Real CW = 0; // See Eq 12 in CK Manual
  // compute denominator in eq 12 first
  CW += c[0] * 156.313000; // C11H24
  CW += c[1] * 31.998000;  // O2
  CW += c[2] * 28.014000;  // N2
  CW += c[3] * 142.286000; // IC10H22
  // Now compute conversion
  amrex::Real CWinv = 1.0 / CW;
  y[0] = c[0] * 156.313000 * CWinv;
  y[1] = c[1] * 31.998000 * CWinv;
  y[2] = c[2] * 28.014000 * CWinv;
  y[3] = c[3] * 142.286000 * CWinv;
}

// get specific heat at constant volume as a function
// of T for all species (molar units)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCVML(const amrex::Real T, amrex::Real cvml[])
{
  cv_R(cvml, T);

  // convert to chemkin units
  for (int id = 0; id < 4; ++id) {
    cvml[id] *= 8.31446261815324e+07;
  }
}

// get specific heat at constant pressure as a
// function of T for all species (molar units)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPML(const amrex::Real T, amrex::Real cpml[])
{
  cp_R(cpml, T);

  // convert to chemkin units
  for (int id = 0; id < 4; ++id) {
    cpml[id] *= 8.31446261815324e+07;
  }
}

// get internal energy as a function
// of T for all species (molar units)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKUML(const amrex::Real T, amrex::Real uml[])
{
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T
  speciesInternalEnergy(uml, T);

  // convert to chemkin units
  for (int id = 0; id < 4; ++id) {
    uml[id] *= RT;
  }
}

// get enthalpy as a function
// of T for all species (molar units)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHML(const amrex::Real T, amrex::Real hml[])
{
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T
  speciesEnthalpy(hml, T);

  // convert to chemkin units
  for (int id = 0; id < 4; ++id) {
    hml[id] *= RT;
  }
}

// Returns the standard-state entropies in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSML(const amrex::Real T, amrex::Real sml[])
{
  speciesEntropy(sml, T);

  // convert to chemkin units
  for (int id = 0; id < 4; ++id) {
    sml[id] *= 8.31446261815324e+07;
  }
}

// Returns the specific heats at constant volume
// in mass units (Eq. 29)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCVMS(const amrex::Real T, amrex::Real cvms[])
{
  cv_R(cvms, T);
  // multiply by R/molecularweight
  cvms[0] *= 5.319111409897603e+05; // C11H24
  cvms[1] *= 2.598431970171023e+06; // O2
  cvms[2] *= 2.967966951578939e+06; // N2
  cvms[3] *= 5.843486090095469e+05; // IC10H22
}

// Returns the specific heats at constant pressure
// in mass units (Eq. 26)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPMS(const amrex::Real T, amrex::Real cpms[])
{
  cp_R(cpms, T);
  // multiply by R/molecularweight
  cpms[0] *= 5.319111409897603e+05; // C11H24
  cpms[1] *= 2.598431970171023e+06; // O2
  cpms[2] *= 2.967966951578939e+06; // N2
  cpms[3] *= 5.843486090095469e+05; // IC10H22
}

// Returns internal energy in mass units (Eq 30.)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKUMS(const amrex::Real T, amrex::Real ums[])
{
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T

  speciesInternalEnergy(ums, T);

  for (int i = 0; i < 4; i++) {
    ums[i] *= RT * imw(i);
  }
}

// Returns enthalpy in mass units (Eq 27.)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHMS(const amrex::Real T, amrex::Real hms[])
{
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T

  speciesEnthalpy(hms, T);

  for (int i = 0; i < 4; i++) {
    hms[i] *= RT * imw(i);
  }
}

// Returns the entropies in mass units (Eq 28.)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSMS(const amrex::Real T, amrex::Real sms[])
{
  speciesEntropy(sms, T);
  // multiply by R/molecularweight
  sms[0] *= 5.319111409897603e+05; // C11H24
  sms[1] *= 2.598431970171023e+06; // O2
  sms[2] *= 2.967966951578939e+06; // N2
  sms[3] *= 5.843486090095469e+05; // IC10H22
}

// GPU version of productionRate: no more use of thermo namespace vectors
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
comp_qfqr(
  amrex::Real* /*qf*/,
  amrex::Real* /*qr*/,
  const amrex::Real* /*sc*/,
  const amrex::Real* /*sc_qss*/,
  const amrex::Real /*T*/,
  const amrex::Real /*invT*/,
  const amrex::Real /*logT*/)
{
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
productionRate(
  amrex::Real* wdot, const amrex::Real* /*sc*/, const amrex::Real /*T*/)
{

  for (int i = 0; i < 4; ++i) {
    wdot[i] = 0.0;
  }
}

// compute the production rate for each species
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKWC(const amrex::Real T, amrex::Real C[], amrex::Real wdot[])
{

  // convert to SI
  for (int id = 0; id < 4; ++id) {
    C[id] *= 1.0e6;
  }

  // convert to chemkin units
  productionRate(wdot, C, T);

  // convert to chemkin units
  for (int id = 0; id < 4; ++id) {
    C[id] *= 1.0e-6;
    wdot[id] *= 1.0e-6;
  }
}

// Returns the molar production rate of species
// Given P, T, and mass fractions
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKWYP(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real y[],
  amrex::Real wdot[])
{
  amrex::Real c[4]; // temporary storage
  amrex::Real YOW = 0;
  amrex::Real PWORT;

  // Compute inverse of mean molecular wt first
  for (int i = 0; i < 4; i++) {
    YOW += y[i] * imw(i);
  }
  // PW/RT (see Eq. 7)
  PWORT = P / (YOW * 8.31446261815324e+07 * T);
  // multiply by 1e6 so c goes to SI
  PWORT *= 1e6;
  // Now compute conversion (and go to SI)
  for (int i = 0; i < 4; i++) {
    c[i] = PWORT * y[i] * imw(i);
  }

  // convert to chemkin units
  productionRate(wdot, c, T);

  // convert to chemkin units
  for (int id = 0; id < 4; ++id) {
    wdot[id] *= 1.0e-6;
  }
}

// Returns the molar production rate of species
// Given P, T, and mole fractions
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKWXP(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real wdot[])
{
  amrex::Real c[4]; // temporary storage
  amrex::Real PORT =
    1e6 * P / (8.31446261815324e+07 * T); // 1e6 * P/RT so c goes to SI units

  // Compute conversion, see Eq 10
  for (int id = 0; id < 4; ++id) {
    c[id] = x[id] * PORT;
  }

  // convert to chemkin units
  productionRate(wdot, c, T);

  // convert to chemkin units
  for (int id = 0; id < 4; ++id) {
    wdot[id] *= 1.0e-6;
  }
}

// Returns the molar production rate of species
// Given rho, T, and mass fractions
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKWYR(
  const amrex::Real rho,
  const amrex::Real T,
  const amrex::Real y[],
  amrex::Real wdot[])
{
  amrex::Real c[4]; // temporary storage

  // See Eq 8 with an extra 1e6 so c goes to SI
  for (int i = 0; i < 4; i++) {
    c[i] = 1e6 * rho * y[i] * imw(i);
  }

  // call productionRate
  productionRate(wdot, c, T);

  // convert to chemkin units
  for (int id = 0; id < 4; ++id) {
    wdot[id] *= 1.0e-6;
  }
}

// Returns the molar production rate of species
// Given rho, T, and mole fractions
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKWXR(
  const amrex::Real rho,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real wdot[])
{
  amrex::Real c[4];   // temporary storage
  amrex::Real XW = 0; // See Eq 4, 11 in CK Manual
  amrex::Real ROW;
  // Compute mean molecular wt first
  XW += x[0] * 156.313000; // C11H24
  XW += x[1] * 31.998000;  // O2
  XW += x[2] * 28.014000;  // N2
  XW += x[3] * 142.286000; // IC10H22
  // Extra 1e6 factor to take c to SI
  ROW = 1e6 * rho / XW;

  // Compute conversion, see Eq 11
  for (int id = 0; id < 4; ++id) {
    c[id] = x[id] * ROW;
  }

  // convert to chemkin units
  productionRate(wdot, c, T);

  // convert to chemkin units
  for (int id = 0; id < 4; ++id) {
    wdot[id] *= 1.0e-6;
  }
}

//  species unit charge number
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCHRG(int kcharge[])
{
  kcharge[0] = 0; // C11H24
  kcharge[1] = 0; // O2
  kcharge[2] = 0; // N2
  kcharge[3] = 0; // IC10H22
}

//  species charge per unit mass
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCHRGMASS(amrex::Real zk[])
{

  int kchrg[4];
  CKCHRG(kchrg);

  for (int id = 0; id < 4; ++id) {
    zk[id] = 6.02214076e+23 * 1.60217663e-19 * kchrg[id] * imw(id);
  }
}

// compute an approx to the reaction Jacobian
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
aJacobian_precond(
  amrex::Real* J,
  const amrex::Real* /*sc*/,
  const amrex::Real /*T*/,
  const int /*HP*/)
{

#if defined(PELE_COMPILE_AJACOBIAN) || !defined(AMREX_USE_HIP)
  for (int i = 0; i < 25; i++) {
    J[i] = 0.0;
  }
#else
  amrex::Abort();
#endif
}

// compute an approx to the reaction Jacobian (for preconditioning)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
DWDOT_SIMPLIFIED(
  amrex::Real* J, const amrex::Real* sc, const amrex::Real* Tp, const int* HP)
{
  amrex::Real c[4];

  for (int k = 0; k < 4; k++) {
    c[k] = 1.e6 * sc[k];
  }

  aJacobian_precond(J, c, *Tp, *HP);

  // dwdot[k]/dT
  // dTdot/d[X]
  for (int k = 0; k < 4; k++) {
    J[20 + k] *= 1.e-6;
    J[k * 5 + 4] *= 1.e6;
  }
}

// compute the reaction Jacobian
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
aJacobian(
  amrex::Real* J,
  const amrex::Real* /*sc*/,
  const amrex::Real /*T*/,
  const int /*consP*/)
{

#if defined(PELE_COMPILE_AJACOBIAN) || !defined(AMREX_USE_HIP)
  for (int i = 0; i < 25; i++) {
    J[i] = 0.0;
  }
#else
  amrex::Abort();
#endif
}

// compute the reaction Jacobian
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
DWDOT(
  amrex::Real* J,
  const amrex::Real* sc,
  const amrex::Real* Tp,
  const int* consP)
{
  amrex::Real c[4];

  for (int k = 0; k < 4; k++) {
    c[k] = 1.e6 * sc[k];
  }

  aJacobian(J, c, *Tp, *consP);

  // dwdot[k]/dT
  // dTdot/d[X]
  for (int k = 0; k < 4; k++) {
    J[20 + k] *= 1.e-6;
    J[k * 5 + 4] *= 1.e6;
  }
}

// Transport function declarations

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetLENIMC(int* LENIMC)
{
  *LENIMC = 16;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetLENRMC(int* LENRMC)
{
  *LENRMC = 412;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetNO(int* NO)
{
  *NO = 4;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetKK(int* KK)
{
  *KK = 4;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetNLITE(int* NLITE)
{
  *NLITE = 0;
}

// Patm in ergs/cm3
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetPATM(amrex::Real* PATM)
{
  *PATM = 0.1013250000000000E+07;
}

// the molecular weights in g/mol
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetWT(amrex::Real* WT)
{
  WT[0] = 1.56313000E+02;
  WT[1] = 3.19980000E+01;
  WT[2] = 2.80140000E+01;
  WT[3] = 1.42286000E+02;
}

// the lennard-jones potential well depth eps/kb in K
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetEPS(amrex::Real* EPS)
{
  EPS[0] = 6.50000000E+02;
  EPS[1] = 1.07400000E+02;
  EPS[2] = 9.75300000E+01;
  EPS[3] = 6.50000000E+02;
}

// the lennard-jones collision diameter in Angstroms
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetSIG(amrex::Real* SIG)
{
  SIG[0] = 8.35000000E+00;
  SIG[1] = 3.45800000E+00;
  SIG[2] = 3.62100000E+00;
  SIG[3] = 8.35000000E+00;
}

// the dipole moment in Debye
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetDIP(amrex::Real* DIP)
{
  DIP[0] = 0.00000000E+00;
  DIP[1] = 0.00000000E+00;
  DIP[2] = 0.00000000E+00;
  DIP[3] = 0.00000000E+00;
}

// the polarizability in cubic Angstroms
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetPOL(amrex::Real* POL)
{
  POL[0] = 0.00000000E+00;
  POL[1] = 1.60000000E+00;
  POL[2] = 1.76000000E+00;
  POL[3] = 0.00000000E+00;
}

// the rotational relaxation collision number at 298 K
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetZROT(amrex::Real* ZROT)
{
  ZROT[0] = 1.00000000E+00;
  ZROT[1] = 3.80000000E+00;
  ZROT[2] = 4.00000000E+00;
  ZROT[3] = 1.00000000E+00;
}

// 0: monoatomic, 1: linear, 2: nonlinear
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetNLIN(int* NLIN)
{
  NLIN[0] = 2;
  NLIN[1] = 1;
  NLIN[2] = 1;
  NLIN[3] = 2;
}

// Poly fits for the viscosities, dim NO*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFETA(amrex::Real* COFETA)
{
  COFETA[0] = -1.57539206E+01;
  COFETA[1] = 4.51057467E-01;
  COFETA[2] = 1.57706649E-01;
  COFETA[3] = -1.19883346E-02;
  COFETA[4] = -1.68118998E+01;
  COFETA[5] = 2.52362554E+00;
  COFETA[6] = -2.49309128E-01;
  COFETA[7] = 1.10211025E-02;
  COFETA[8] = -1.62526676E+01;
  COFETA[9] = 2.24839597E+00;
  COFETA[10] = -2.13428438E-01;
  COFETA[11] = 9.46192413E-03;
  COFETA[12] = -1.58009313E+01;
  COFETA[13] = 4.51057467E-01;
  COFETA[14] = 1.57706649E-01;
  COFETA[15] = -1.19883346E-02;
}

// Poly fits for the conductivities, dim NO*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFLAM(amrex::Real* COFLAM)
{
  COFLAM[0] = -1.56093273E+01;
  COFLAM[1] = 5.98125337E+00;
  COFLAM[2] = -4.00000620E-01;
  COFLAM[3] = 6.18804016E-03;
  COFLAM[4] = -3.01283106E+00;
  COFLAM[5] = 3.37554996E+00;
  COFLAM[6] = -3.43353130E-01;
  COFLAM[7] = 1.51043449E-02;
  COFLAM[8] = 1.15506870E+01;
  COFLAM[9] = -2.91452034E+00;
  COFLAM[10] = 5.55043078E-01;
  COFLAM[11] = -2.75172223E-02;
  COFLAM[12] = -1.60779072E+01;
  COFLAM[13] = 6.11517254E+00;
  COFLAM[14] = -4.13760039E-01;
  COFLAM[15] = 6.64416792E-03;
}

// Poly fits for the diffusion coefficients, dim NO*KK*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFD(amrex::Real* COFD)
{
  COFD[0] = -1.86096642E+01;
  COFD[1] = 2.58040365E+00;
  COFD[2] = -1.08977961E-02;
  COFD[3] = -3.70103387E-03;
  COFD[4] = -2.25525117E+01;
  COFD[5] = 5.43595045E+00;
  COFD[6] = -4.70905585E-01;
  COFD[7] = 1.96130544E-02;
  COFD[8] = -2.22283977E+01;
  COFD[9] = 5.34409219E+00;
  COFD[10] = -4.61334919E-01;
  COFD[11] = 1.92897986E-02;
  COFD[12] = -1.85856066E+01;
  COFD[13] = 2.58040365E+00;
  COFD[14] = -1.08977961E-02;
  COFD[15] = -3.70103387E-03;
  COFD[16] = -2.25525117E+01;
  COFD[17] = 5.43595045E+00;
  COFD[18] = -4.70905585E-01;
  COFD[19] = 1.96130544E-02;
  COFD[20] = -1.53110597E+01;
  COFD[21] = 3.37317428E+00;
  COFD[22] = -2.24900439E-01;
  COFD[23] = 9.81228151E-03;
  COFD[24] = -1.50096252E+01;
  COFD[25] = 3.25515933E+00;
  COFD[26] = -2.09710110E-01;
  COFD[27] = 9.15941830E-03;
  COFD[28] = -2.25442054E+01;
  COFD[29] = 5.43595045E+00;
  COFD[30] = -4.70905585E-01;
  COFD[31] = 1.96130544E-02;
  COFD[32] = -2.22283977E+01;
  COFD[33] = 5.34409219E+00;
  COFD[34] = -4.61334919E-01;
  COFD[35] = 1.92897986E-02;
  COFD[36] = -1.50096252E+01;
  COFD[37] = 3.25515933E+00;
  COFD[38] = -2.09710110E-01;
  COFD[39] = 9.15941830E-03;
  COFD[40] = -1.47639411E+01;
  COFD[41] = 3.15955654E+00;
  COFD[42] = -1.97590757E-01;
  COFD[43] = 8.64692156E-03;
  COFD[44] = -2.22209619E+01;
  COFD[45] = 5.34409219E+00;
  COFD[46] = -4.61334919E-01;
  COFD[47] = 1.92897986E-02;
  COFD[48] = -1.85856066E+01;
  COFD[49] = 2.58040365E+00;
  COFD[50] = -1.08977961E-02;
  COFD[51] = -3.70103387E-03;
  COFD[52] = -2.25442054E+01;
  COFD[53] = 5.43595045E+00;
  COFD[54] = -4.70905585E-01;
  COFD[55] = 1.96130544E-02;
  COFD[56] = -2.22209619E+01;
  COFD[57] = 5.34409219E+00;
  COFD[58] = -4.61334919E-01;
  COFD[59] = 1.92897986E-02;
  COFD[60] = -1.85626536E+01;
  COFD[61] = 2.58040365E+00;
  COFD[62] = -1.08977961E-02;
  COFD[63] = -3.70103387E-03;
}

// List of specs with small weight, dim NLITE
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetKTDIF(int* /*KTDIF*/)
{
}

// Poly fits for thermal diff ratios, dim NO*NLITE*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFTD(amrex::Real* /*COFTD*/)
{
}

// compute the critical parameters for each species
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
GET_CRITPARAMS(
  amrex::Real* Tci, amrex::Real* ai, amrex::Real* bi, amrex::Real* acentric_i)
{

  amrex::Real EPS[4];
  amrex::Real SIG[4];
  amrex::Real wt[4];
  amrex::Real Rcst = 83.144598; // in bar [CGS] !
  amrex::Real avogadro = 6.02214199e23;
  amrex::Real boltzmann = 1.3806503e-16; // we work in CGS

  egtransetEPS(EPS);
  egtransetSIG(SIG);
  get_mw(wt);

  // species 0: C11H24
  Tci[0] = 1.316 * EPS[0];
  ai[0] = (5.55 * avogadro * avogadro * EPS[0] * boltzmann * 1e-24 * SIG[0] *
           SIG[0] * SIG[0]) /
          (wt[0] * wt[0]);
  bi[0] = 0.855 * avogadro * 1e-24 * SIG[0] * SIG[0] * SIG[0] / (wt[0]);
  acentric_i[0] = 0.0;

  // species 1: O2
  // Imported from NIST
  Tci[1] = 154.581000;
  ai[1] = 1e6 * 0.42748 * Rcst * Rcst * Tci[1] * Tci[1] /
          (31.998800 * 31.998800 * 50.430466);
  bi[1] = 0.08664 * Rcst * Tci[1] / (31.998800 * 50.430466);
  acentric_i[1] = 0.022200;

  // species 2: N2
  // Imported from NIST
  Tci[2] = 126.192000;
  ai[2] = 1e6 * 0.42748 * Rcst * Rcst * Tci[2] * Tci[2] /
          (28.013400 * 28.013400 * 33.958000);
  bi[2] = 0.08664 * Rcst * Tci[2] / (28.013400 * 33.958000);
  acentric_i[2] = 0.037200;

  // species 3: IC10H22
  Tci[3] = 1.316 * EPS[3];
  ai[3] = (5.55 * avogadro * avogadro * EPS[3] * boltzmann * 1e-24 * SIG[3] *
           SIG[3] * SIG[3]) /
          (wt[3] * wt[3]);
  bi[3] = 0.855 * avogadro * 1e-24 * SIG[3] * SIG[3] * SIG[3] / (wt[3]);
  acentric_i[3] = 0.0;
}

// compute the critical parameter quantities for each species for SRK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
GET_CRITPARAMS_SRK(
  amrex::Real* sqrtOneOverTc,
  amrex::Real* sqrtAsti,
  amrex::Real* Bi,
  amrex::Real* Fomega)
{

  // species 0: C11H24
  // Computed from Lennard-Jones
  sqrtOneOverTc[0] = 3.4191281377498e-02;
  sqrtAsti[0] = 6.5604045589555e+04;
  Bi[0] = 1.9177033813400e+00;
  Fomega[0] = 4.8508000000000e-01;

  // species 1: O2
  // Imported from NIST
  sqrtOneOverTc[1] = 8.0430717653241e-02;
  sqrtAsti[1] = 3.6980081924740e+04;
  Bi[1] = 6.9005216444999e-01;
  Fomega[1] = 5.1945301904908e-01;

  // species 2: N2
  // Imported from NIST
  sqrtOneOverTc[2] = 8.9019282240563e-02;
  sqrtAsti[2] = 4.2022983929562e+04;
  Bi[2] = 9.5560052221563e-01;
  Fomega[2] = 5.4259343186608e-01;

  // species 3: IC10H22
  // Computed from Lennard-Jones
  sqrtOneOverTc[3] = 3.4191281377498e-02;
  sqrtAsti[3] = 7.2071498097073e+04;
  Bi[3] = 2.1067565933921e+00;
  Fomega[3] = 4.8508000000000e-01;
}

// gauss-jordan solver external routine
// Replace this routine with the one generated by the Gauss Jordan solver of DW
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
sgjsolve(amrex::Real* /*A*/, amrex::Real* /*x*/, amrex::Real* /*b*/)
{
  amrex::Abort("sgjsolve not implemented, choose a different solver ");
}

// Replace this routine with the one generated by the Gauss Jordan solver of DW
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
sgjsolve_simplified(amrex::Real* /*A*/, amrex::Real* /*x*/, amrex::Real* /*b*/)
{
  amrex::Abort(
    "sgjsolve_simplified not implemented, choose a different solver ");
}

#endif
