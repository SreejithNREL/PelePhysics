#ifndef SPRAY_DRYING_H
#define SPRAY_DRYING_H
//======================================================================================
//  SprayDrying.H
//
//  GPU-safe, allocation-free port of the three-stage slurry-droplet drying model of
//    Bhattacharjee & Gnanaskandan, Phys. Fluids 36, 083354 (2024)
//  from the research code
//    Bi-Component-Droplet-Evaporation / "Non-Uniform Temperature Model"
//    (main.cpp + functions.cpp + inputs.h)
//
//  Stage 1 : shrinking droplet, internal T(r) and solid-concentration C(r) fields.
//  Stage 2 : frozen outer radius, porous crust + receding wet core, Darcy vapour flux
//            through the crust pores (Newton-Raphson solve for mdot_v).
//  Stage 3 : sensible heating of the fully dry particle.
//
//  HARD CONSTRAINTS honoured here:
//    * no dynamic allocation (no std::vector / new / malloc) -- all radial arrays are
//      fixed-size C arrays with compile-time bounds
//    * no iostream, no file I/O, no exit(), no exceptions
//    * every function AMREX_GPU_DEVICE AMREX_FORCE_INLINE
//    * all parcel state in one POD struct
//
//  BUG-FOR-BUG FIDELITY NOTE
//  -------------------------
//  The original code contains several integer-division expressions "(4/3)*M_PI*..."
//  which evaluate to 1*M_PI, i.e. the 4/3 factor is silently dropped.  Those
//  expressions feed the crust/wet-core mass bookkeeping and the reported volume
//  fractions.  They are reproduced EXACTLY here (marked "ORIGINAL (4/3)->1") so that
//  this port reproduces the reference Result.dat.  Search for FOURTHIRDS_BUG to find
//  every occurrence if you ever want to switch to the physically correct form.
//======================================================================================

#include <cmath>

#ifndef AMREX_GPU_DEVICE
#define AMREX_GPU_DEVICE
#define AMREX_FORCE_INLINE inline
#endif

namespace SprayDrying {

//--------------------------------------------------------------------------------------
// Compile-time array bounds.  Original run used numr = 20, numr_cr = 5, numr_wc = 20.
//--------------------------------------------------------------------------------------
// Reduced from 32/16 to exactly the Silica_101 case size (numr = numr_wc = 20,
// numr_cr = 5).  When built inside PeleLMeX the value is taken from the build
// flag -DPELE_SPRAY_DRYING_NR (Exec/Make.PeleLMeX, SPRAY_DRYING_NR), which must
// therefore be set to 20 for this case.
#ifdef PELE_SPRAY_DRYING_NR
constexpr int NR_MAX = PELE_SPRAY_DRYING_NR;
#else
constexpr int NR_MAX    = 20;   // droplet / wet-core / dry-core radial points
#endif
constexpr int NR_CR_MAX = 5;    // crust radial points
constexpr int NR_NEWTON = 100;  // hard cap on Newton-Raphson iterations (was while i<=100)

constexpr double SD_PI = 3.14159265358979323846;

//======================================================================================
//  Material / model constants (mirrors inputs.h, Inputs/Silica_101)
//======================================================================================
struct DryingProps
{
    // ---- droplet ----
    double Rp_0     = 980e-6;          // initial droplet radius [m]
    double Td_0     = 29.8 + 273.15;   // initial droplet temperature [K]
    double X0       = 2.33;            // initial moisture content mw/ms [kg/kg]
    int    numr     = 20;              // radial points, stage 1
    int    numr_cr  = 5;               // radial points in crust
    int    numr_wc  = 20;              // radial points in wet core

    // ---- solvent (water) ----
    double rho_l    = 1000.0;          // [kg/m^3]
    double Kw       = 1.445;           // [W/m/K]  (unused; polynomial fit used instead)
    double Cpw      = 4.22e3;          // [J/kg/K] (unused; polynomial fit used instead)
    double Tboil    = 100.0 + 273.15;  // [K]
    double MW_l     = 0.01801;         // [kg/mol]

    // ---- solute (silica) ----
    double rho_s    = 1800.0;          // [kg/m^3]
    double Ks       = 0.660;           // [W/m/K]
    double Cps      = 700.0;           // [J/kg/K]
    double Rs_0     = 16e-9;           // primary particle radius [m]
    double MW_s     = 0.06008;         // [kg/mol]
    double Cs_crit_shell = 0.74*1800.0;// critical solid concentration for shell [kg/m^3]
    double k_s      = -5.0;            // Nesic & Vodnik sorption constant
    double n_s      = 0.5;             // Nesic & Vodnik sorption exponent
    double Dsl      = 1.0e-7;          // solute diffusivity [m^2/s]
    double eps      = 0.4;             // crust porosity
    double beta     = 7.5;             // porosity power coefficient (Darcy)
    double d_pore   = 1.369e-5;        // crust pore diameter [m]
    int    knudsen_diff = 1;           // 1 = include Knudsen diffusion in pores

    // ---- gas / ambient ----
    double MW_a     = 0.02897;         // [kg/mol]
    double Rgas     = 8.314;           // [J/mol/K]
    double Dab      = 3.564e-10;       // vapour diffusivity prefactor [m^2/s/K^1.75]
    double Xg       = 0.01141;         // ambient humidity ratio [kg/kg dry air]

    // ---- crust ----
    double ini_thick = 300.0*(2.0*16e-9); // initial crust thickness [m]
    double dc        = 2.64e-10;          // water-vapour collision diameter [m]
    double rho_v_cr  = 0.59;              // vapour density in crust (crust density) [kg/m^3]
    double rho_v_vap = 0.598;             // vapour density in crust (vapour mass)  [kg/m^3]

    // ---- variable-pressure option ----
    int    var_Pt      = 0;
    double amp         = 0.1*101325.0;
    double time_period = 5.0;
    double vert_shift  = 101325.0;

    // ---- activation-energy model (Mezhericher model-2) ----
    double ae_a = 0.19;
    double ae_b = 1.1;
    double ae_c = 0.02;

    // ---- sub-stage timestep ladder (see recommendedDryingDt) ----
    double fdt  = 1e-3;   // stage 1
    double sidt = 1e-10;  // stage 2, initial  (thin crust)
    double smdt = 1e-9;   // stage 2, mid      (dp >= 30 * 2*Rs_0)
    double sldt = 1e-6;   // stage 2, late     (dp >  300 * 2*Rs_0) and stage 3
};

//======================================================================================
//  Ambient gas seen by the parcel
//======================================================================================
struct DryingGasState
{
    double T_inf = 101.0 + 273.15;  // gas temperature [K]
    double P     = 101325.0;        // gas pressure [Pa]
    double RH    = 0.4/100.0;       // relative humidity [-]
    double V_rel = 1.73;            // |u_gas - u_parcel| [m/s]
};

//======================================================================================
//  Per-parcel state (POD, fixed width)
//======================================================================================
struct DryingParcelState
{
    // --- radial fields ---------------------------------------------------------------
    double T[NR_MAX];        // stage 1: droplet T(r); stage 2: wet-core T(r); stage 3: dry-core T(r)
    double C[NR_MAX];        // stage 1: solvent concentration C(r) [kg/m^3]; unused in stages 2-3
    double T_cr[NR_CR_MAX];  // crust T(r), stages 2 and 3

    // --- geometry / mass -------------------------------------------------------------
    double R_out;            // outer radius (droplet radius in stage 1, frozen particle radius after)
    double R_int;            // crust / wet-core interface radius (stages 2, 3)
    double crust_thick;      // R_out - R_int
    double mass;             // total parcel mass (solid + remaining solvent) [kg]
    double mass0;            // initial parcel mass [kg]
    double m_solid;          // (constant) solid mass [kg]

    // --- carried scalars -------------------------------------------------------------
    double mdot;             // last evaporation rate [kg/s] (also Newton-Raphson seed in stage 2)
    double t;                // parcel-local elapsed drying time [s]
    double dt_cur;           // current recommended dt (latched by recommendedDryingDt)
    double D_vap;            // last vapour diffusivity [m^2/s] (stage 3 reuses stage-2 value)

    // --- diagnostics (written every substep; optional, only needed for reporting) -----
    double k_eff;            // effective conductivity reported by the active stage
    double cp_eff;           // effective specific heat
    double rho_eff;          // effective density
    double vol_frac_l;       // liquid volume fraction
    double mfrac_s;          // solid mass fraction
    double d_sq;             // d^2-law diameter squared [m^2]

    // --- integers --------------------------------------------------------------------
    int stage;               // 1, 2 or 3
    int nr;                  // active radial points, stage-1 droplet
    int nr_cr;               // active radial points, crust
    int nr_wc;               // active radial points, wet/dry core
    int nstep;               // global step counter (activation-energy model needs n > 1)
    int nsub;                // substage step counter (ns in the original)
    int lat_mid;             // latch: smdt already engaged (n_sec in the original)
    int lat_late;            // latch: sldt already engaged (n_th  in the original)
};

//======================================================================================
//  Small stack-resident property bundles (no allocation)
//======================================================================================
struct ThermoBundle
{
    double l[4];    // stage 1/2: 0 cp, 1 mu, 2 k, 3 h_fg   (water)
                    // stage 3  : 0 cp, 1 mu, 2 k, 3 rho    (air inside particle)
    double a[4];    // 0 cp, 1 mu, 2 k, 3 rho  (ambient air)
    double dim[5];  // 0 Re, 1 Pr, 2 Sc, 3 Nu, 4 Sh
};

struct MassBundle
{
    double Pvs, Pv_inf, rho_vg, Bm, rho_vs, rho_g;
};

struct ParticleBundle
{
    double rho_cr, m_cr, X0, m_wc_s, m_wc_w, m_p, X, X_wc, rho_wc, m_wc;
    double K_cr, Cp_cr, K_wc, c, Cp_wc, Pvi, Pv_inf, M_vap;
};

struct ParticleBundleDry
{
    double rho_cr, m_cr, m_p, rho_dc, m_dc, K_cr, Cp_cr, K_dc, Cp_dc;
};

//======================================================================================
//  Helpers
//======================================================================================

// i-th point of linspace(a, b, n) -- replaces the original's heap-allocating linspace()
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
double linspaceAt(double a, double b, int n, int i)
{
    return a + double(i) * ((b - a) / double(n - 1));
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
double ambientPressure(const DryingProps& p, const DryingGasState& gas, double tt)
{
    if (p.var_Pt == 1) {
        return p.amp * std::sin(2.0*SD_PI*tt/p.time_period) + p.vert_shift;
    }
    return gas.P;
}

// solid concentration / mass fractions from solvent concentration C  (mass_fraction())
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
double solidConc(const DryingProps& p, double Cl)
{
    return p.rho_s * (1.0 - (Cl / p.rho_l));
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
double solidMassFrac(const DryingProps& p, double Cs)
{
    return Cs * p.rho_s / (p.rho_s*p.rho_l + Cs*(p.rho_s - p.rho_l));
}

//--------------------------------------------------------------------------------------
//  thermophysical()  -- water + ambient air properties and dimensionless groups
//--------------------------------------------------------------------------------------
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void thermophysical(const DryingProps& p, const DryingGasState& gas,
                    double T, double Tf, double d, double D, double tt,
                    ThermoBundle& tb)
{
    const double Pt = ambientPressure(p, gas, tt);

    // water
    tb.l[0] = (2.108052e-9*T*T*T*T - 2.841073e-6*T*T*T + 1.441786e-3*T*T
               - 3.260186e-1*T + 3.18591e1) * 1.0e3;                        // cp   [J/kg/K]
    tb.l[1] = 2.338519e-11*T*T*T*T - 3.24421e-8*T*T*T + 1.692507e-5*T*T
              - 3.940708e-3*T + 3.463261e-1;                                // mu   [Ns/m^2]
    tb.l[2] = 5.33818e-10*T*T*T*T - 6.91901e-7*T*T*T + 3.25465e-4*T*T
              - 6.44535e-2*T + 5.01189e0;                                   // k    [W/m/K]
    tb.l[3] = (-0.0013*(T-273.15)*(T-273.15) - 2.29618*(T-273.15) + 2500.0)*1e3; // h_fg [J/kg]

    // ambient air
    tb.a[0] = 4e-4*(Tf-273.15)*(Tf-273.15) + 2.38e-2*(Tf-273.15) + 1004.5;
    tb.a[1] = 1.097e-6*(std::sqrt(Tf)/(1.453 - 0.0243*std::sqrt(Tf)));
    tb.a[2] = 1.97e-4*std::pow(Tf, 0.858);
    tb.a[3] = Pt/((8.314/p.MW_a)*Tf);

    tb.dim[0] = tb.a[3]*d*gas.V_rel/tb.a[1];                                    // Re
    tb.dim[1] = tb.a[0]*tb.a[1]/tb.a[2];                                        // Pr
    tb.dim[2] = tb.a[1]/(tb.a[3]*D);                                            // Sc
    tb.dim[3] = 2.0 + 0.6*std::sqrt(tb.dim[0])*std::pow(tb.dim[1], 0.333);      // Nu
    tb.dim[4] = 2.0 + 0.6*std::sqrt(tb.dim[0])*std::pow(tb.dim[2], 0.333);      // Sh
}

//--------------------------------------------------------------------------------------
//  thermophysical_dp() -- stage 3: air inside the dry particle + ambient air
//--------------------------------------------------------------------------------------
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void thermophysicalDry(const DryingProps& p, const DryingGasState& gas,
                       double T, double Tf, double d, double D, double tt,
                       ThermoBundle& tb)
{
    const double Pt = ambientPressure(p, gas, tt);

    tb.l[0] = 4e-4*(T-273.15)*(T-273.15) + 2.38e-2*(T-273.15) + 1004.5;
    tb.l[1] = 1.097e-6*(std::sqrt(T)/(1.453 - 0.0243*std::sqrt(T)));
    tb.l[2] = 1.97e-4*std::pow(T, 0.858);
    tb.l[3] = Pt/((8.314/p.MW_a)*T);

    tb.a[0] = 4e-4*(Tf-273.15)*(Tf-273.15) + 2.38e-2*(Tf-273.15) + 1004.5;
    tb.a[1] = 1.097e-6*(std::sqrt(Tf)/(1.453 - 0.0243*std::sqrt(Tf)));
    tb.a[2] = 1.97e-4*std::pow(Tf, 0.858);
    tb.a[3] = Pt/((8.314/p.MW_a)*Tf);

    tb.dim[0] = tb.a[3]*d*gas.V_rel/tb.a[1];
    tb.dim[1] = tb.a[0]*tb.a[1]/tb.a[2];
    tb.dim[2] = tb.a[1]/(tb.a[3]*D);
    tb.dim[3] = 2.0 + 0.6*std::sqrt(tb.dim[0])*std::pow(tb.dim[1], 0.333);
    tb.dim[4] = 2.0 + 0.6*std::sqrt(tb.dim[0])*std::pow(tb.dim[2], 0.333);
}

//--------------------------------------------------------------------------------------
//  mass_eqn_properties() -- surface vapour state, Spalding number, activation energy
//--------------------------------------------------------------------------------------
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void massEqnProperties(const DryingProps& p, const DryingGasState& gas,
                       double hfg, double T, double W_ln,
                       double Mtot, double Mtot_0, int n, double tt,
                       MassBundle& mb)
{
    const double Pt = ambientPressure(p, gas, tt);

    const double Psat     = Pt*std::exp((-hfg/(p.Rgas/p.MW_l))*(1.0/T - 1.0/p.Tboil));
    mb.Pvs                = Psat*(1.0 - std::exp(p.k_s*std::pow(W_ln, p.n_s)));
    const double Psat_inf = Pt*std::exp((-hfg/(p.Rgas/p.MW_l))*(1.0/gas.T_inf - 1.0/p.Tboil));
    mb.Pv_inf             = gas.RH*Psat_inf;
    mb.rho_vg             = p.MW_l*mb.Pv_inf/(p.Rgas*gas.T_inf);
    mb.rho_vs             = p.MW_l*mb.Pvs/(p.Rgas*T);

    // Mezhericher activation-energy model (model-2); skipped on the very first step
    if (n > 1) {
        const double Xd  = (Mtot*(1.0 + p.X0)/Mtot_0) - 1.0;
        const double Evg = -p.Rgas*T*std::log(mb.rho_vg/mb.rho_vs);
        const double Ev  = p.ae_a*Evg*std::exp(-p.ae_b*std::pow(Xd - p.Xg, p.ae_c));
        const double shi = std::exp(-Ev/(p.Rgas*T));
        mb.rho_vs       *= shi;
    }

    const double Xs     = mb.Pvs/Pt;
    const double X_inf  = mb.Pv_inf/Pt;
    const double MW_mix = Xs*p.MW_l + (1.0 - Xs)*p.MW_a;
    const double MW_avg = 0.5*(MW_mix + p.MW_a);
    mb.rho_g            = Pt/((p.Rgas/MW_avg)*gas.T_inf);
    const double Ys     = Xs*(p.MW_l/MW_mix);
    const double Yinf   = X_inf*(p.MW_l/MW_mix);   // NOTE: original divides by MW_mix, not MW_a
    mb.Bm               = (Ys - Yinf)/(1.0 - Ys);
}

//--------------------------------------------------------------------------------------
//  mass_tr_rate()  -- stage 1 evaporation rate
//--------------------------------------------------------------------------------------
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
double massTrRate(const MassBundle& mb, const ThermoBundle& tb, double d, double Dv)
{
    const double Sh_s = tb.dim[4]/std::pow(1.0 + mb.Bm, 0.7);
    const double h_d  = Sh_s*Dv/d;
    return h_d*SD_PI*d*d*(mb.rho_vs - mb.rho_vg);
}

//--------------------------------------------------------------------------------------
//  particle_properties() -- stage 2 crust / wet-core bookkeeping
//--------------------------------------------------------------------------------------
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void particleProperties(const DryingProps& p, const DryingGasState& gas,
                        double R_s, double Ri, double M_s,
                        const ThermoBundle& tb, double T, double Cs_s,
                        double Mtot, double Mtot_0, double tt,
                        ParticleBundle& pp)
{
    const double Pt = ambientPressure(p, gas, tt);
    const double Rs3 = R_s*R_s*R_s;
    const double Ri3 = Ri*Ri*Ri;

    pp.rho_cr = p.rho_s*(1.0 - p.eps) + (p.eps*p.rho_v_cr);
    // FOURTHIRDS_BUG: ORIGINAL (4/3)->1
    pp.m_cr   = SD_PI*pp.rho_cr*(Rs3 - Ri3);
    pp.X0     = (Mtot_0 - M_s)/M_s;
    pp.m_wc_s = (Mtot_0/(1.0 + pp.X0)) - pp.m_cr;
    // FOURTHIRDS_BUG: ORIGINAL (4/3)->1
    pp.m_wc_w = (p.rho_l/p.rho_s)*((SD_PI*Ri3*p.rho_s) - pp.m_wc_s);
    // FOURTHIRDS_BUG: ORIGINAL (4/3)->1
    pp.m_p    = (Mtot_0/(1.0 + pp.X0))*(1.0 - (p.rho_l/p.rho_s))
                + (SD_PI*p.rho_l*(p.eps*Ri3 + ((1.0 - p.eps)*Rs3)));
    pp.X      = (pp.m_p*(1.0 + pp.X0)/Mtot_0) - 1.0;

    const double Mw_0 = Mtot_0 - M_s;
    pp.X_wc   = (Mw_0 - (Mtot_0 - Mtot))/(M_s - pp.m_cr);
    pp.rho_wc = (1.0 + pp.X_wc)*p.rho_s*p.rho_l/(p.rho_l + (pp.X_wc*p.rho_s));
    // FOURTHIRDS_BUG: ORIGINAL (4/3)->1
    pp.m_wc   = SD_PI*pp.rho_wc*Ri3;

    pp.K_cr   = p.Ks*(1.0 - p.eps) + p.eps*tb.a[2];
    pp.Cp_cr  = p.Cps;

    const double delta = 1.0 - 6.0*pp.m_wc_s/(SD_PI*p.rho_s*(2.0*Ri)*(2.0*Ri)*(2.0*Ri));
    pp.K_wc   = delta*tb.l[2] + (1.0 - delta)*p.Ks;
    pp.c      = pp.m_wc_s/pp.m_wc;
    pp.Cp_wc  = tb.l[0]*(1.0 - pp.c) + p.Cps*pp.c;

    // FOURTHIRDS_BUG: ORIGINAL (4/3)->1
    const double V_vap = SD_PI*(Rs3 - Ri3)*p.eps;
    pp.M_vap  = V_vap*p.rho_v_vap;

    // vapour pressure at the interface (sorption isotherm at the critical concentration)
    const double Psat     = Pt*std::exp((-tb.l[3]/(p.Rgas/p.MW_l))*(1.0/T - 1.0/p.Tboil));
    pp.Pvi                = Psat*(1.0 - std::exp(p.k_s*std::pow(Cs_s, p.n_s)));
    const double Psat_inf = Pt*std::exp((-tb.l[3]/(p.Rgas/p.MW_l))*(1.0/gas.T_inf - 1.0/p.Tboil));
    pp.Pv_inf             = gas.RH*Psat_inf;

    // activation energy (model-2)
    const double Xd  = (Mtot*(1.0 + p.X0)/Mtot_0) - 1.0;
    const double Evg = -p.Rgas*T*std::log(pp.Pv_inf/pp.Pvi);
    const double Ev  = p.ae_a*Evg*std::exp(-p.ae_b*std::pow(Xd - p.Xg, p.ae_c));
    const double shi = std::exp(-Ev/(p.Rgas*T));
    pp.Pvi          *= shi;
}

//--------------------------------------------------------------------------------------
//  mass_tr_rate_second_stage() -- Darcy flux through the crust, Newton-Raphson solve
//  (Eq. 18 of the paper).  Fixed iteration cap, no allocation, no I/O.
//--------------------------------------------------------------------------------------
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
double massTrRateSecondStage(const DryingProps& p, const DryingGasState& gas,
                             const ThermoBundle& tb, const ParticleBundle& pp,
                             double Dcr, double R_s, double Ri,
                             double T_cr_in, double T_cr_surf,
                             double mv, double tt)
{
    const double Pt = ambientPressure(p, gas, tt);

    const double Bm   = 1.9e3*(gas.T_inf - T_cr_surf)/tb.l[3];
    const double Sh_s = tb.dim[4]/std::pow(1.0 + Bm, 0.7);
    const double hd   = Sh_s*Dcr/(2.0*R_s);

    const double a1 = -8.0*SD_PI*std::pow(p.eps, p.beta)*Dcr*p.MW_l*Pt
                      /(p.Rgas*(T_cr_surf + T_cr_in));
    const double a2 = R_s*Ri/(R_s - Ri);
    const double a3 = p.Rgas/(4.0*SD_PI*p.MW_l*hd*R_s*R_s);
    const double a4 = pp.Pv_inf/gas.T_inf;

    const double a = a1*a2;
    const double b = Pt - pp.Pvi;
    const double c = Pt;
    const double d = a3;
    const double e = a4;
    const double f = T_cr_surf;

    double x1 = mv;
    double r  = 1e-4;
    for (int i = 1; i <= NR_NEWTON && r >= 1e-4; ++i) {
        const double den = c - (d*x1 + e)*f;
        const double fx  = x1 - a*std::log(b/den);
        const double fxd = 1.0 - a*d*f/den;
        const double y   = x1 - fx/fxd;
        r  = std::fabs((x1 - y)/y);
        x1 = y;
    }
    return x1;
}

//--------------------------------------------------------------------------------------
//  particle_properties_dp() -- stage 3
//--------------------------------------------------------------------------------------
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void particlePropertiesDry(const DryingProps& p, double R_s, double Ri, double M_s,
                           const ThermoBundle& tb, ParticleBundleDry& ps)
{
    const double Rs3 = R_s*R_s*R_s;
    const double Ri3 = Ri*Ri*Ri;
    ps.rho_cr = p.rho_s*(1.0 - p.eps) + (p.eps*tb.l[3]);
    // FOURTHIRDS_BUG: ORIGINAL (4/3)->1
    ps.m_cr   = SD_PI*p.eps*tb.l[3]*(Rs3 - Ri3) + M_s;
    ps.rho_dc = tb.l[3];
    // FOURTHIRDS_BUG: ORIGINAL (4/3)->1
    ps.m_dc   = SD_PI*ps.rho_dc*Ri3;
    ps.m_p    = ps.m_cr + ps.m_dc;
    ps.K_cr   = p.Ks*(1.0 - p.eps) + p.eps*tb.l[2];
    ps.Cp_cr  = p.Cps;
    ps.K_dc   = tb.l[2];
    ps.Cp_dc  = tb.l[0];
}

//======================================================================================
//  Sub-stage timestep ladder.
//
//  Reproduces the original's dt switching exactly:
//     stage 1                       -> fdt  (1e-3)
//     stage 2, dp <  30*(2*Rs_0)    -> sidt (1e-10), i.e. keep the current dt
//     stage 2, dp >= 30*(2*Rs_0)    -> smdt (1e-9)   [latched once]
//     stage 2, dp >  300*(2*Rs_0)   -> sldt (1e-6)   [latched once]
//     stage 3                       -> sldt (1e-6)
//
//  The original uses an if / else-if chain with one-shot latches (n_sec, n_th), so the
//  30x branch fires first and the 300x branch can only fire on a LATER step.  That
//  ordering is preserved here (lat_mid / lat_late), because it changes the trajectory.
//
//  NOTE: this helper MUTATES the latches, exactly as the original loop body did.  Call
//  it once per step, immediately before dryingSubStep.
//======================================================================================
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
double recommendedDryingDt(DryingParcelState& s, const DryingProps& p)
{
    if (s.stage == 1) { s.dt_cur = p.fdt;  return s.dt_cur; }
    if (s.stage == 3) { s.dt_cur = p.sldt; return s.dt_cur; }

    const double dp   = s.R_out - s.R_int;
    const double t30  =  30.0*(2.0*p.Rs_0);
    const double t300 = 300.0*(2.0*p.Rs_0);

    if (dp < t30) {
        // keep current dt (sidt right after crust onset)
    } else if (dp >= t30 && s.lat_mid == 0) {
        s.lat_mid = s.nsub;
        s.dt_cur  = p.smdt;
    } else if (dp > t300 && s.lat_late == 0) {
        s.lat_late = s.nsub;
        s.dt_cur   = p.sldt;
    }
    return s.dt_cur;
}

//======================================================================================
//  Initialisation
//======================================================================================
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void dryingInit(DryingParcelState& s, const DryingProps& p)
{
    const double rho_d_0 = ((1.0 + p.X0)*p.rho_s*p.rho_l)/(p.rho_l + (p.X0*p.rho_s));
    const double Mtot_0  = 4.0*SD_PI*p.Rp_0*p.Rp_0*p.Rp_0*rho_d_0/3.0;
    const double C0      = ((Mtot_0*p.X0)/(p.X0 + 1.0))
                           /(4.0*SD_PI*p.Rp_0*p.Rp_0*p.Rp_0/3.0);
    const double Ws_0    = 1.0/(p.X0 + 1.0);

    s.nr    = (p.numr    < NR_MAX)    ? p.numr    : NR_MAX;
    s.nr_cr = (p.numr_cr < NR_CR_MAX) ? p.numr_cr : NR_CR_MAX;
    s.nr_wc = (p.numr_wc < NR_MAX)    ? p.numr_wc : NR_MAX;

    for (int i = 0; i < NR_MAX; ++i)    { s.T[i]    = p.Td_0; s.C[i] = C0; }
    for (int i = 0; i < NR_CR_MAX; ++i) { s.T_cr[i] = p.Td_0; }

    s.R_out       = p.Rp_0;
    s.R_int       = p.Rp_0;
    s.crust_thick = 0.0;
    s.mass        = Mtot_0;
    s.mass0       = Mtot_0;
    s.m_solid     = Ws_0*Mtot_0;
    s.mdot        = 1.0;          // original initialises dmdt = 1 before the first step
    s.t           = 0.0;
    s.dt_cur      = p.fdt;
    s.D_vap       = p.Dab*std::pow(2.0*p.Td_0, 1.75);

    s.k_eff = 0.0; s.cp_eff = 0.0; s.rho_eff = 0.0;
    s.vol_frac_l = 1.0; s.mfrac_s = Ws_0; s.d_sq = (2.0*p.Rp_0)*(2.0*p.Rp_0);

    s.stage = 1;
    s.nstep = 1;
    s.nsub  = 1;
    s.lat_mid = 0;
    s.lat_late = 0;
}

//======================================================================================
//  Stage 1: shrinking droplet with internal T and C fields
//======================================================================================
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void stageOneStep(DryingParcelState& s, const DryingProps& p, const DryingGasState& gas,
                  double dt, double& mdot_out)
{
    const int    nr  = s.nr;
    const double Rp  = s.R_out;
    const double d   = 2.0*Rp;
    const double tt  = s.t;
    const double M_s = s.m_solid;

    // ---- surface mass fractions -----------------------------------------------------
    const double Cs_surf = solidConc(p, s.C[nr-1]);
    const double W_l_surf = 1.0 - solidMassFrac(p, Cs_surf);

    // ---- volume-averaged temperature proxy (centre, mid, surface) --------------------
    const int m = ((nr % 2) == 0) ? (nr/2) : ((nr+1)/2);
    const double Tavg = (s.T[0] + s.T[m-1] + s.T[nr-1])/3.0;

    // ---- vapour diffusivity (Eq. 19) ------------------------------------------------
    const double D = p.Dab*std::pow(s.T[nr-1] + gas.T_inf, 1.75);
    s.D_vap = D;

    ThermoBundle tb;
    thermophysical(p, gas, Tavg, gas.T_inf, d, D, tt, tb);

    MassBundle mb;
    massEqnProperties(p, gas, tb.l[3], Tavg, W_l_surf, s.mass, s.mass0, s.nstep, tt, mb);

    const double dmdt = massTrRate(mb, tb, d, D);
    mdot_out = dmdt;

    const double dRdt  = -dmdt/(p.rho_l*4.0*SD_PI*Rp*Rp);
    const double Rp_n  = Rp + dRdt*dt;
    const double Mtot_n = s.mass - dmdt*dt;

    // ---- first_stage(): interior update ---------------------------------------------
    const double X     = (s.mass - M_s)/M_s;
    const double cfrac = 1.0/(1.0 + X);
    const double rho_d = (1.0 + X)*p.rho_s*p.rho_l/(p.rho_l + X*p.rho_s);
    const double Cp_d  = tb.l[0]*(1.0 - cfrac) + cfrac*p.Cps;
    const double Nu_s  = tb.dim[3]/std::pow(1.0 + mb.Bm, 0.7);
    const double h     = Nu_s*tb.a[2]/d;
    const double delta = 1.0 - 6.0*M_s/(SD_PI*p.rho_s*d*d*d);
    const double K_d   = delta*tb.l[2] + (1.0 - delta)*p.Ks;

    s.k_eff = K_d; s.cp_eff = Cp_d; s.rho_eff = rho_d;

    // solid diffusivity (independent of r in the original)
    const double Ws = M_s/s.mass;
    const double f  = 0.5*(1.0 - std::tanh(20.0*SD_PI*(Ws - 0.35)));
    const double Dsolid = f*p.Dsl
                        + (1.0 - f)*std::exp(-(28.1 + 282.0*(1.0 - Ws))/(1.0 + 15.47*(1.0 - Ws)));
    const double alpha = K_d/(rho_d*Cp_d);

    double Cn[NR_MAX];
    double Tn[NR_MAX];

    for (int x = 1; x < nr-1; ++x) {
        const double rxm = linspaceAt(0.0, Rp, nr, x-1);
        const double rx  = linspaceAt(0.0, Rp, nr, x);
        const double rxp = linspaceAt(0.0, Rp, nr, x+1);
        const double del_r  = rxp - rx;
        const double del_r1 = rx  - rxm;

        const double d1 = (rxp*rxp*((s.C[x+1]-s.C[x])/del_r)
                         - rx *rx *((s.C[x]-s.C[x-1])/del_r1))/del_r;
        const double d2 = (rxp*rxp*((s.T[x+1]-s.T[x])/del_r)
                         - rx *rx *((s.T[x]-s.T[x-1])/del_r1))/del_r;

        Cn[x] = s.C[x] + (dt*Dsolid/(rx*rx))*d1;
        Tn[x] = s.T[x] + (dt*alpha /(rx*rx))*d2;
    }

    // centre
    Cn[0] = Cn[1];
    Tn[0] = Tn[1];

    // surface
    const double dr = linspaceAt(0.0, Rp, nr, nr-1) - linspaceAt(0.0, Rp, nr, nr-2);
    const double Ad = SD_PI*d*d;
    Cn[nr-1] = Cn[nr-2] + dRdt*(p.rho_l - Cn[nr-2])*dr/Dsolid;
    Tn[nr-1] = (1.0/(K_d + h*dr))*((h*dr*gas.T_inf) + (K_d*Tn[nr-2]) - (tb.l[3]*dmdt*dr/Ad));

    // ---- diagnostics (pre-update state, as the original prints) ----------------------
    s.mfrac_s = M_s/s.mass;
    // FOURTHIRDS_BUG: ORIGINAL (4/3)->1
    const double Vtot = SD_PI*Rp*Rp*Rp;
    s.vol_frac_l = ((s.mass - M_s)/p.rho_l)/Vtot;

    const double rho_n2 = ambientPressure(p, gas, tt)/((8315.0/28.014)*gas.T_inf);
    const double K_d2   = 8.0*rho_n2*D*std::log(1.0 + mb.Bm)/rho_d;
    s.d_sq = (2.0*p.Rp_0)*(2.0*p.Rp_0) - K_d2*tt;

    // ---- advance --------------------------------------------------------------------
    s.t     = tt + dt;
    s.nstep = s.nstep + 1;
    s.R_out = Rp_n;
    s.R_int = Rp_n;
    s.mass  = Mtot_n;
    s.mdot  = dmdt;
    for (int i = 0; i < nr; ++i) { s.C[i] = Cn[i]; s.T[i] = Tn[i]; }

    // ---- crust onset test (original: while (Cs_crit_shell > Cs[nr-1])) ---------------
    if (!(p.Cs_crit_shell > Cs_surf)) {
        // ------- transition to stage 2, using the JUST-UPDATED fields -----------------
        const double R_s = s.R_out;
        const double Ri  = R_s - p.ini_thick;

        const double r_last = linspaceAt(0.0, R_s, nr, nr-1);
        const double r_pen  = linspaceAt(0.0, R_s, nr, nr-2);
        const double Tcr    = s.T[nr-2]
                            + ((s.T[nr-1] - s.T[nr-2])/(r_last - r_pen))*(Ri - r_pen);

        double Twc_tmp[NR_MAX];
        for (int i = 0; i < s.nr_wc; ++i) Twc_tmp[i] = linspaceAt(s.T[0], Tcr, s.nr_wc, i);
        for (int i = 0; i < s.nr_cr; ++i) s.T_cr[i]  = linspaceAt(Tcr, s.T[nr-1], s.nr_cr, i);
        for (int i = 0; i < s.nr_wc; ++i) s.T[i]     = Twc_tmp[i];

        s.R_int       = Ri;
        s.crust_thick = p.ini_thick;
        s.stage       = 2;
        s.nsub        = 1;
        s.dt_cur      = p.sidt;
        s.lat_mid     = 0;
        s.lat_late    = 0;
    }
}

//======================================================================================
//  Stage 2: porous crust + receding wet core
//======================================================================================
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void stageTwoStep(DryingParcelState& s, const DryingProps& p, const DryingGasState& gas,
                  double dt, double& mdot_out)
{
    const int    k   = s.nr_cr;
    const int    kw  = s.nr_wc;
    const double R_s = s.R_out;
    const double Ri  = s.R_int;
    const double tt  = s.t;
    const double M_s = s.m_solid;

    s.crust_thick = R_s - Ri;

    // ---- vapour diffusivity in the pores, with Knudsen correction -------------------
    const double Pt = ambientPressure(p, gas, tt);
    double D = p.Dab*std::pow(s.T_cr[0] + gas.T_inf, 1.75);
    if (p.knudsen_diff == 1) {
        const double lambda = (1.0/std::sqrt(2.0))
                            * (1.38e-23*s.T_cr[k-1]/(SD_PI*p.dc*p.dc*Pt));
        const double Kn = lambda/p.d_pore;
        if (Kn > 1.0) {
            const double v_bar = std::sqrt(8.0*p.Rgas*s.T_cr[k-1]/(SD_PI*p.MW_l));
            const double Dk    = (p.d_pore/3.0)*v_bar;
            D = 1.0/((1.0/D) + (1.0/Dk));
        }
    }
    const double D_cr = D;
    s.D_vap = D;

    ThermoBundle tb;
    thermophysical(p, gas, s.T_cr[0], gas.T_inf, 2.0*R_s, D, tt, tb);

    ParticleBundle pp;
    particleProperties(p, gas, R_s, Ri, M_s, tb, s.T_cr[0], p.Cs_crit_shell,
                       s.mass, s.mass0, tt, pp);

    const double dmvdt = massTrRateSecondStage(p, gas, tb, pp, D_cr, R_s, Ri,
                                               s.T_cr[0], s.T_cr[k-1], s.mdot, tt);
    mdot_out = dmvdt;

    const double Mtot_n = s.mass - dmvdt*dt;
    const double dRidt  = -dmvdt/(p.eps*p.rho_l*4.0*SD_PI*Ri*Ri);
    const double Ri_n   = Ri + dt*dRidt;

    // ---- crust temperature ----------------------------------------------------------
    const double Bm    = 1.9e3*(gas.T_inf - s.T_cr[k-1])/tb.l[3];
    const double K_cr  = pp.K_cr;
    const double K_wc  = pp.K_wc;
    const double h_cr  = (tb.dim[3]/std::pow(1.0 + Bm, 0.7))*tb.a[2]/(2.0*R_s);
    const double alpha_cr = K_cr/(pp.rho_cr*pp.Cp_cr);

    s.k_eff = K_cr; s.cp_eff = pp.Cp_cr; s.rho_eff = pp.rho_cr;

    double Tcr_n[NR_CR_MAX];
    for (int x = 1; x < k-1; ++x) {
        const double rxm = linspaceAt(Ri, R_s, k, x-1);
        const double rx  = linspaceAt(Ri, R_s, k, x);
        const double rxp = linspaceAt(Ri, R_s, k, x+1);
        const double del_r  = rxp - rx;
        const double del_r1 = rx  - rxm;
        const double d2 = (rxp*rxp*((s.T_cr[x+1]-s.T_cr[x])/del_r)
                         - rx *rx *((s.T_cr[x]-s.T_cr[x-1])/del_r1))/del_r;
        Tcr_n[x] = s.T_cr[x] + (dt*alpha_cr/(rx*rx))*d2;
    }

    const double dr_cr = linspaceAt(Ri, R_s, k, k-1) - linspaceAt(Ri, R_s, k, k-2);
    const double dr_wc = linspaceAt(0.0, Ri, kw, 1) - linspaceAt(0.0, Ri, kw, 0);

    // outer surface: convection to the gas
    Tcr_n[k-1] = (h_cr*gas.T_inf*dr_cr + K_cr*Tcr_n[k-2])/(K_cr + h_cr*dr_cr);
    // inner surface: conduction match minus latent heat carried by the vapour flux
    const double Ai = SD_PI*(2.0*Ri)*(2.0*Ri);
    Tcr_n[0] = 1.0/(K_cr*dr_wc + K_wc*dr_cr)
               *(K_cr*dr_wc*Tcr_n[1] + K_wc*dr_cr*s.T[kw-2]
                 - tb.l[3]*dr_cr*dr_wc*dmvdt/Ai);

    // ---- wet-core temperature -------------------------------------------------------
    const double alpha_wc = K_wc/(pp.rho_wc*pp.Cp_wc);
    double Twc_n[NR_MAX];
    for (int x = 1; x < kw-1; ++x) {
        const double rxm = linspaceAt(0.0, Ri, kw, x-1);
        const double rx  = linspaceAt(0.0, Ri, kw, x);
        const double rxp = linspaceAt(0.0, Ri, kw, x+1);
        const double del_r  = rxp - rx;
        const double del_r1 = rx  - rxm;
        const double d2 = (rxp*rxp*((s.T[x+1]-s.T[x])/del_r)
                         - rx *rx *((s.T[x]-s.T[x-1])/del_r1))/del_r;
        Twc_n[x] = s.T[x] + (dt*alpha_wc/(rx*rx))*d2;
    }
    Twc_n[0]    = Twc_n[1];
    Twc_n[kw-1] = Tcr_n[0];

    // ---- diagnostics (pre-update state) ---------------------------------------------
    s.mfrac_s = M_s/(s.mass + pp.M_vap);
    // FOURTHIRDS_BUG: ORIGINAL (4/3)->1
    const double Vtot = SD_PI*R_s*R_s*R_s;
    s.vol_frac_l = (pp.m_wc_w/p.rho_l)/Vtot;

    // ---- advance --------------------------------------------------------------------
    s.t     = tt + dt;
    s.nstep = s.nstep + 1;
    s.nsub  = s.nsub + 1;
    s.mass  = Mtot_n;
    s.mdot  = dmvdt;
    for (int i = 0; i < k;  ++i) s.T_cr[i] = Tcr_n[i];
    for (int i = 0; i < kw; ++i) s.T[i]    = Twc_n[i];

    // ---- wet core exhausted -> stage 3 (grids freeze at the PRE-update Ri, as in the
    //      original where r_cr / r_wc are never rebuilt in stage 3) --------------------
    if (!(s.vol_frac_l > 0.0)) {
        s.stage  = 3;
        s.nsub   = 1;
        s.dt_cur = p.sldt;
        // R_int deliberately left at the value that built the last stage-2 grids
        s.R_int  = Ri;
    } else {
        s.R_int = Ri_n;
    }
    s.crust_thick = s.R_out - s.R_int;
}

//======================================================================================
//  Stage 3: sensible heating of the dry particle
//======================================================================================
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void stageThreeStep(DryingParcelState& s, const DryingProps& p, const DryingGasState& gas,
                    double dt, double& mdot_out)
{
    const int    k   = s.nr_cr;
    const int    kw  = s.nr_wc;
    const double R_s = s.R_out;
    const double Ri  = s.R_int;   // frozen
    const double tt  = s.t;

    mdot_out = 0.0;

    ThermoBundle tb;
    thermophysicalDry(p, gas, s.T_cr[k-1], gas.T_inf, 2.0*R_s, s.D_vap, tt, tb);

    ParticleBundleDry ps;
    particlePropertiesDry(p, R_s, Ri, s.m_solid, tb, ps);

    const double K_cr = ps.K_cr;
    const double K_dc = ps.K_dc;
    const double h_cr = tb.dim[3]*tb.a[2]/(2.0*R_s);   // no Spalding correction in stage 3
    const double alpha_cr = K_cr/(ps.rho_cr*ps.Cp_cr);

    s.k_eff = K_cr; s.cp_eff = ps.Cp_cr; s.rho_eff = ps.rho_cr;

    double Tcr_n[NR_CR_MAX];
    for (int x = 1; x < k-1; ++x) {
        const double rxm = linspaceAt(Ri, R_s, k, x-1);
        const double rx  = linspaceAt(Ri, R_s, k, x);
        const double rxp = linspaceAt(Ri, R_s, k, x+1);
        const double del_r  = rxp - rx;
        const double del_r1 = rx  - rxm;
        const double d2 = (rxp*rxp*((s.T_cr[x+1]-s.T_cr[x])/del_r)
                         - rx *rx *((s.T_cr[x]-s.T_cr[x-1])/del_r1))/del_r;
        Tcr_n[x] = s.T_cr[x] + (dt*alpha_cr/(rx*rx))*d2;
    }

    const double dr_cr = linspaceAt(Ri, R_s, k, k-1) - linspaceAt(Ri, R_s, k, k-2);
    const double dr_dc = linspaceAt(0.0, Ri, kw, 1) - linspaceAt(0.0, Ri, kw, 0);
    Tcr_n[k-1] = (h_cr*gas.T_inf*dr_cr + K_cr*Tcr_n[k-2])/(K_cr + h_cr*dr_cr);
    Tcr_n[0]   = 1.0/(K_cr*dr_dc + K_dc*dr_cr)
                 *(K_cr*dr_dc*Tcr_n[1] + K_dc*dr_cr*s.T[kw-2]);

    const double alpha_dc = K_dc/(ps.rho_dc*ps.Cp_dc);
    double Tdc_n[NR_MAX];
    for (int x = 1; x < kw-1; ++x) {
        const double rxm = linspaceAt(0.0, Ri, kw, x-1);
        const double rx  = linspaceAt(0.0, Ri, kw, x);
        const double rxp = linspaceAt(0.0, Ri, kw, x+1);
        const double del_r  = rxp - rx;
        const double del_r1 = rx  - rxm;
        const double d2 = (rxp*rxp*((s.T[x+1]-s.T[x])/del_r)
                         - rx *rx *((s.T[x]-s.T[x-1])/del_r1))/del_r;
        Tdc_n[x] = s.T[x] + (dt*alpha_dc/(rx*rx))*d2;
    }
    Tdc_n[0]    = Tdc_n[1];
    Tdc_n[kw-1] = Tcr_n[0];

    s.t     = tt + dt;
    s.nstep = s.nstep + 1;
    s.nsub  = s.nsub + 1;
    for (int i = 0; i < k;  ++i) s.T_cr[i] = Tcr_n[i];
    for (int i = 0; i < kw; ++i) s.T[i]    = Tdc_n[i];
}

//======================================================================================
//  SINGLE ENTRY POINT: advance one parcel by one timestep
//======================================================================================
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void dryingSubStep(DryingParcelState& s, const DryingProps& props,
                   const DryingGasState& gas, double dt, double& mdot_out)
{
    mdot_out = 0.0;
    if (s.stage == 1)      stageOneStep  (s, props, gas, dt, mdot_out);
    else if (s.stage == 2) stageTwoStep  (s, props, gas, dt, mdot_out);
    else                   stageThreeStep(s, props, gas, dt, mdot_out);
}

//--------------------------------------------------------------------------------------
//  Convenience accessors for the three reported temperatures
//--------------------------------------------------------------------------------------
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
double centreTemperature(const DryingParcelState& s)
{ return s.T[0]; }

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
double interfaceTemperature(const DryingParcelState& s)
{
    if (s.stage == 1) {
        const int m = ((s.nr % 2) == 0) ? (s.nr/2) : ((s.nr+1)/2);
        return s.T[m-1];
    }
    return s.T_cr[0];
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
double surfaceTemperature(const DryingParcelState& s)
{
    if (s.stage == 1) return s.T[s.nr-1];
    return s.T_cr[s.nr_cr-1];
}

//======================================================================================
//  Serialisation of DryingParcelState into a flat block of reals
//
//  Layout (NDRY_REAL entries):
//     [0                     , NR_MAX)                 T[]
//     [NR_MAX                , 2*NR_MAX)               C[]
//     [2*NR_MAX              , 2*NR_MAX+NR_CR_MAX)     T_cr[]
//     [2*NR_MAX+NR_CR_MAX    , +16)                    the 16 scalar doubles, in
//                                                      declaration order
//     [2*NR_MAX+NR_CR_MAX+16 , +8)                     the 8 ints, stored as reals
//======================================================================================
constexpr int NDRY_REAL = 2*NR_MAX + NR_CR_MAX + 16 + 8;

template <typename RT>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void dryingPack(const DryingParcelState& s, RT* dst)
{
    int k = 0;
    for (int i = 0; i < NR_MAX; ++i)    dst[k++] = static_cast<RT>(s.T[i]);
    for (int i = 0; i < NR_MAX; ++i)    dst[k++] = static_cast<RT>(s.C[i]);
    for (int i = 0; i < NR_CR_MAX; ++i) dst[k++] = static_cast<RT>(s.T_cr[i]);

    dst[k++] = static_cast<RT>(s.R_out);
    dst[k++] = static_cast<RT>(s.R_int);
    dst[k++] = static_cast<RT>(s.crust_thick);
    dst[k++] = static_cast<RT>(s.mass);
    dst[k++] = static_cast<RT>(s.mass0);
    dst[k++] = static_cast<RT>(s.m_solid);
    dst[k++] = static_cast<RT>(s.mdot);
    dst[k++] = static_cast<RT>(s.t);
    dst[k++] = static_cast<RT>(s.dt_cur);
    dst[k++] = static_cast<RT>(s.D_vap);
    dst[k++] = static_cast<RT>(s.k_eff);
    dst[k++] = static_cast<RT>(s.cp_eff);
    dst[k++] = static_cast<RT>(s.rho_eff);
    dst[k++] = static_cast<RT>(s.vol_frac_l);
    dst[k++] = static_cast<RT>(s.mfrac_s);
    dst[k++] = static_cast<RT>(s.d_sq);

    dst[k++] = static_cast<RT>(s.stage);
    dst[k++] = static_cast<RT>(s.nr);
    dst[k++] = static_cast<RT>(s.nr_cr);
    dst[k++] = static_cast<RT>(s.nr_wc);
    dst[k++] = static_cast<RT>(s.nstep);
    dst[k++] = static_cast<RT>(s.nsub);
    dst[k++] = static_cast<RT>(s.lat_mid);
    dst[k++] = static_cast<RT>(s.lat_late);
}

template <typename RT>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void dryingUnpack(const RT* src, DryingParcelState& s)
{
    int k = 0;
    for (int i = 0; i < NR_MAX; ++i)    s.T[i]    = static_cast<double>(src[k++]);
    for (int i = 0; i < NR_MAX; ++i)    s.C[i]    = static_cast<double>(src[k++]);
    for (int i = 0; i < NR_CR_MAX; ++i) s.T_cr[i] = static_cast<double>(src[k++]);

    s.R_out       = static_cast<double>(src[k++]);
    s.R_int       = static_cast<double>(src[k++]);
    s.crust_thick = static_cast<double>(src[k++]);
    s.mass        = static_cast<double>(src[k++]);
    s.mass0       = static_cast<double>(src[k++]);
    s.m_solid     = static_cast<double>(src[k++]);
    s.mdot        = static_cast<double>(src[k++]);
    s.t           = static_cast<double>(src[k++]);
    s.dt_cur      = static_cast<double>(src[k++]);
    s.D_vap       = static_cast<double>(src[k++]);
    s.k_eff       = static_cast<double>(src[k++]);
    s.cp_eff      = static_cast<double>(src[k++]);
    s.rho_eff     = static_cast<double>(src[k++]);
    s.vol_frac_l  = static_cast<double>(src[k++]);
    s.mfrac_s     = static_cast<double>(src[k++]);
    s.d_sq        = static_cast<double>(src[k++]);

    s.stage    = static_cast<int>(src[k++]);
    s.nr       = static_cast<int>(src[k++]);
    s.nr_cr    = static_cast<int>(src[k++]);
    s.nr_wc    = static_cast<int>(src[k++]);
    s.nstep    = static_cast<int>(src[k++]);
    s.nsub     = static_cast<int>(src[k++]);
    s.lat_mid  = static_cast<int>(src[k++]);
    s.lat_late = static_cast<int>(src[k++]);
}

//--------------------------------------------------------------------------------------
//  Saturation pressure of the solvent [Pa] -- SINGLE definition, shared by the kernel's
//  internal mass-transfer routines (massEqnProperties / particleProperties, which inline
//  the same Clausius-Clapeyron form) and by the PeleLMeX coupling layer, which needs it
//  to convert a gas-phase water mass fraction into a relative humidity.
//--------------------------------------------------------------------------------------
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
double saturationPressure(const DryingProps& p, double T, double Pt)
{
    const double hfg = (-0.0013*(T-273.15)*(T-273.15) - 2.29618*(T-273.15) + 2500.0)*1e3;
    return Pt*std::exp((-hfg/(p.Rgas/p.MW_l))*(1.0/T - 1.0/p.Tboil));
}

//--------------------------------------------------------------------------------------
//  Convective heat flux from the ambient gas into the parcel [W], evaluated with the
//  SAME air-property polynomials and the SAME Ranz-Marshall Nusselt correlation the
//  kernel uses internally (thermophysical(), air props at Tf = gas.T_inf).  Positive
//  when the gas is hotter than the parcel surface, i.e. energy leaving the gas.
//  d is the parcel outer diameter, T_surf the parcel surface temperature.
//--------------------------------------------------------------------------------------
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
double convectiveHeatFlux(const DryingProps& p, const DryingGasState& gas,
                          double d, double T_surf)
{
    const double Tf   = gas.T_inf;
    const double cp_a = 4e-4*(Tf-273.15)*(Tf-273.15) + 2.38e-2*(Tf-273.15) + 1004.5;
    const double mu_a = 1.097e-6*(std::sqrt(Tf)/(1.453 - 0.0243*std::sqrt(Tf)));
    const double k_a  = 1.97e-4*std::pow(Tf, 0.858);
    const double rho_a = gas.P/((8.314/p.MW_a)*Tf);

    const double Re = rho_a*d*gas.V_rel/mu_a;
    const double Pr = cp_a*mu_a/k_a;
    const double Nu = 2.0 + 0.6*std::sqrt(Re)*std::pow(Pr, 0.333);
    return SD_PI*d*k_a*Nu*(gas.T_inf - T_surf);
}

} // namespace SprayDrying

#endif // SPRAY_DRYING_H
