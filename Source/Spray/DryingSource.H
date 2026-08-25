#ifndef DRYINGSOURCE_H
#define DRYINGSOURCE_H

//======================================================================================
//  DryingSource.H
//
//  PeleLMeX/PelePhysics coupling layer for the three-stage slurry-droplet drying
//  kernel in SprayDrying.H (Bhattacharjee & Gnanaskandan, Phys. Fluids 36, 083354).
//
//  calculateDryingSource() is called from the top of calculateSpraySource() (Drag.H)
//  and completely REPLACES the standard PelePhysics evaporation path for the parcel:
//  the drying kernel is a closed model that owns the parcel's internal temperature and
//  solid-concentration profiles, its own vapour-diffusion / Darcy crust submodels and
//  its own Ranz-Marshall transfer correlations.  Mixing it with the Spalding-number
//  path in Drag.H would double count.
//
//  What this layer does, per parcel per flow step:
//    1. unpack SprayDrying::DryingParcelState from the parcel's opaque real block
//       (SprayComps::pstateDryState .. +NDRY_REAL); stage == 0 means "never visited",
//       so dryingInit() is called
//    2. build a SprayDrying::DryingGasState from the interpolated gas state
//    3. sub-cycle the kernel's own dt ladder until exactly flow_dt has been consumed
//    4. turn the parcel mass change into gas mass / species / energy / momentum sources
//       using the same sign conventions as the tail of calculateSpraySource()
//    5. write pstateT, pstateDia and the packed state back to the parcel
//======================================================================================

#include "SprayDrying.H"

// Hard cap on kernel substeps per flow step. The ladder is 1e-3 (stage 1) then
// 1e-9 for a single step at crust onset then 1e-6, so a 0.01 s flow step needs
// at most ~1e4 substeps. The cap only exists so a pathological gas state cannot
// hang the solver.
#define PELE_DRYING_MAX_SUBSTEP 2000000

//--------------------------------------------------------------------------------------
//  Relative humidity of the local gas, from the gas-phase mass fraction of the solvent
//  vapour.  Uses SprayDrying::saturationPressure(), the SAME Clausius-Clapeyron form the
//  kernel uses internally for Psat -- there is deliberately no second psat routine here.
//--------------------------------------------------------------------------------------
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::Real
dryingRelHumidity(
  const SprayDrying::DryingProps& dp,
  const amrex::Real Y_vap,
  const amrex::Real mw_mix, // [kg/mol]
  const amrex::Real p_gas,  // [Pa]
  const amrex::Real T_gas)  // [K]
{
  if (Y_vap <= 0. || p_gas <= 0.) {
    return 0.;
  }
  // mole fraction -> partial pressure
  const amrex::Real X_vap = Y_vap * mw_mix / dp.MW_l;
  const amrex::Real p_vap = amrex::min<amrex::Real>(X_vap, 1.) * p_gas;
  const amrex::Real p_sat = SprayDrying::saturationPressure(dp, T_gas, p_gas);
  if (p_sat <= 0.) {
    return 0.;
  }
  return amrex::min<amrex::Real>(1., amrex::max<amrex::Real>(0., p_vap / p_sat));
}

//--------------------------------------------------------------------------------------
//  Advance one parcel with the drying kernel and deposit the gas-phase sources.
//  Returns the parcel Reynolds number, matching what calculateSpraySource() returns
//  (the caller stores it in Reyn_d and only uses it for the breakup models).
//--------------------------------------------------------------------------------------
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::Real
calculateDryingSource(
  const amrex::Real flow_dt,
  GasPhaseVals& gpv,
  const SprayData& fdat,
  SprayParticleContainer::ParticleType& p,
  const amrex::Real num_ppp,
  const amrex::RealVect& vel_part)
{
  auto eos = pele::physics::PhysicsType::eos(fdat.eosparm);
  const SprayDrying::DryingProps& dp = fdat.dryprops;
  amrex::Real* pdry = &p.rdata(SprayComps::pstateDryState);

  // ---- 1. parcel state ---------------------------------------------------------
  SprayDrying::DryingParcelState s;
  SprayDrying::dryingUnpack(pdry, s);
  if (s.stage == 0) {
    SprayDrying::dryingInit(s, dp);
  }

  // ---- 2. gas state ------------------------------------------------------------
  const int vap_indx = fdat.dep_indx[0];
  amrex::RealVect diff_vel = gpv.vel_fluid - vel_part;
  const amrex::Real diff_vel_mag = diff_vel.vectorLength();

  SprayDrying::DryingGasState gas;
  gas.T_inf = gpv.T_fluid;
  gas.P = gpv.p_fluid;
  gas.RH = dryingRelHumidity(
    dp, gpv.Y_fluid[vap_indx], gpv.mw_mix, gpv.p_fluid, gpv.T_fluid);
  // The reference experiment blows gas past a suspended droplet at a fixed
  // speed, so the slip velocity is prescribed by default. A negative
  // particles.drying_V_rel switches to the live |u_gas - u_parcel|.
  gas.V_rel = (fdat.drying_V_rel >= 0.) ? fdat.drying_V_rel : diff_vel_mag;

  // ---- 3. sub-cycle exactly flow_dt --------------------------------------------
  const amrex::Real mass_old = s.mass;
  const amrex::Real m_solid_old = s.m_solid;
  const int stage_old = s.stage;
  amrex::Real t_done = 0.;
  amrex::Real mdot_sub = 0.;
  int nsubstep = 0;
  const amrex::Real t_eps = 1.e-12 * flow_dt;
  while (t_done < flow_dt - t_eps && nsubstep < PELE_DRYING_MAX_SUBSTEP) {
    // MUST be called exactly once immediately before each substep: it latches
    // the kernel's dt ladder.
    amrex::Real dt_rec = SprayDrying::recommendedDryingDt(s, dp);
    // Clamp the last substep so the accumulated time is exactly flow_dt.
    const amrex::Real dt_rem = flow_dt - t_done;
    const amrex::Real dt_use = (dt_rec > dt_rem) ? dt_rem : dt_rec;
    SprayDrying::dryingSubStep(s, dp, gas, dt_use, mdot_sub);
    t_done += dt_use;
    ++nsubstep;
  }
  amrex::ignore_unused(stage_old);

  // ---- 4. gas-phase sources ----------------------------------------------------
  // mass_new - mass_old is SOLVENT ONLY: the drying kernel never moves solid
  // mass, m_solid is set once in dryingInit() and is invariant thereafter.
  AMREX_ASSERT(s.m_solid == m_solid_old);
  amrex::ignore_unused(m_solid_old);

  const amrex::Real T_surf = SprayDrying::surfaceTemperature(s);
  const amrex::Real dia_part = 2. * s.R_out;

  // Negative for evaporation, matching the convention at the tail of
  // calculateSpraySource(). fdat.dtmod is 1 under PeleLMeX.
  const amrex::Real mdot_total =
    (s.mass - mass_old) / (fdat.dtmod * flow_dt);

  gpv.fluid_mass_src = num_ppp * mdot_total;
  gpv.fluid_Y_dot[vap_indx] = num_ppp * mdot_total;

  // Convective sensible heat drawn out of the gas, computed with the kernel's
  // own air-property polynomials and Nusselt correlation (positive => gas
  // loses energy, same sign as conv_src in calculateSpraySource()).
  const amrex::Real conv_src =
    SprayDrying::convectiveHeatFlux(dp, gas, dia_part, T_surf);
  gpv.fluid_eng_src = num_ppp * conv_src;

#ifndef USE_MANIFOLD_EOS
  // Enthalpy carried by the vapour leaving the parcel, evaluated at the parcel
  // surface temperature. mdot_total < 0, matching the convention used for
  // midot * h_gas at the tail of calculateSpraySource().
  amrex::Real h_gas[NUM_SPECIES];
  eos.T2Hi(T_surf, h_gas);
  gpv.fluid_eng_src +=
    num_ppp * mdot_total * h_gas[vap_indx] * SprayUnits::eng_conv;
#else
  gpv.fluid_eng_src = 0.0;
  amrex::ignore_unused(eos);
#endif
#ifndef PELELM_USE_SPRAY
  const amrex::Real part_ke = 0.5 * vel_part.radSquared();
  gpv.fluid_eng_src += num_ppp * part_ke * mdot_total;
#endif

  // ---- momentum ------------------------------------------------------------
  // Air properties at the free-stream temperature, consistent with the kernel.
  const amrex::Real Tf = gas.T_inf;
  const amrex::Real mu_a =
    1.097e-6 * (std::sqrt(Tf) / (1.453 - 0.0243 * std::sqrt(Tf)));
  const amrex::Real rho_a = gas.P / ((8.314 / dp.MW_a) * Tf);
  const amrex::Real Reyn = rho_a * diff_vel_mag * dia_part / mu_a;
  if (fdat.mom_trans) {
    amrex::Real drag_coef = 0.;
    if (Reyn > 1000.) {
      drag_coef = 0.424;
    } else if (Reyn > 1.) {
      drag_coef = 24. / Reyn * (1. + std::cbrt(Reyn * Reyn) / 6.);
    } else if (Reyn > 0.) {
      drag_coef = 24. / Reyn;
    }
    const amrex::Real drag_force = 0.125 * rho_a * drag_coef * M_PI *
                                   dia_part * dia_part * diff_vel_mag;
    gpv.fluid_mom_src += num_ppp * drag_force * diff_vel;
    gpv.fluid_mom_src += num_ppp * vel_part * mdot_total;
  }

  // ---- 5. write back -----------------------------------------------------------
  p.rdata(SprayComps::pstateT) = T_surf;
  p.rdata(SprayComps::pstateDia) = dia_part;
  SprayDrying::dryingPack(s, pdry);
  return Reyn;
}

#endif // DRYINGSOURCE_H
