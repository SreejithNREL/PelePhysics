#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <iomanip>

#include "mechanism.H"
#include <PelePhysics.H>

namespace {

constexpr int NSTATE = 8; // 4 states at 108 bar, then the same 4 at 1 bar

struct State
{
  amrex::Real T;
  amrex::Real P_bar;
  amrex::Real Y_O2, Y_H2O, Y_CO2, Y_H2, Y_CH4, Y_CO;
  amrex::Real mu_pelec_ref; // Erroneous viscosity values
};

const State states[NSTATE] = {
  {563.740, 108.233, 0.626080, 0.095131, 0.050876, 0.006044, 0.215715,
   0.006154, 1.360e-3},
  {526.174, 108.193, 0.448088, 0.129061, 0.051088, 0.009840, 0.351892,
   0.010031, 2.400e-5},
  {582.354, 108.218, 0.674247, 0.086700, 0.050757, 0.005114, 0.178172,
   0.005010, 3.718e-5},
  {611.901, 108.817, 0.772383, 0.067981, 0.050652, 0.003025, 0.103076,
   0.002883, 3.400e-5},
  {563.740, 1.0, 0.626080, 0.095131, 0.050876, 0.006044, 0.215715, 0.006154,
   0.0},
  {526.174, 1.0, 0.448088, 0.129061, 0.051088, 0.009840, 0.351892, 0.010031,
   0.0},
  {582.354, 1.0, 0.674247, 0.086700, 0.050757, 0.005114, 0.178172, 0.005010,
   0.0},
  {611.901, 1.0, 0.772383, 0.067981, 0.050652, 0.003025, 0.103076, 0.002883,
   0.0}};

}

int
main(int argc, char* argv[])
{
  amrex::Initialize(argc, argv);
  {
    pele::physics::PeleParams<pele::physics::eos::EosParm<pele::physics::PhysicsType::eos_type>> eos_parms;
    pele::physics::PeleParams<pele::physics::transport::TransParm<pele::physics::PhysicsType::eos_type, pele::physics::PhysicsType::transport_type>> trans_parms;
    eos_parms.initialize();
    trans_parms.initialize();

    amrex::Vector<std::string> names;
    pele::physics::eos::speciesNames<pele::physics::PhysicsType::eos_type>(
      names);
    auto idx = [&names](const std::string& s) {
      for (int i = 0; i < NUM_SPECIES; ++i) {
        if (names[i] == s) {
          return i;
        }
      }
      amrex::Abort("species not found: " + s);
      return -1;
    };
    const int iO2 = idx("O2"), iH2O = idx("H2O"), iCO2 = idx("CO2");
    const int iH2 = idx("H2"), iCH4 = idx("CH4"), iCO = idx("CO");

    // Using one cell per state
    amrex::Box domain(amrex::IntVect(AMREX_D_DECL(0, 0, 0)), amrex::IntVect(AMREX_D_DECL(NSTATE - 1, 0, 0)));
    amrex::RealBox real_box({AMREX_D_DECL(0.0, 0.0, 0.0)}, {AMREX_D_DECL(1.0, 1.0, 1.0)});
    amrex::Array<int, AMREX_SPACEDIM> is_per{AMREX_D_DECL(1, 1, 1)};
    amrex::Geometry geom(domain, real_box, 0, is_per);
    amrex::BoxArray ba(domain);
    amrex::DistributionMapping dm{ba};

    amrex::MultiFab mass_frac(ba, dm, NUM_SPECIES, 0);
    amrex::MultiFab temperature(ba, dm, 1, 0);
    amrex::MultiFab density(ba, dm, 1, 0);

    for (amrex::MFIter mfi(mass_frac); mfi.isValid(); ++mfi) {
      auto const& Y_a = mass_frac.array(mfi);
      auto const& T_a = temperature.array(mfi);
      auto const& rho_a = density.array(mfi);
      amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
      {
          const State& s = states[i];
          for (int n = 0; n < NUM_SPECIES; ++n) {
            Y_a(i, j, k, n) = 0.0;
          }
          Y_a(i, j, k, iO2) = s.Y_O2;
          Y_a(i, j, k, iH2O) = s.Y_H2O;
          Y_a(i, j, k, iCO2) = s.Y_CO2;
          Y_a(i, j, k, iH2) = s.Y_H2;
          Y_a(i, j, k, iCH4) = s.Y_CH4;
          Y_a(i, j, k, iCO) = s.Y_CO;
          T_a(i, j, k) = s.T;

          amrex::Real Yl[NUM_SPECIES];
          for (int n = 0; n < NUM_SPECIES; ++n) {
            Yl[n] = Y_a(i, j, k, n);
          }
          auto eos = pele::physics::PhysicsType::eos();
          amrex::Real rho = 0.0;
          eos.PYT2R(s.P_bar * 1.0e6, Yl, s.T, rho);
          rho_a(i, j, k) = rho;
        });
    }

    amrex::MultiFab D(ba, dm, NUM_SPECIES, 0);
    amrex::MultiFab	chi(ba, dm, NUM_SPECIES, 0);
    amrex::MultiFab mu(ba, dm, 1, 0);
    amrex::MultiFab xi(ba, dm, 1, 0);
    amrex::MultiFab lam(ba, dm, 1, 0);

    auto const* ltransparm = trans_parms.device_parm();
    for (amrex::MFIter mfi(mass_frac); mfi.isValid(); ++mfi) {
      const amrex::Box& gbox = mfi.tilebox();
      auto const& Y_a = mass_frac.array(mfi);
      auto const& T_a = temperature.array(mfi);
      auto const& rho_a = density.array(mfi);
      auto const& D_a = D.array(mfi);
      auto const& mu_a = mu.array(mfi);
      auto const& xi_a = xi.array(mfi);
      auto const& lam_a = lam.array(mfi);
      auto const& chi_a = chi.array(mfi);
      amrex::launch(gbox, [=] AMREX_GPU_DEVICE(amrex::Box const& tbx) {
        auto trans = pele::physics::PhysicsType::transport();
        trans.get_transport_coeffs(tbx, Y_a, T_a, rho_a, D_a, chi_a, mu_a, xi_a,
                                   lam_a, ltransparm);
      });
    }

    // Print transport property values
    amrex::Print() << "\n"
                   << "========================================================="
                      "=====================\n"
                   << " PelePhysics SRK + Simple transport, mechanism RAMEC_17\n"
                   << "========================================================="
                      "=====================\n";
    amrex::Print() << std::setw(3) << "#" << std::setw(10) << "T[K]"
                   << std::setw(10) << "P[bar]" << std::setw(12) << "rho[g/cm3]"
                   << std::setw(13) << "mu[Pa s]" << std::setw(13)
                   << "lam[W/m/K]" << std::setw(13) << "mu_ref(xlsx)"
                   << std::setw(13) <<"\n";

    for (amrex::MFIter mfi(mu); mfi.isValid(); ++mfi) {
      auto const& mu_a = mu.array(mfi);
      auto const& lam_a = lam.array(mfi);
      auto const& rho_a = density.array(mfi);
      const amrex::Box& bx = mfi.tilebox();
      for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
        amrex::Print() << std::setw(3) << i + 1 << std::setw(10)
                       << states[i].T << std::setw(10) << states[i].P_bar
                       << std::setw(12) << std::scientific
                       << std::setprecision(4) << rho_a(i, 0, 0)
                       << std::setw(13) << mu_a(i, 0, 0) * 0.1
                       << std::setw(13)
                       << lam_a(i, 0, 0) * 1.0e-5
                       << std::setw(13) << states[i].mu_pelec_ref
                       << "\n";
      }
    }
    amrex::Print() << "\n";
    trans_parms.deallocate();
    eos_parms.deallocate();
  }
  amrex::Finalize();
  return 0;
}
