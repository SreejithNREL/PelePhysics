#ifndef DIAGPROBE_H
#define DIAGPROBE_H

#include "DiagBase.H"
#include <AMReX_VisMF.H>

class DiagProbe : public DiagBase::Register<DiagProbe>
{
public:
  static std::string identifier() { return "DiagProbe"; }

  enum InterpType { CellCenter, Linear};

  void init(const std::string& a_prefix, std::string_view a_diagName) override;

  void prepare(
    int a_nlevels,
    const amrex::Vector<amrex::Geometry>& a_geoms,
    const amrex::Vector<amrex::BoxArray>& a_grids,
    const amrex::Vector<amrex::DistributionMapping>& a_dmap,
    const amrex::Vector<std::string>& a_varNames) override;

  void processDiag(
    int a_nstep,
    const amrex::Real& a_time,
    const amrex::Vector<const amrex::MultiFab*>& a_state,
    const amrex::Vector<std::string>& a_varNames) override;

  void addVars(amrex::Vector<std::string>& a_varList) override;
  void close() override {}

private:
  // Variables output
  amrex::Vector<std::string> m_fieldNames;
  amrex::Gpu::DeviceVector<int> m_fieldIndices_d;
  int m_nfiles_probe;
  int m_finest_level_probe;
  int m_box_probe_num;
  amrex::Box m_box_probe;
  int m_probe_idx[AMREX_SPACEDIM];

  // Probe definition
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_probe_loc;
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_finest_lev_probe;
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> x_low_cell;

  // Interpolation data
  InterpType m_interpType;
  amrex::Array3D<amrex::Real, 0, 1, 0, 1, 0, 1> cell_data{0.0};
  amrex::Vector<amrex::Real> m_values_at_probe;

  //Probe file name
  std::ofstream tmpProbeFile;
  std::string tmpProbeFileName;

  // 2D-plane boxArray vector
  amrex::Geometry m_geomLev0;
  amrex::Vector<amrex::BoxArray> m_probebox;
  amrex::Vector<amrex::Vector<int>> m_dmConvert;
  amrex::Vector<amrex::DistributionMapping> m_probeboxDM;
};

#endif
