#include "DiagProbe.H"
#include <AMReX_VisMF.H>
#include <AMReX_FPC.H>
#include <AMReX_PlotFileUtil.H>
#include <regex>
#include <cstdio>

amrex::Real
LinearInterpolate(
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> xp,
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> x_low,
  amrex::Array3D<amrex::Real, 0, 1, 0, 1, 0, 1> cell_data,
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx)
{
  amrex::Real alpha, beta, gama;
  amrex::Real value = 0.0;

  alpha = 0.0;
  beta = 0.0;
  gama = 0.0;

  const int XDIR = 0;
  const int YDIR = 1;
  const int ZDIR = 2;

  alpha = (xp[XDIR] - x_low[XDIR]) / dx[XDIR];
  if (AMREX_SPACEDIM >= 2) {
    beta = (xp[YDIR] - x_low[YDIR]) / dx[YDIR];
  }
  if (AMREX_SPACEDIM == 3) {
    gama = (xp[ZDIR] - x_low[ZDIR]) / dx[ZDIR];
  }

  value += (1.0 - alpha) * (1 - beta) * (1 - gama) * cell_data(0, 0, 0);
  value += alpha * (1 - beta) * (1 - gama) * cell_data(0 + 1, 0, 0);
  value += (1.0 - alpha) * beta * (1 - gama) * cell_data(0, 0 + 1, 0);
  value += alpha * beta * (1 - gama) * cell_data(0 + 1, 0 + 1, 0);

  value += (1.0 - alpha) * (1 - beta) * gama * cell_data(0, 0, 0 + 1);
  value += alpha * (1 - beta) * gama * cell_data(0 + 1, 0, 0 + 1);
  value += (1.0 - alpha) * beta * gama * cell_data(0, 0 + 1, 0 + 1);
  value += alpha * beta * gama * cell_data(0 + 1, 0 + 1, 0 + 1);
  return (value);
}

void
DiagProbe::init(const std::string& a_prefix, std::string_view a_diagName)
{
  DiagBase::init(a_prefix, a_diagName);

  if (m_filters.empty()) {
    amrex::Print() << " Filters are not available on DiagFrameProbe and will "
                      "be discarded \n";
  }
  amrex::ParmParse pp(a_prefix);
  // Outputted variables
  int nOutFields = pp.countval("field_names");
  AMREX_ASSERT(nOutFields > 0);
  m_values_at_probe.resize(nOutFields);

  m_fieldNames.resize(nOutFields);
  m_fieldIndices_d.resize(nOutFields);
  for (int f{0}; f < nOutFields; ++f) {
    pp.get("field_names", m_fieldNames[f], f);
  }
  m_nfiles_probe =
    std::max(1, std::min(amrex::ParallelDescriptor::NProcs(), 256));
  pp.query("n_files", m_nfiles_probe);

  // Plane center

  amrex::Vector<amrex::Real> probe_loc;
  pp.getarr("probe_location", probe_loc, 0, pp.countval("probe_location"));
  if (probe_loc.size() >= AMREX_SPACEDIM) {
    for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
      m_probe_loc[idim] = probe_loc[idim];
    }
  } else {
    amrex::Abort("\nProvide probe location array with same dimension as "
                 "problem dimension");
  }

  // Interpolation
  std::string intType = "CellCenter";
  pp.query("interpolation", intType);
  if (intType == "Linear") {
    m_interpType = Linear;
  } else if (intType == "CellCenter") {
    m_interpType = CellCenter;
  } else {
    amrex::Abort(
      "Unknown interpolation type for " + a_prefix +
      ". Allowed values are Linear or CellCenter.\n");
  }

  // Set output file properties
  amrex::UtilCreateDirectory("temporals", 0755);
  std::string tmpProbeFileName =
    "temporals/Probe_" + std::string(a_diagName) + ".out";
  if (amrex::ParallelDescriptor::IOProcessor()) {
    tmpProbeFile.open(
      tmpProbeFileName.c_str(),
      std::ios::out | std::ios::app | std::ios_base::binary);
    tmpProbeFile.precision(12);
  }
}

void
DiagProbe::addVars(amrex::Vector<std::string>& a_varList)
{
  DiagBase::addVars(a_varList);
  for (const auto& v : m_fieldNames) {
    a_varList.push_back(v);
  }
}

void
DiagProbe::prepare(
  int a_nlevels,
  const amrex::Vector<amrex::Geometry>& a_geoms,
  const amrex::Vector<amrex::BoxArray>& a_grids,
  const amrex::Vector<amrex::DistributionMapping>& a_dmap,
  const amrex::Vector<std::string>& a_varNames)
{
  // Check if probe lies within the geometry
  if (!(a_geoms[0].insideRoundoffDomain(
        AMREX_D_DECL(m_probe_loc[0], m_probe_loc[1], m_probe_loc[2])))) {
    amrex::Abort("\nProbe " + m_diagfile + " lies outside problem domain");
  }

  if (first_time) {
    int nOutFields = static_cast<int>(m_fieldIndices_d.size());
    tmpProbeFile << "time,iter";
    for (int f{0}; f < nOutFields; ++f) {
      m_fieldIndices_d[f] = getFieldIndex(m_fieldNames[f], a_varNames);
      tmpProbeFile << "," << m_fieldNames[f];
    }
    tmpProbeFile << "\n";
    first_time = false;
  }

  // Search for highest lev and box where probe is located.
  bool probe_found = false;

  for (int lev = a_nlevels - 1; lev >= 0; lev--) {
    const amrex::Real* dx = a_geoms[lev].CellSize();
    const amrex::Real* problo = a_geoms[lev].ProbLo();

    amrex::Real dist[AMREX_SPACEDIM];

    for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
      dist[idim] = (m_probe_loc[idim] - (problo[idim])) / dx[idim];
    }

    // index of the cell where probe is located.
    amrex::IntVect idx_lev(AMREX_D_DECL(
      static_cast<int>(dist[0]), static_cast<int>(dist[1]),
      static_cast<int>(dist[2])));

    // loop through all boxes in lev to find which box the cell is located.
    for (int i = 0; i < a_grids[lev].size(); i++) {
      auto cBox = a_grids[lev][i];
      if (cBox.contains(idx_lev) && !probe_found) {
        // box found. store the lev, box number and box reference. set
        // probe_found to true to stop searching any further
        m_finest_level_probe = lev;
        m_box_probe_num = i;
        for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
          dx_finest_lev_probe[idim] = dx[idim];
          m_probe_idx[idim] = idx_lev[idim];
          x_low_cell[idim] =
            problo[idim] + dx[idim] * 0.5 + m_probe_idx[idim] * dx[idim];
        }
        probe_found = true;
      }
    }
  }

  // What if the probe is in an EB?
  if (!probe_found) {
    amrex::Abort(
      "\nUnable to find the probe location. There seems to be something wrong");
  }

  // We are storing only the single box which contains the probe
  m_probebox.resize(1);
  m_probeboxDM.resize(1);
  m_dmConvert.resize(1);
  amrex::Vector<int> pmap;

  amrex::BoxList bl(a_grids[m_finest_level_probe].ixType());
  bl.reserve(1);
  amrex::Vector<int> dmConvertLev;
  auto cBox = a_grids[m_finest_level_probe][m_box_probe_num];
  bl.push_back(cBox);
  pmap.push_back(a_dmap[m_finest_level_probe][m_box_probe_num]);
  dmConvertLev.push_back(m_box_probe_num);
  m_dmConvert[0] = dmConvertLev;
  m_probebox[0].define(bl);
  m_probeboxDM[0].define(pmap);
}

void
DiagProbe::processDiag(
  int a_nstep,
  const amrex::Real& a_time,
  const amrex::Vector<const amrex::MultiFab*>& a_state,
  const amrex::Vector<std::string>& a_varNames)
{
  // since we are only taking the single box, just create a single multifab
  amrex::Vector<amrex::MultiFab> planeData(1);
  planeData[0].define(
    m_probebox[0], m_probeboxDM[0], static_cast<int>(m_fieldNames.size()), 0);

  // Is there a way to isolate the state array given a box? I am not sure about
  // this. So I am iterating using an MFI (useless operation) to find the box
  for (amrex::MFIter mfi(planeData[0], amrex::TilingIfNotGPU()); mfi.isValid();
       ++mfi) {
    const auto& bx = mfi.tilebox();
    const int state_idx = m_dmConvert[0][mfi.index()];
    auto const& state =
      a_state[m_finest_level_probe]->const_array(state_idx, 0);
    auto* idx_d_p = m_fieldIndices_d.dataPtr();
    for (int n{0}; n < m_fieldIndices_d.size(); n++) {
      int stIdx = idx_d_p[n];
      if (m_interpType == Linear) {

#if (AMREX_SPACEDIM == 1)
        {
          cell_data(0, 0, 0) = state(m_probe_idx[0], 0, 0, stIdx);
          cell_data(1, 0, 0) = state(m_probe_idx[0] + 1, 0, 0, stIdx);
        }
#elif (AMREX_SPACEDIM == 2)
        {
          cell_data(0, 0, 0) = state(m_probe_idx[0], m_probe_idx[1], 0, stIdx);
          cell_data(1, 0, 0) =
            state(m_probe_idx[0] + 1, m_probe_idx[1], 0, stIdx);
          cell_data(0, 1, 0) =
            state(m_probe_idx[0], m_probe_idx[1] + 1, 0, stIdx);
          cell_data(1, 1, 0) =
            state(m_probe_idx[0] + 1, m_probe_idx[1] + 1, 0, stIdx);
        }
#else
        {
          cell_data(0, 0, 0) =
            state(m_probe_idx[0], m_probe_idx[1], m_probe_idx[2], stIdx);
          cell_data(1, 0, 0) =
            state(m_probe_idx[0] + 1, m_probe_idx[1], m_probe_idx[2], stIdx);
          cell_data(0, 1, 0) =
            state(m_probe_idx[0], m_probe_idx[1] + 1, m_probe_idx[2], stIdx);
          cell_data(1, 1, 0) = state(
            m_probe_idx[0] + 1, m_probe_idx[1] + 1, m_probe_idx[2], stIdx);

          cell_data(0, 0, 1) =
            state(m_probe_idx[0], m_probe_idx[1], m_probe_idx[2] + 1, stIdx);
          cell_data(1, 0, 1) = state(
            m_probe_idx[0] + 1, m_probe_idx[1], m_probe_idx[2] + 1, stIdx);
          cell_data(0, 1, 1) = state(
            m_probe_idx[0], m_probe_idx[1] + 1, m_probe_idx[2] + 1, stIdx);
          cell_data(1, 1, 1) = state(
            m_probe_idx[0] + 1, m_probe_idx[1] + 1, m_probe_idx[2] + 1, stIdx);
        }
#endif
        amrex::Real interpolatedval = LinearInterpolate(
          m_probe_loc, x_low_cell, cell_data, dx_finest_lev_probe);
        m_values_at_probe[n] = interpolatedval;
      } else if (m_interpType == CellCenter) {
#if (AMREX_SPACEDIM == 1)
        m_values_at_probe[n] = state(m_probe_idx[0], 0, 0, stIdx);
#elif (AMREX_SPACEDIM == 2)
        m_values_at_probe[n] = state(m_probe_idx[0], m_probe_idx[1], 0, stIdx);
#else
        m_values_at_probe[n] =
          state(m_probe_idx[0], m_probe_idx[1], m_probe_idx[2], stIdx);
#endif
      }
    }
  }

  amrex::ParallelDescriptor::ReduceRealSum(
    m_values_at_probe.data(), static_cast<int>(m_values_at_probe.size()));

  // Write probe values to file
  if (amrex::ParallelDescriptor::IOProcessor()) {
    tmpProbeFile << a_time << "," << a_nstep;
    for (int f{0}; f < m_values_at_probe.size(); ++f) {

      tmpProbeFile << "," << m_values_at_probe[f];
    }
    tmpProbeFile << "\n";
    tmpProbeFile.flush();
  }
}
