"""Spectrum, beam-hardening table and the tables of LEAP's scatter model (built on ``xrayphysics``).

Units follow ``xrayphysics.use_mm()``: lengths in mm, densities in g/mm^3, cross sections in mm^2/g.
"""
import numpy as np


def parse_pairs(text):
    """``'Sn:1.0,Cu:0.5'`` -> ``[('Sn', 1.0), ('Cu', 0.5)]``."""
    out = []
    for item in str(text).split(','):
        item = item.strip()
        if item:
            mat, mm = item.split(':')
            out.append((mat.strip(), float(mm)))
    return out


class Physics:
    """Source spectrum, detector response, beam-hardening table and LEAP scatter tables for one material.

    Args:
        material: Object material (an ``xrayphysics`` material name or chemical formula).
        density_g_mm3: Mass density of the object (g/mm^3).
        kvp: Tube voltage (kV).
        take_off_deg: Anode take-off angle (degrees).
        filters: Beam filtration as ``'Material:mm,...'`` (``''`` for none).
        detector: Scintillator as ``'Material:mm'``.
        bh_thickness_mm: Object thickness at which the beam-hardening table is centred.
        num_scatter_energies: Energy samples handed to LEAP's scatter model (at most 20).
    """

    def __init__(self, material, density_g_mm3, kvp, take_off_deg=11.0, filters='', detector='CsI:0.6', bh_thickness_mm=40.0,
                 num_scatter_energies=20, verbose=1):
        from xrayphysics import xrayPhysics
        self.xp = xrayPhysics()
        self.xp.use_mm()
        self.material, self.density = material, float(density_g_mm3)
        energies = np.arange(1, int(kvp) + 1, dtype=np.float32)
        self.energies, src = self.xp.simulateSpectra(float(kvp), float(take_off_deg), gammas=energies)
        spectrum = np.array(src, dtype=np.float32)
        for mat, mm in parse_pairs(filters):
            spectrum = spectrum * self.xp.filterResponse(mat, self.xp.massDensity(mat), mm, self.energies)
        self.spectrum_filtered = spectrum
        det_mat, det_mm = parse_pairs(detector)[0]
        self.detector_material, self.detector_thickness_mm = det_mat, det_mm
        self.spectrum_detected = spectrum * self.xp.detectorResponse(det_mat, self.xp.massDensity(det_mat), det_mm, self.energies)
        self.effective_energy_keV = float(self.xp.effectiveEnergy(material, self.density, float(bh_thickness_mm) / 2, self.spectrum_detected, self.energies))
        self.bh_lut, self.bh_t = self.xp.setBHClookupTable(self.spectrum_detected, self.energies, material, self.effective_energy_keV)
        self.mu_expected_per_mm = float(self.xp.sigma(material, self.effective_energy_keV) * self.density)
        self.scatter_tables = leap_scatter_tables(self.xp, self.energies, self.spectrum_filtered, material, det_mat, det_mm,
                                                  num_samples=num_scatter_energies, verbose=verbose)
        if verbose:
            print(f'[physics] {material} {self.density:.4e} g/mm^3, {kvp:g} kVp, filters {filters or "none"}, detector {detector}; '
                  f'effective energy {self.effective_energy_keV:.1f} keV; attenuation of the solid after beam hardening {self.mu_expected_per_mm:.4f} /mm')

    def apply_beam_hardening(self, leap_model, attenuation_vrc):
        """Linearise an attenuation stack in place with LEAP's ``applyTransferFunction`` and this table."""
        leap_model.applyTransferFunction(attenuation_vrc, self.bh_lut, self.bh_t)

    def primary(self, mass_thickness):
        """Polychromatic primary transmission of the object for a mass-thickness image or stack (g/mm^2)."""
        s_dn, e_dn, det, sigma, _ = self.scatter_tables
        return polychromatic_primary(mass_thickness, s_dn, e_dn, det, sigma)


def leap_scatter_tables(xp, energies, source_filtered, material, detector_material, detector_thickness_mm, num_samples=20, verbose=1):
    """Arguments of ``leapct.scatter_model``.

    Returns:
        ``(source_dn, energies_dn, detector, sigma (3, E), scatter_dist (2, E, 181))``: the filtered spectrum at up to
        ``num_samples`` energies (normalised), and the detector response, mass cross sections (photoelectric, Compton,
        Rayleigh) and normalised angular scatter distributions in 1 keV bins from 1 keV to the last source sample.
    """
    energies = np.asarray(energies, dtype=np.float32).ravel()
    source = np.asarray(source_filtered, dtype=np.float32).ravel()
    n = max(2, min(int(num_samples), 20, energies.size))
    idx = np.unique(np.linspace(0, energies.size - 1, n).round().astype(int))
    e_dn = np.ascontiguousarray(energies[idx], dtype=np.float32)
    s_dn = np.ascontiguousarray(xp.resample(energies, source, e_dn), dtype=np.float32)
    xp.normalizeSpectrum(s_dn, e_dn)
    es = np.arange(1, int(np.ceil(e_dn[-1])) + 1, dtype=np.float32)
    det = np.ascontiguousarray(xp.detectorResponse(detector_material, xp.massDensity(detector_material), float(detector_thickness_mm), es), dtype=np.float32)
    sigma = np.zeros((3, es.size), dtype=np.float32)
    sigma[0] = xp.sigmaPE(material, es)
    sigma[1] = xp.sigmaCS(material, es)
    sigma[2] = xp.sigmaRS(material, es)
    thetas = np.arange(181, dtype=np.float32)
    dsigma = np.zeros((2, es.size, 181), dtype=np.float32)
    for k in range(es.size):
        dsigma[0, k] = xp.incoherentScatterDistribution(material, es[k], thetas, doNormalize=True)
        dsigma[1, k] = xp.coherentScatterDistribution(material, es[k], thetas, doNormalize=True)
    if verbose:
        print(f'[physics] LEAP scatter tables: {e_dn.size} source samples {e_dn[0]:.0f}..{e_dn[-1]:.0f} keV, {es.size} 1-keV bins')
    return s_dn, e_dn, det, np.ascontiguousarray(sigma), np.ascontiguousarray(dsigma)


def polychromatic_primary(mass_thickness, source_dn, energies_dn, detector, sigma):
    """Primary transmission of a single-material object from its mass thickness (g/mm^2), with LEAP's tables.

    ``sum_E s(E) D(E) exp(-sigma_tot(E) L) / sum_E s(E) D(E)`` with the 1-keV tables read at the source samples.
    """
    L = np.asarray(mass_thickness, dtype=np.float32)
    e = np.asarray(energies_dn, dtype=np.float64)
    bins = np.clip(np.round(e).astype(int) - 1, 0, np.asarray(detector).size - 1)
    w = np.asarray(source_dn, dtype=np.float64) * np.asarray(detector, dtype=np.float64)[bins]
    w /= w.sum()
    sig_tot = np.asarray(sigma, dtype=np.float64).sum(axis=0)[bins]
    out = np.zeros(L.shape, dtype=np.float64)
    for wk, sk in zip(w, sig_tot):
        out += wk * np.exp(-sk * L.astype(np.float64))
    return out.astype(np.float32)
