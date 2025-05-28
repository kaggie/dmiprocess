import numpy as np
import warnings # For user warnings
from typing import Dict, Optional, List, Union, Tuple

class AbsoluteQuantifier:
    """
    Handles absolute quantification of metabolite concentrations from MRS data,
    typically using an internal water reference.
    """

    def __init__(self,
                 default_water_conc_tissue_mM: float = 35880.0, 
                 default_protons_water: int = 2):
        """
        Initializes the AbsoluteQuantifier.

        Args:
            default_water_conc_tissue_mM (float): Effective concentration of water 
                in the tissue water compartment (e.g., ~35.88M or 35880 mM for brain 
                tissue water). This value is used if tissue-specific water content 
                and fractions are not provided. Defaults to 35880.0.
            default_protons_water (int): Number of protons contributing to the water signal
                (typically 2 for H2O). Defaults to 2.
        """
        if not isinstance(default_water_conc_tissue_mM, (float, int)):
            raise TypeError("default_water_conc_tissue_mM must be a float.")
        if not isinstance(default_protons_water, int):
            raise TypeError("default_protons_water must be an integer.")
        if default_water_conc_tissue_mM <= 0:
            raise ValueError("default_water_conc_tissue_mM must be positive.")
        if default_protons_water <= 0:
            raise ValueError("default_protons_water must be positive.")
            
        self.default_water_conc_tissue_mM = float(default_water_conc_tissue_mM)
        self.default_protons_water = default_protons_water

    def calculate_concentrations(self,
                               metabolite_amplitudes: Dict[str, float],
                               water_amplitude: float,
                               proton_counts_metabolites: Dict[str, int],
                               te_ms: Optional[float],
                               tr_ms: Optional[float],
                               relaxation_times: Optional[Dict[str, Dict[str, float]]] = None,
                               tissue_fractions: Optional[Dict[str, float]] = None, 
                               water_conc_tissue_specific_fractions: Optional[Dict[str, float]] = None,
                               attenuation_factors_metabolites: Optional[Dict[str, float]] = None
                              ) -> Tuple[Dict[str, float], List[str]]:
        """
        Calculates absolute concentrations of metabolites using an internal water reference.

        Args:
            metabolite_amplitudes (Dict[str, float]): Metabolite signal amplitudes from LCM.
            water_amplitude (float): Signal amplitude of the internal water reference.
            proton_counts_metabolites (Dict[str, int]): Proton counts for each metabolite.
            te_ms (Optional[float]): Echo time in milliseconds. If None, relaxation correction is skipped.
            tr_ms (Optional[float]): Repetition time in milliseconds. If None, relaxation correction is skipped.
            relaxation_times (Optional[Dict[str, Dict[str, float]]]): T1/T2 times (ms) for
                water and metabolites, e.g., `{'water': {'T1_ms': v, 'T2_ms': v}, ...}`.
            tissue_fractions (Optional[Dict[str, float]]): Voxel tissue fractions 
                (e.g., {'gm': 0.7, 'wm': 0.2, 'csf': 0.1}). Sum should be ~1.0.
            water_conc_tissue_specific_fractions (Optional[Dict[str, float]]): Tissue-specific 
                water content (e.g., {'gm': 0.80, 'wm': 0.70, 'csf': 0.99}). Values are fractions (0-1).
            attenuation_factors_metabolites (Optional[Dict[str, float]]): Other known
                attenuation factors for metabolites.

        Returns:
            Tuple[Dict[str, float], List[str]]: 
                - Dictionary of metabolite names mapped to their absolute concentrations (mM).
                - List of warnings generated during the calculation.
        """
        intermediate_results: Dict[str, float] = {}
        final_concentrations: Dict[str, float] = {}
        warnings_list: List[str] = []

        if water_amplitude <= 1e-9: # Check for effectively zero or negative water amplitude
            warnings_list.append("Input water amplitude is zero or negative. Absolute quantification is not possible.")
            for metab_name in metabolite_amplitudes:
                final_concentrations[metab_name] = np.nan
            # intermediate_results['_water_signal_corrected'] = np.nan # Not needed if returning early
            # intermediate_results['_metabolite_compartment_fraction'] = 1.0 # Not needed
            return final_concentrations, warnings_list

        # --- Validate Tissue Fraction Inputs ---
        valid_tissue_fractions_provided = False
        if tissue_fractions is not None:
            if not isinstance(tissue_fractions, dict):
                warnings_list.append("`tissue_fractions` provided but not a dictionary. Tissue corrections will be skipped.")
            else:
                sum_frac = sum(f for f in tissue_fractions.values() if isinstance(f, (float, int)))
                if not np.isclose(sum_frac, 1.0, atol=0.05): # Allow 5% tolerance
                    warnings_list.append(f"Sum of provided tissue_fractions ({sum_frac:.3f}) is not close to 1.0.")
                if any(f < 0 or f > 1 for f in tissue_fractions.values() if isinstance(f, (float, int))):
                    warnings_list.append("Values in `tissue_fractions` should be between 0 and 1. Proceeding with provided values, but this is unusual.")
                valid_tissue_fractions_provided = True
        
        valid_water_conc_fractions_provided = False
        if water_conc_tissue_specific_fractions is not None:
            if not isinstance(water_conc_tissue_specific_fractions, dict):
                warnings_list.append("`water_conc_tissue_specific_fractions` provided but not a dictionary. Tissue corrections will be skipped.")
            else:
                if any(not (0 <= f <= 1) for f in water_conc_tissue_specific_fractions.values() if isinstance(f, (float, int))):
                    warnings_list.append("Values in `water_conc_tissue_specific_fractions` should be between 0 and 1. Proceeding with provided values, but this is unusual.")
                valid_water_conc_fractions_provided = True


        # --- Water Relaxation Correction ---
        corrected_water_amplitude_relaxation = water_amplitude
        att_h2o = 1.0
        can_correct_relaxation = True
        if te_ms is None or tr_ms is None or te_ms < 0 or tr_ms <= 1e-9 : 
            warnings_list.append("TE or TR is None or non-positive. Skipping all relaxation corrections.")
            can_correct_relaxation = False
        
        if can_correct_relaxation and relaxation_times:
            water_relax_key = next((key for key in ['water', 'H2O', 'h2o'] if key in relaxation_times), None)
            if water_relax_key and isinstance(relaxation_times.get(water_relax_key), dict):
                t1_h2o = relaxation_times[water_relax_key].get('T1_ms')
                t2_h2o = relaxation_times[water_relax_key].get('T2_ms')
                if t1_h2o is not None and t2_h2o is not None:
                    att_h2o = self._calculate_relaxation_attenuation(te_ms, tr_ms, t1_h2o, t2_h2o)
                    if att_h2o < 1e-6:
                        warnings_list.append(f"Water relaxation attenuation factor is near zero ({att_h2o:.2e}). Corrected water amplitude may be unreliable or NaN.")
                        corrected_water_amplitude_relaxation = np.nan 
                    else:
                        corrected_water_amplitude_relaxation /= att_h2o
                else:
                    warnings_list.append(f"Missing T1_ms or T2_ms for '{water_relax_key}'. No water relaxation correction applied.")
            else:
                 warnings_list.append(f"Relaxation times for water not found or invalid. No water relaxation correction applied.")
        elif can_correct_relaxation: 
             warnings_list.append("TE/TR provided, but no 'relaxation_times' data. No water relaxation correction applied.")

        # --- Tissue Correction for Water Signal ---
        corrected_water_amplitude_tissue_adj = corrected_water_amplitude_relaxation
        if valid_tissue_fractions_provided and valid_water_conc_fractions_provided and not np.isnan(corrected_water_amplitude_relaxation):
            effective_water_content_in_voxel = 0.0
            # Ensure all relevant keys are present in water_conc_tissue_specific_fractions
            missing_water_frac_keys = [t_type for t_type in tissue_fractions if isinstance(tissue_fractions.get(t_type), (float,int)) and tissue_fractions.get(t_type, 0.0) > 1e-6 and t_type not in water_conc_tissue_specific_fractions]
            if missing_water_frac_keys:
                warnings_list.append(f"Missing water content fractions for tissue types with non-zero volume: {', '.join(missing_water_frac_keys)}. Skipping tissue correction for water signal.")
            else:
                for t_type_key in tissue_fractions.keys(): 
                    fraction = tissue_fractions.get(t_type_key, 0.0)
                    water_content = water_conc_tissue_specific_fractions.get(t_type_key, 0.0) 
                    if not isinstance(fraction, (float, int)) or not isinstance(water_content, (float, int)):
                        warnings_list.append(f"Non-numeric value found for fraction or water content for tissue type '{t_type_key}'. Skipping this tissue type for water correction.")
                        continue
                    effective_water_content_in_voxel += fraction * water_content
                
                if effective_water_content_in_voxel < 1e-6:
                    warnings_list.append("Effective water content in voxel (sum of f_tissue * water_content_tissue) is near zero. Tissue-corrected water amplitude set to NaN.")
                    corrected_water_amplitude_tissue_adj = np.nan
                else:
                    corrected_water_amplitude_tissue_adj = corrected_water_amplitude_relaxation / effective_water_content_in_voxel
        elif valid_tissue_fractions_provided != valid_water_conc_fractions_provided and not np.isnan(corrected_water_amplitude_relaxation): 
             warnings_list.append("Both `tissue_fractions` and `water_conc_tissue_specific_fractions` must be valid dictionaries for water tissue correction. Skipping.")
        
        intermediate_results['_water_signal_corrected'] = corrected_water_amplitude_tissue_adj

        # --- Metabolite Compartment Fraction ---
        metabolite_compartment_fraction = 1.0 
        if valid_tissue_fractions_provided:
            gm_fraction = tissue_fractions.get('gm', 0.0)
            wm_fraction = tissue_fractions.get('wm', 0.0)
            if not isinstance(gm_fraction, (float, int)): gm_fraction = 0.0; warnings_list.append("GM fraction is not a number, using 0.")
            if not isinstance(wm_fraction, (float, int)): wm_fraction = 0.0; warnings_list.append("WM fraction is not a number, using 0.")
            
            metabolite_compartment_fraction = gm_fraction + wm_fraction
            if metabolite_compartment_fraction < 1e-6:
                warnings_list.append("Metabolite compartment fraction (GM+WM) is near zero. Concentrations may be unreliable or NaN.")
            elif metabolite_compartment_fraction > 1.00001 : 
                warnings_list.append(f"Metabolite compartment fraction (GM+WM = {metabolite_compartment_fraction:.3f}) exceeds 1.0. Using 1.0.")
                metabolite_compartment_fraction = 1.0
        else:
            warnings_list.append("No valid tissue fractions provided; metabolite concentrations will not be corrected for partial volume (assuming fraction of 1.0).")
        intermediate_results['_metabolite_compartment_fraction'] = metabolite_compartment_fraction


        # --- Metabolite Loop for Amplitudes and Relaxation/Attenuation Correction ---
        for metab_name, metab_amp in metabolite_amplitudes.items():
            if metab_amp is None or np.isnan(metab_amp) or metab_amp <= 1e-9: 
                intermediate_results[metab_name] = 0.0 
                if metab_amp is None or np.isnan(metab_amp):
                    warnings_list.append(f"Input amplitude for {metab_name} is None or NaN. Setting corrected amplitude to 0.0.")
                else: 
                    warnings_list.append(f"Input amplitude for {metab_name} is near zero or negative. Setting corrected amplitude to 0.0.")
                continue

            corrected_metab_amplitude = metab_amp
            
            if attenuation_factors_metabolites and metab_name in attenuation_factors_metabolites:
                att_metab_val = attenuation_factors_metabolites[metab_name]
                if att_metab_val < 1e-6: 
                    warnings_list.append(f"Provided attenuation factor for {metab_name} is near zero ({att_metab_val:.2e}). Corrected amplitude for {metab_name} set to NaN.")
                    corrected_metab_amplitude = np.nan
                else:
                    corrected_metab_amplitude /= att_metab_val
            
            elif can_correct_relaxation and relaxation_times and metab_name in relaxation_times and not np.isnan(corrected_metab_amplitude):
                if isinstance(relaxation_times.get(metab_name), dict):
                    t1_metab = relaxation_times[metab_name].get('T1_ms')
                    t2_metab = relaxation_times[metab_name].get('T2_ms')

                    if t1_metab is not None and t2_metab is not None:
                        att_metab_relax = self._calculate_relaxation_attenuation(te_ms, tr_ms, t1_metab, t2_metab)
                        if att_metab_relax < 1e-6:
                            warnings_list.append(f"Relaxation attenuation factor for {metab_name} is near zero ({att_metab_relax:.2e}). Corrected amplitude for {metab_name} set to NaN.")
                            corrected_metab_amplitude = np.nan
                        else:
                            corrected_metab_amplitude /= att_metab_relax
                    else:
                        warnings_list.append(f"Missing T1_ms or T2_ms for {metab_name}. No relaxation correction applied for this metabolite.")
                else:
                    warnings_list.append(f"Relaxation times for {metab_name} is not a dictionary. No relaxation correction applied.")
            
            elif can_correct_relaxation and not np.isnan(corrected_metab_amplitude): 
                warnings_list.append(f"No specific attenuation or relaxation times found for {metab_name}. Using uncorrected amplitude for relaxation effects.")

            intermediate_results[metab_name] = corrected_metab_amplitude
        
        # --- Calculate Final Absolute Concentrations ---
        water_signal_corrected = intermediate_results.get('_water_signal_corrected', np.nan)
        metabolite_compartment_fraction = intermediate_results.get('_metabolite_compartment_fraction', 1.0)

        if np.isnan(water_signal_corrected) or water_signal_corrected < 1e-9:
            warnings_list.append("Corrected water signal is zero, negative, or NaN. Cannot calculate absolute concentrations.")
            for metab_name in metabolite_amplitudes:
                final_concentrations[metab_name] = np.nan
            return final_concentrations, warnings_list

        for metab_name, initial_amplitude in metabolite_amplitudes.items():
            corrected_metabolite_amplitude = intermediate_results.get(metab_name, 0.0)
            
            # If initial amplitude was None/NaN or corrected is None/NaN/zero, concentration is 0 or NaN
            if corrected_metabolite_amplitude is None or np.isnan(corrected_metabolite_amplitude) or corrected_metabolite_amplitude < 1e-9 : # Effectively zero
                final_concentrations[metab_name] = 0.0 
                if corrected_metabolite_amplitude is None or np.isnan(corrected_metabolite_amplitude):
                     # Warning for this already added when setting intermediate_results[metab_name] to 0.0 or NaN
                     pass # warnings_list.append(f"Corrected amplitude for {metab_name} is NaN. Setting concentration to NaN.")
                # else: # Amplitude is near zero
                     # warnings_list.append(f"Corrected amplitude for {metab_name} is near zero. Setting concentration to 0.0.")
                continue

            protons_metabolite = proton_counts_metabolites.get(metab_name)
            if protons_metabolite is None or not isinstance(protons_metabolite, int) or protons_metabolite <= 0:
                warnings_list.append(f"Proton count for {metab_name} is missing, invalid, or zero ({protons_metabolite}). Cannot calculate concentration.")
                final_concentrations[metab_name] = np.nan
                continue

            # Core concentration calculation
            try:
                conc_met = (corrected_metabolite_amplitude / protons_metabolite) * \
                           (self.default_protons_water / water_signal_corrected) * \
                           self.default_water_conc_tissue_mM
            except ZeroDivisionError: # Should be caught by water_signal_corrected check earlier
                warnings_list.append(f"Division by zero encountered during concentration calculation for {metab_name} (unexpected). Setting to NaN.")
                final_concentrations[metab_name] = np.nan
                continue
            
            # Apply metabolite compartment fraction
            if np.isnan(metabolite_compartment_fraction) or metabolite_compartment_fraction < 1e-9:
                if not np.isnan(conc_met): # Only add warning if conc_met was valid before this check
                    warnings_list.append(f"Metabolite compartment fraction is zero or NaN. Concentration for {metab_name} set to NaN.")
                final_concentrations[metab_name] = np.nan
            elif not np.isclose(metabolite_compartment_fraction, 1.0, atol=1e-9): # Apply if not 1.0
                if not np.isnan(conc_met):
                    conc_met /= metabolite_compartment_fraction
            
            final_concentrations[metab_name] = conc_met

        return final_concentrations, warnings_list

    @staticmethod
    def _calculate_relaxation_attenuation(te_ms: float, tr_ms: float, t1_ms: float, t2_ms: float) -> float:
        """
        Calculates the signal attenuation factor due to T1 and T2 relaxation.
        
        Formula: (1 - exp(-TR/T1)) * exp(-TE/T2)

        Args:
            te_ms (float): Echo time in milliseconds. Must be >= 0.
            tr_ms (float): Repetition time in milliseconds. Must be > 0.
            t1_ms (float): T1 relaxation time in milliseconds. Must be > 0.
            t2_ms (float): T2 relaxation time in milliseconds. Must be > 0.

        Returns:
            float: The relaxation attenuation factor (between 0 and 1).
                   Returns 0.0 if T1, T2, or TR are effectively zero or negative,
                   or if TE is negative.
        """
        if t1_ms <= 1e-9 or t2_ms <= 1e-9 or tr_ms <= 1e-9 or te_ms < -1e-9: 
            # Using -1e-9 for TE to allow TE=0, as negative TE is physically impossible
            return 0.0 
        
        term_t1 = (1.0 - np.exp(-tr_ms / t1_ms))
        term_t2 = 1.0 if te_ms == 0 else np.exp(-te_ms / t2_ms) # Handle TE=0 explicitly
        
        # Pathological case: TE=0 and T2 is also effectively zero.
        # np.exp(-0/small_positive) = 1. np.exp(-0/0) is nan.
        # If T2 is zero, signal is gone, so attenuation should be total (factor = 0).
        if t2_ms <= 1e-9 and te_ms > 1e-9: # If T2 is zero and TE is positive, factor is zero
            term_t2 = 0.0

        return term_t1 * term_t2
```
