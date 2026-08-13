import awkward as ak
from pyutils.pyselect import Select
from pyutils.pymcutil import MC
from pyutils.pylogger import Logger
from pyutils.pycut import CutManager
from pyutils.pyvector import Vector
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import traceback

class Analyze:
    """Class to handle analysis functions
    """
    def __init__(self,  verbosity=1, sign="minus", cut_switch=[]):
        """Initialise the analysis handler
        Args:
            verbosity (int, optional): Level of output detail (0: critical errors only, 1: info, 2: debug, 3: deep debug)
        """
        # Verbosity
        self.verbosity = verbosity
        # Start logger
        self.logger = Logger(
            print_prefix="[Analyse]",
            verbosity=self.verbosity
        )
        # Initialise tools
        self.selector = Select(verbosity=self.verbosity)
        self.mcutil = MC(verbosity=self.verbosity)
        self.vector = Vector()
        # Analysis configuration
        self.logger.log(f"Initialised", "info")
        self.sign = sign
        self.switch = cut_switch
        
        # Load MLP model for scoring
        self.mlp_model = None
        self.mlp_trainer = None
        self.mlp_training_cuts = None
        self._load_mlp_model()

    def _load_mlp_model(self):
        """Load trained MLP model and trainer state for scoring."""
        try:
            from mlp import MLP, MLPTrainer
            import json
            
            self.logger.log("Starting MLP model load", "info")
            
            model_path = 'mlp_model.pth'
            norm_path = 'mlp_normalization.json'
            cuts_path = 'mlp_training_cuts.json'
            
            if not os.path.exists(model_path):
                self.logger.log(f"MLP model not found at {model_path}, skipping", "warning")
                return
            
            self.logger.log(f"Loading model from {model_path}", "info")
            
            # Load model architecture and weights
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.mlp_model = MLP(input_dim=3, hidden_dim=64, dropout_rate=0.2)
            self.mlp_model.load_state_dict(torch.load(model_path, map_location=device))
            self.mlp_model.eval()
            self.mlp_model.to(device)
            
            # Create trainer wrapper (needed for score() method)
            self.mlp_trainer = MLPTrainer(self.mlp_model, device=device)
            self.logger.log(f"MLP trainer created", "info")
            
            # Load normalization parameters
            if os.path.exists(norm_path):
                with open(norm_path, 'r') as f:
                    norm_params = json.load(f)
                self.mlp_trainer.d0_mean = norm_params['d0_mean']
                self.mlp_trainer.d0_std = norm_params['d0_std']
                self.mlp_trainer.rmax_mean = norm_params['rmax_mean']
                self.mlp_trainer.rmax_std = norm_params['rmax_std']
                self.mlp_trainer.costheta_mean = norm_params['costheta_mean']
                self.mlp_trainer.costheta_std = norm_params['costheta_std']
                self.logger.log(f"Loaded normalization params from {norm_path}", "info")
            else:
                self.logger.log(f"Normalization params not found at {norm_path}, using defaults", "warning")
            
            # Load training cuts configuration
            if os.path.exists(cuts_path):
                with open(cuts_path, 'r') as f:
                    cuts_config = json.load(f)
                self.mlp_training_cuts = cuts_config['cuts']
                self.logger.log(f"Loaded training cuts: {sum(self.mlp_training_cuts)}/{len(self.mlp_training_cuts)} active", "info")
            else:
                self.logger.log(f"Training cuts not found at {cuts_path}, will use all defined cuts during scoring", "warning")
            
            self.logger.log(f"Loaded MLP model from {model_path}", "info")
        except Exception as e:
            self.logger.log(f"Could not load MLP model: {e}", "error")
            self.logger.log(traceback.format_exc(), "error")
            self.mlp_model = None
            self.mlp_trainer = None
            self.mlp_training_cuts = None

    def has_trk_front_segment(self, trkfit, surface_name="TT_Front"):
        """Return a track-level boolean mask indicating whether each track
        has at least one segment intersecting the requested surface.

        Args:
            trkfit (ak.Array): The `trkfit` branch (events × tracks × segments)
            surface_name (str): Surface to check (default: "TT_Front")

        Returns:
            ak.Array: boolean array with shape (events, tracks)
        """
        seg_mask = self.selector.select_surface(trkfit, surface_name=surface_name)
        return ak.any(seg_mask, axis=-1)

    def define_cuts(self, data, cut_manager, skip_mlp=False):
        """Define analysis cuts in NEW ORDER (21 cuts, indices 0-20)

        Cuts in order:
        0. has_trk_front_seg - has a trk seg for trk front
        1. fit_status_is_good - fit status is good (NEW)
        2. is_reco_electron_or_positron - is electron or positron (generic)
        3. has_downstream - tracker downstream pz > 0
        4. charge_selection - charge matches analysis sign (NEW)
        5. or_trigger - is trigger
        6. no_upstream - is there an upstream track?
        7. alt_hypothesis_upstream_veto - alt hypothesis cut
        8. no_multi_trk_veto - Multitrack
        9. good_trkpid - PID
        10. pz_over_pt - pz/pt or tanDip
        11. has_st - NST
        12. no_opa - NOPA
        13. good_trkqual - trkqual
        14. has_hits - nhits
        15. within_t0err - t0err
        16. no_crv_veto - Asymmetric CRV veto: 0 < dt < 150 ns
        17. in_mom_range - 100-110 MeV/c
        18. within_t0_475 - 475-1650 ns
        19. within_t0_540 - 540-1650 ns
        20. within_t0_640 - 640-1650 ns
        21. signal_region - SR (103.34-104.74 MeV/c)

        Args:
            data (ak.Array): data to apply cuts to
            cut_manager: The CutManager instance to use
            skip_mlp (bool): If True, skip the MLP scoring block
        """

        
        selector = self.selector

        # Track segments cuts
        try:
    
    
            # Track segments level definition
            at_trk_front = self.selector.select_surface(data["trkfit"], surface_name="TT_Front") 
            at_trk_mid = self.selector.select_surface(data["trkfit"], surface_name="TT_Mid")
            at_trk_back = self.selector.select_surface(data["trkfit"], surface_name="TT_Back")
            in_trk = (at_trk_front | at_trk_mid | at_trk_back)

            # Helper definitions for later cuts
            # is_downstream: earliest time front segment is going downstream (pz > 0)
            pz_at_front = data['trkfit']["trksegs"]["mom"]["fCoordinates"]["fZ"][at_trk_front]
            times_at_front = data['trkfit']["trksegs"]["time"][at_trk_front]
            
            # For each track, find minimum time among segments at TT_Front
            min_time_per_track = ak.min(times_at_front, axis=-1, keepdims=True)
            min_time_mask = (times_at_front == min_time_per_track)
            # Use 0 as padding (won't trigger > 0 or < 0 checks) instead of -1000 (which is < 0)
            pz_of_earliest = ak.where(min_time_mask, pz_at_front, 0.0)
            is_downstream = ak.any(pz_of_earliest > 0, axis=-1)
            data["is_downstream"] = is_downstream
            
            # is_upstream: track has pz < 0 at TT_Front (upstream trajectory, matches C++ PZFront() <= 0)
            # Check if earliest segment at front has pz < 0
            is_upstream = ak.any(pz_of_earliest < 0, axis=-1)
            data["is_upstream"] = is_upstream

            # Get vector and momentum for later use
            vector = Vector()
            trkfit_ent = ak.mask(data['trkfit']["trksegs"], at_trk_front)
            mom_mag = vector.get_mag(trkfit_ent, 'mom')


            # ============================================================================
            # CUT 0: has_a_track (NEW - event-level, broadcasted to track level)
            # ============================================================================
            # Event must have at least one reconstructed track
            ntracks = ak.count(data["trk"]["trk.pdg"])  # count of tracks per event
            has_a_track_event = ntracks > 0
            # Broadcast to track level: same value for all tracks in an event
            dummy = ak.ones_like(data["trk"]["trk.pdg"], dtype=bool)
            has_a_track = ak.where(dummy, has_a_track_event, False)
            
            cut_manager.add_cut(
                name="has_a_track",
                description="Event has at least one reconstructed track (ntracks > 0)",
                mask=has_a_track,
                active=self.switch[0]
            )
            data["has_a_track"] = has_a_track

            # ============================================================================
            # CUT 1: is_good_track
            # ============================================================================
            # Initial track quality check: status >= 0 and goodfit != 0
            try:
                trk_status = data["trk"]["trk.status"]
                trk_goodfit = data["trk"]["trk.goodfit"]
                is_good_track = (trk_status >= 0) & (trk_goodfit != 0)
                cut_manager.add_cut(
                    name="is_good_track",
                    description="Track quality: status >= 0 and goodfit != 0",
                    mask=is_good_track,
                    active=self.switch[1]
                )
                data["is_good_track"] = is_good_track
            except Exception as e:
                self.logger.log(f"Error in is_good_track cut: {e}", "warning")
                is_good_track = ak.ones_like(has_a_track, dtype=bool)

            # ============================================================================
            # CUT 2: has_trk_front_seg
            # ============================================================================
            has_trk_front_seg = self.has_trk_front_segment(data['trkfit'], surface_name="TT_Front")
            data["has_trk_front_seg"] = has_trk_front_seg
            cut_manager.add_cut(
                name="has_trk_front_seg",
                description="Track has >=1 segment intersecting TT_Front",
                mask=has_trk_front_seg,
                active=self.switch[2]
            )

            # ============================================================================
            # CUT N/A: fit_status_is_good
            # ============================================================================
            # Check track fit quality via fitcon > threshold
            try:
                fit_con = data["trk"]["trk.fitcon"]
                fit_status_is_good = fit_con > 1.e-5  # threshold for good fit
                cut_manager.add_cut(
                    name="fit_status_is_good",
                    description="Fit status is good (trk.fitcon > 1e-5)",
                    mask=fit_status_is_good,
                    active=False #self.switch[2]
                )
                data["fit_status_is_good"] = fit_status_is_good
            except Exception as e:
                self.logger.log(f"Error in fit_status_is_good cut: {e}", "warning")
                fit_status_is_good = ak.ones_like(is_downstream, dtype=bool)

            # ============================================================================
            # CUT 3: is_reco_electron_or_positron (generic check for either)
            # ============================================================================
            # Check if track is reconstructed as electron (11) or positron (-11)
            is_electron = selector.is_electron(data["trk"])
            is_positron = selector.is_positron(data["trk"])
            is_reco_epm = is_electron | is_positron
            data["is_reco_epm"] = is_reco_epm
            cut_manager.add_cut(
                name="is_reco_electron_or_positron", 
                description="Tracks are assumed to be electrons or positrons (FitPDG = ±11)", 
                mask=is_reco_epm,
                active=self.switch[3]
            )

            # ============================================================================
            # CUT 4: has_downstream
            # ============================================================================
            cut_manager.add_cut(
                name="has_downstream",
                description="Downstream tracks (p_z > 0 through tracker)",
                mask=is_downstream,
                active=self.switch[4]
            )

            # ============================================================================
            # CUT 5: charge_selection
            # ============================================================================
            # Select based on analysis sign: minus = electrons (PDG=11), plus = positrons (PDG=-11)
            # PDG convention: electron=11 (positive), positron=-11 (negative)
            if (str(self.sign) == "minus"):
                charge_cut = data["trk"]["trk.pdg"] == 11
                charge_description = "PDG = 11 (electrons)"
            elif (str(self.sign) == "plus"):
                charge_cut = data["trk"]["trk.pdg"] == -11
                charge_description = "PDG = -11 (positrons)"
            else:
                charge_cut = ak.ones_like(is_downstream, dtype=bool)
                charge_description = "No charge selection"
            
            data["charge_selection"] = charge_cut
            cut_manager.add_cut(
                name="charge_selection",
                description=charge_description,
                mask=charge_cut,
                active=self.switch[5]
            )

            # ============================================================================
            # CUT 6: or_trigger
            # ============================================================================
            trig_cpr = selector.get_trigger(data["evt"], "trig_cpr_TrkDe_80m70p")
            trig_apr = selector.get_trigger(data["evt"], "trig_apr_TrkDe_80m70p")
            or_trigger = trig_cpr | trig_apr
            data["or_trigger"] = or_trigger
            cut_manager.add_cut( 
                name="or_trigger",
                description="OR trigger selection (CPR | APR)",
                mask=or_trigger,
                active=self.switch[6]
            )

            # ============================================================================
            # CUT 7: upstream_veto (timing-based)
            # ============================================================================
            try:
                # 1. Compute IsGood() condition matching C++:
                # bool IsGood(): track != null && status >= 0 && goodfit != 0
                trk_status = data['trk']["trk.status"]
                trk_goodfit = data['trk']["trk.goodfit"]

                is_good = (
                    ~ak.is_none(trk_status)
                    & ~ak.is_none(trk_goodfit)
                    & (trk_status >= 0)
                    & (trk_goodfit != 0)
                )
                is_good_clean = ak.fill_none(is_good, False)
                n_good = int(ak.sum(is_good_clean))

                # 2. Extract track times - use trk.t0 (fitted track time) to match C++
                # Fallback to first segment time at TT_Front if trk.t0 unavailable
                try:
                    track_times = data["trk"]["trk.t0"]
                    if track_times is None:
                        raise ValueError("trk.t0 unavailable")
                    self.logger.log(f"[upstream_veto] Using trk.t0 for track times", "debug")
                except Exception as e:
                    self.logger.log(f"trk.t0 access failed ({e}), using segment times fallback", "warning")
                    trk_t_front = data['trkfit']["trksegs"]["time"][at_trk_front]
                    track_times = ak.fill_none(ak.firsts(trk_t_front, axis=-1), 0.0)
                    self.logger.log(f"[upstream_veto] Using fallback segment times", "debug")

                # 3. Sanitize masks and handle missing OptionTypes/Nones
                trk_t_front_mean_clean = ak.fill_none(track_times, np.nan)
                has_valid_time = ak.fill_none(~np.isnan(trk_t_front_mean_clean), False)
                n_valid_time = int(ak.sum(has_valid_time))
                
                is_downstream_clean = ak.fill_none(is_downstream, False)
                n_ds = int(ak.sum(is_downstream_clean))

                # 4. Define Target (downstream) and Partner (upstream) track masks
                # Target: downstream signal track (PZFront > 0, IsGood(), Valid Time)
                is_target_track = is_downstream_clean & is_good_clean & has_valid_time
                
                # Partner: candidate upstream reflection track (PZFront < 0, IsGood(), Valid Time)
                is_partner_track = (~is_downstream_clean) & is_good_clean & has_valid_time

                # 5. Build FULL pairwise matrix (events, all_tracks, all_tracks)
                # for all possible track pairs
                trk_times_i = trk_t_front_mean_clean[:, :, None]  # (events, all_tracks, 1)
                trk_times_j = trk_t_front_mean_clean[:, None, :]  # (events, 1, all_tracks)
                dt_matrix = trk_times_i - trk_times_j  # (events, all_tracks, all_tracks)
                
                # Broadcast track property flags
                is_downstream_i = is_downstream_clean[:, :, None]  # (events, all_tracks, 1)
                is_upstream_j = (~is_downstream_clean)[:, None, :]  # (events, 1, all_tracks)
                is_good_i = is_good_clean[:, :, None]  # (events, all_tracks, 1)
                is_good_j = is_good_clean[:, None, :]  # (events, 1, all_tracks)
                
                # Exclude self-pairing
                track_idx_i = ak.local_index(is_downstream_clean, axis=1)[:, :, None]
                track_idx_j = ak.local_index(is_downstream_clean, axis=1)[:, None, :]
                is_self = track_idx_i == track_idx_j

                # 6. Check C++ veto window: 40 <= dt <= 110 ns
                # Negation of (dt < 40.f || dt > 110.f) -> flags bad candidate reflection pairs
                dt_in_veto_window = (dt_matrix >= 40.0) & (dt_matrix <= 110.0)

                # 7. Flag bad pairs: track_i is downstream (good), track_j is upstream (good), not self, dt in window
                bad_pair_matrix = (
                    dt_in_veto_window
                    & is_downstream_i & is_good_i  # track_i must be downstream & good
                    & is_upstream_j & is_good_j    # track_j must be upstream & good
                    & ~is_self                      # exclude self-pairing
                )

                # 8. For each track i, check if it has ANY bad upstream partner
                # Reduce axis=2 (all j partners) to get per-track result (events, all_tracks)
                has_bad_partner = ak.any(bad_pair_matrix, axis=2)  # (events, all_tracks)
                upstream_veto_per_track = ~has_bad_partner  # True = pass, False = fail
                upstream_veto_per_track = ak.fill_none(upstream_veto_per_track, True)

                # 8. PER-TRACK REDUCTION (Collapse axis 2 only to get per-track boolean)
                # Matches C++: For each target track, upstream_veto &= condition for ALL partners
                # Output: (events, tracks) per-track mask, True if track passes veto
                try:
                    n_pass = int(ak.sum(upstream_veto_per_track))
                    n_total = int(ak.count(upstream_veto_per_track))
                    n_fail_downstream = int(ak.sum(~upstream_veto_per_track & is_downstream_clean))
                    n_pass_downstream = int(ak.sum(upstream_veto_per_track & is_downstream_clean))
                except Exception as e2:
                    print(f"[ERROR] During summary: {type(e2).__name__}: {e2}")
                    print(f"[ERROR] Traceback: {traceback.format_exc()}")
                    raise

                # Register the cut mask
                cut_manager.add_cut(
                    name="upstream_veto",
                    description="Upstream timing veto: dt to upstream partners must be < 40 or > 110 ns",
                    mask=upstream_veto_per_track,
                    active=self.switch[7]
                )

            except Exception as e:
                print(f"[ERROR] upstream_veto EXCEPTION: {type(e).__name__}: {e}")
                print(f"[ERROR] Full traceback:\n{traceback.format_exc()}")
                self.logger.log(f"ERROR in upstream_veto: {type(e).__name__}: {e}", "error")
                self.logger.log(f"Full traceback:\n{traceback.format_exc()}", "error")

                # FALLBACK: All tracks pass (no veto applied)
                fallback_mask = ak.ones_like(ak.num(is_downstream, axis=1) > 0, dtype=bool)
                print(f"[ERROR] upstream_veto: Using FALLBACK (all tracks pass) - CHECK ERRORS ABOVE")
                self.logger.log(f"upstream_veto: Using FALLBACK (all tracks pass) - CHECK ERRORS ABOVE", "warning")
                cut_manager.add_cut(
                    name="upstream_veto",
                    description="Upstream timing veto: dt to upstream partners must be < 40 or > 110 ns",
                    mask=fallback_mask,
                    active=self.switch[7]
                )
            # ============================================================================
            # CUT 8: no_multi_trk_veto
            # ============================================================================
            if (str(self.sign) == "minus"):
                try:
                    dt_threshold = 150.0
                    
                    # DIAGNOSTICS: Step-by-step breakdown
                    is_reco_electron = selector.is_electron(data["trk"])
                    is_reco_positron = selector.is_positron(data["trk"])
                    
                    # Get downstream electron/positron tracks (must also be IsGood like C++)
                    # IsGood(): status >= 0 && goodfit != 0
                    trk_status = data['trk']["trk.status"]
                    trk_goodfit = data['trk']["trk.goodfit"]
                    is_good = (
                        ~ak.is_none(trk_status)
                        & ~ak.is_none(trk_goodfit)
                        & (trk_status >= 0)
                        & (trk_goodfit != 0)
                    )
                    is_good_clean = ak.fill_none(is_good, False)
                    
                    is_downstream_epm = is_downstream & (is_reco_electron | is_reco_positron) & is_good_clean
                    
                    # If no e/e+ tracks found, veto passes all (no coincident pairs possible)
                    n_total_epm = int(ak.sum(is_downstream_epm))
                    if n_total_epm == 0:
                        multi_trk_per_track = ak.ones_like(is_downstream, dtype=bool)
                    else:
                        # Get track times using TFront property (safer than extracting segments)
                        # This uses the fitted track time at the tracker front
                        try:
                            track_times = data["trk"]["trk.t0"]
                            if track_times is None:
                                raise ValueError("Track times unavailable")
                            print(f"[no_multi_trk_veto DIAG-3] Using trk.t0 for track times")
                        except Exception as time_err:
                            # Fallback: extract from trkfit segments - use MEAN like upstream_veto does
                            trk_times = data['trkfit']["trksegs"]["time"][at_trk_front]
                            track_times = ak.mean(trk_times, axis=-1, keepdims=False)
                            track_times = ak.fill_none(track_times, 0.0)
                        
                        # Broadcast for pairwise comparison: (events, tracks, tracks)
                        track_times_i = track_times[:, :, None]
                        track_times_j = track_times[:, None, :]
                        
                        # Pairwise time differences (use np.abs, not ak.abs)
                        dt_matrix = np.abs(track_times_i - track_times_j)
                        
                        # Broadcast particle type flags: both track i and track j must be downstream e/e+
                        is_good_i = is_downstream_epm[:, :, None]  # (events, tracks, 1)
                        is_good_j = is_downstream_epm[:, None, :]  # (events, 1, tracks)
                        
                        # Check for coincident tracks: 
                        # - Both track i and j are downstream e/e+
                        # - Within 150 ns
                        is_coincident = (dt_matrix < dt_threshold) & is_good_i & is_good_j
                        
                        # Count coincident tracks per track (including self)
                        n_coincident_per_track = ak.sum(is_coincident, axis=-1)
                        
                        # Subtract self-contribution (diagonal where i==j, only if track is e/e+)
                        self_count = ak.where(is_downstream_epm, 1, 0)
                        has_other_coincident = n_coincident_per_track - self_count > 0
                        
                        # Per-track: PASS if this track has NO other coincident partners (matches C++: multi_trk &= std::fabs(dt) > 150)
                        multi_trk_per_track = ~has_other_coincident
                    
                    cut_manager.add_cut(
                        name="no_multi_trk_veto",
                        description="No coincident multi-track: |dt| >= 150 ns between downstream e/e+ tracks",
                        mask=multi_trk_per_track,
                        active=self.switch[8]
                    )
                    data["no_multi_trk_veto"] = multi_trk_per_track
                except Exception as e:
                    self.logger.log(f"Error in multi-track timing veto: {e}", "warning")
                    import traceback
                    self.logger.log(traceback.format_exc(), "warning")
                    # On error, pass the cut (don't veto)
                    cut_manager.add_cut(
                        name="no_multi_trk_veto",
                        description="No coincident multi-track: |dt| >= 150 ns between downstream e/e+ tracks",
                        mask=ak.ones_like(is_downstream, dtype=bool),
                        active=self.switch[8]
                    )
            else:
                # For plus sign, no multi-track veto
                cut_manager.add_cut(
                    name="no_multi_trk_veto",
                    description="No coincident multi-track: |dt| >= 150 ns between downstream e/e+ tracks",
                    mask=ak.ones_like(is_downstream, dtype=bool),
                    active=self.switch[8]
                )

            # ============================================================================
            # CUT 10: good_trkpid
            # ============================================================================
            good_trkpid = selector.select_trkpid(data["trk"], value=0.54)
            
            # Also check for calorimeter cluster energy > 0 (some energy deposited)
            try:
                # Access calorimeter cluster energy from calo branch
                calo_energy = data["calo"]["caloclusters.energyDep_"]
                # Check if any cluster has energy > 0
                has_calo_energy = ak.any(calo_energy > 0, axis=-1)
                # Broadcast to track level: each track's PID check includes calo energy requirement
                has_calo_energy_broadcast = ak.broadcast_arrays(has_calo_energy, data["trk"]["trk.pdg"])[0]
                # Combine PID with calorimeter energy requirement
                good_trkpid = good_trkpid & has_calo_energy_broadcast
            except (KeyError, TypeError) as e:
                # If calorimeter energy not available, just use PID
                self.logger.log(f"Calorimeter energy not found: {e}, PID cut uses selector only", "warning")
            
            data["good_trkpid"] = good_trkpid
            cut_manager.add_cut(
                name="good_trkpid",
                description="Track PID > 0.55 and event has calorimeter cluster energy > 0",
                mask=good_trkpid,
                active=self.switch[9]
            )

            # ============================================================================
            # CUT 10: pz_over_pt
            # ============================================================================
            # Use tanDip from segment parameters at TT_Front (matches C++ TanDipFront())
            try:
                # Get tanDip directly from segment parameters at TT_Front
                tandip_at_front = data['trkfit']["trksegpars_lh"]["tanDip"][at_trk_front]
                # For each track, use the tanDip value at TT_Front (should be single value per track)
                # Extract the first (or only) value at TT_Front for each track
                tandip_values = ak.firsts(tandip_at_front, axis=-1)
                # Fill any missing with -100 sentinel
                tandip_values = ak.fill_none(tandip_values, -100.0)
            except Exception:
                # Fallback: compute from momentum if segment params unavailable
                vec = Vector(verbosity=0)
                trkfit_ent_pzpt = ak.mask(data['trkfit']["trksegs"], at_trk_front)
                vec3 = vec.get_vector(trkfit_ent_pzpt, 'mom')
                if vec3 is None:
                    tandip_values = ak.full_like(data['trk']["trk.status"], -100.0, dtype=float)
                else:
                    px = vec3.x
                    py = vec3.y
                    pz = vec3.z
                    pt = vec3.rho
                    # Compute tanDip = pz / pt
                    tandip_values = ak.where((pt > 0) & (pz != 0), pz / pt, -100.0)
                    # Get per-track value (use first segment at TT_Front)
                    tandip_values = ak.firsts(tandip_values, axis=-1)
                    tandip_values = ak.fill_none(tandip_values, -100.0)

            # Apply cut: 0.575 < tanDip < 0.85
            mask_pzpt = (tandip_values > 0.575) & (tandip_values < 0.85)

            cut_manager.add_cut(
                name="pz_over_pt",
                description="Track-level cut: 0.575 < tanDip < 0.85",
                mask=mask_pzpt,
                active=self.switch[10]
            )

            # ============================================================================
            # CUT 10: st_boundary
            # ============================================================================
            # Check if track has segments at any stopping target (ST) surface
            # Matches C++ Track_t::STBoundary() logic
            try:
                st_surfaces = ["ST_Front", "ST_Back", "ST_Inner", "ST_Outer"]
                st_boundary_mask = None
                
                for surface in st_surfaces:
                    trk_st = selector.select_surface(data['trkfit'], surface_name=surface)
                    if trk_st is not None:
                        # Check if any segment at this surface exists (reduce to track level)
                        has_surface = ak.any(trk_st, axis=-1)
                        if st_boundary_mask is None:
                            st_boundary_mask = has_surface
                        else:
                            st_boundary_mask = st_boundary_mask | has_surface
                
                # If no ST surfaces found, default to False for all tracks
                if st_boundary_mask is None:
                    st_boundary = ak.zeros_like(is_downstream, dtype=bool)
                else:
                    st_boundary = st_boundary_mask
            except Exception as e:
                self.logger.log(f"ST boundary check failed: {e}", "warning")
                st_boundary = ak.zeros_like(is_downstream, dtype=bool)
            
            cut_manager.add_cut(
                name="st_boundary",
                description="STBoundary > 0",
                mask=st_boundary,
                active=self.switch[11]
            )

            # ============================================================================
            # CUT 12: has_st
            # ============================================================================
            has_st = selector.has_ST(data['trkfit'])
            cut_manager.add_cut(
                name="has_st",
                description="has Nst > 0",
                mask=has_st,
                active=self.switch[12]
            )

            # ============================================================================
            # CUT 13: no_opa
            # ============================================================================
            no_OPA = selector.has_OPA(data['trkfit'])
            cut_manager.add_cut(
                name="no_opa",
                description="has N_opa == 0",
                mask=no_OPA,
                active=self.switch[12]
            )

            # ============================================================================
            # CUT 14: good_trkqual
            # ============================================================================
            good_trkqual = selector.select_trkqual(data["trk"], quality=0.155)
            data["good_trkqual"] = good_trkqual
            cut_manager.add_cut(
                name="good_trkqual",
                description="Track quality > 0.155",
                mask=good_trkqual,
                active=self.switch[13]
            )

            # ============================================================================
            # CUT 15: has_hits
            # ============================================================================
            has_hits = selector.has_n_hits(data["trk"], n_hits=20)
            cut_manager.add_cut(
                name="has_hits",
                description="Minimum of 20 active hits in the tracker",
                mask=has_hits,
                active=self.switch[14]
            )

            # ============================================================================
            # CUT 16: within_t0err
            # ============================================================================
            within_t0err = ((data['trkfit']["trksegpars_lh"]["t0err"]) < 0.85)
            within_t0err = ak.all(~at_trk_mid | within_t0err, axis=-1)
            cut_manager.add_cut(
                name="within_t0err",
                description="t0err < 0.85",
                mask=within_t0err,
                active=self.switch[15]
            )

            # ============================================================================
            # CUT 17: no_crv_veto
            # ============================================================================
            # Asymmetric veto: only veto if CRV cluster is in the future (0 < dt < 150)
            # C++ logic: if(deltat_crv > 0.f && deltat_crv < 150.f) fail_crv = true;
            dt_threshold = 150
            trk_times_crv = data['trkfit']["trksegs"]["time"][at_trk_front]
            coinc_times = data["crv"]["crvcoincs.time"]
            
            coinc_broadcast = coinc_times[:, None, None, :]
            trk_broadcast = trk_times_crv[:, :, :, None]

            # Asymmetric timing: dt = track_time - crv_time
            dt_crv = trk_broadcast - coinc_broadcast
            # Veto if 0 < dt < 150 (CRV cluster in future)
            any_coinc = (dt_crv > 0) & (dt_crv < dt_threshold)
            min_dt_any = ak.any(any_coinc, axis=3)

            # Reduce over segments (axis=2)
            veto = ak.any(min_dt_any, axis=2)
            data["no_crv_veto"] = ~veto
            cut_manager.add_cut(
                name="no_crv_veto",
                description="No crv-trk veto: 0 < dt < 150 ns (asymmetric)",
                mask=~veto,
                active=self.switch[16]
            )

            # ============================================================================
            # CUT 17: in_mom_range (100-110 MeV/c)
            # ============================================================================
            in_mom_range = ((100 < mom_mag) & (mom_mag < 110))
            in_mom_range = ak.all(~at_trk_front | in_mom_range, axis=-1)
            cut_manager.add_cut(
                name="in_mom_range",
                description="100 < mom < 110 MeV/c",
                mask=in_mom_range,
                active=self.switch[17]
            )

            # ============================================================================
            # CUT 18: within_t0_475 (475-1650 ns)
            # ============================================================================
            within_t0_475 = ((475 < data['trkfit']["trksegs"]["time"]) & 
                            (data['trkfit']["trksegs"]["time"] < 1650))
            within_t0_475 = ak.all(~at_trk_front | within_t0_475, axis=-1)
            cut_manager.add_cut(
                name="within_t0_475",
                description="475 < t_0 < 1650 ns",
                mask=within_t0_475,
                active=self.switch[18]
            )

            # ============================================================================
            # CUT 19: within_t0_540 (540-1650 ns)
            # ============================================================================
            within_t0_540 = ((540 < data['trkfit']["trksegs"]["time"]) & 
                            (data['trkfit']["trksegs"]["time"] < 1650))
            within_t0_540 = ak.all(~at_trk_front | within_t0_540, axis=-1)
            cut_manager.add_cut(
                name="within_t0_540",
                description="540 < t_0 < 1650 ns",
                mask=within_t0_540,
                active=self.switch[19]
            )

            # ============================================================================
            # CUT 20: within_t0_640 (640-1650 ns)
            # ============================================================================
            within_t0_640 = ((640 < data['trkfit']["trksegs"]["time"]) & 
                            (data['trkfit']["trksegs"]["time"] < 1650))
            within_t0_640 = ak.all(~at_trk_front | within_t0_640, axis=-1)
            cut_manager.add_cut(
                name="within_t0_640",
                description="640 < t_0 < 1650 ns",
                mask=within_t0_640,
                active=self.switch[20]
            )

            # ============================================================================
            # CUT 21: signal_region
            # ============================================================================
            # Matches C++ RunACutFlow: PFront() > 103.34 && PFront() < 104.74 && TFront() > 640 && TFront() < 1650
            
            # Get momentum magnitude at TT_Front (matches C++ PFront())
            mom_at_front = data['trkfit']["trksegs"]["mom"][at_trk_front]
            mom_mag_at_front = np.sqrt(
                mom_at_front["fCoordinates"]["fX"]**2 + 
                mom_at_front["fCoordinates"]["fY"]**2 + 
                mom_at_front["fCoordinates"]["fZ"]**2
            )
            # Extract per-track value (earliest segment at front)
            mom_front_per_track = ak.firsts(mom_mag_at_front, axis=-1)
            mom_front_per_track = ak.fill_none(mom_front_per_track, -100.0)
            
            # Get time at TT_Front (matches C++ TFront())
            time_at_front = data['trkfit']["trksegs"]["time"][at_trk_front]
            # Extract per-track value (earliest segment at front)
            time_front_per_track = ak.firsts(time_at_front, axis=-1)
            time_front_per_track = ak.fill_none(time_front_per_track, -100.0)
            
            # Apply signal region cuts: momentum and time at front
            signal_region = (
                (mom_front_per_track > 103.34) & (mom_front_per_track < 104.74) &
                (time_front_per_track > 640.0) & (time_front_per_track < 1650.0)
            )
            
            cut_manager.add_cut(
                name="signal_region",
                description="Signal region: 103.34 < P_Front < 104.74, 640 < T_Front < 1650 ns",
                mask=signal_region,
                active=self.switch[21]
            )
            data["signal_region"] = signal_region

            # ============================================================================
            # MLP Scoring (optional, diagnostic only - NOT a cut)
            # ============================================================================
            self.logger.log(f"MLP trainer available: {self.mlp_trainer is not None}", "info")
            if self.mlp_trainer is not None and not skip_mlp:
                self.logger.log(f"Computing MLP scores", "info")
                try:
                    at_trk_front_segs = self.selector.select_surface(data["trkfit"], surface_name="TT_Front")
                    at_trk_front_trk = ak.any(at_trk_front_segs, axis=-1)
                    at_trk_front_trk = ak.fill_none(at_trk_front_trk, False)
                    
                    d0_segs = ak.mask(data["trkfit"]["trksegpars_lh"]["d0"], at_trk_front_segs)
                    rmax_segs = ak.mask(data["trkfit"]["trksegpars_lh"]["maxr"], at_trk_front_segs)
                    
                    trkfit_segs_masked = ak.mask(data["trkfit"]["trksegs"], at_trk_front_segs)
                    mom_vec = self.vector.get_vector(trkfit_segs_masked, "mom")
                    p_mag = self.vector.get_mag(trkfit_segs_masked, "mom")
                    costheta_segs = mom_vec.z / p_mag
                    
                    d0_flat = ak.to_numpy(ak.flatten(d0_segs, axis=None))
                    rmax_flat = ak.to_numpy(ak.flatten(rmax_segs, axis=None))
                    costheta_flat = ak.to_numpy(ak.flatten(costheta_segs, axis=None))
                    
                    valid_indices = np.where(
                        ~np.isnan(d0_flat) & ~np.isnan(rmax_flat) & ~np.isnan(costheta_flat)
                    )[0]
                    
                    self.logger.log(f"Scoring {len(valid_indices)} / {len(d0_flat)} valid segments", "info")
                    
                    scores_flat = np.full(len(d0_flat), -999.0, dtype=np.float32)
                    if len(valid_indices) > 0:
                        scores_valid = self.mlp_trainer.score(
                            d0_flat[valid_indices],
                            rmax_flat[valid_indices],
                            costheta_flat[valid_indices]
                        )
                        scores_flat[valid_indices] = np.asarray(scores_valid, dtype=np.float32).ravel()
                        self.logger.log(
                            f"Score stats — Min: {scores_valid.min():.4f}, Max: {scores_valid.max():.4f}, "
                            f"Mean: {scores_valid.mean():.4f}, Pass (>0.6): {np.sum(scores_valid > 0.6)}/{len(scores_valid)}",
                            "info"
                        )
                    
                    counts_per_track_2d = ak.num(d0_segs, axis=-1)
                    mlp_pass_trk_list = []
                    score_idx = 0
                    
                    for evt_idx in range(len(ak.to_numpy(ak.num(d0_segs, axis=1)))):
                        event_trk_counts = ak.to_numpy(counts_per_track_2d[evt_idx])
                        event_passes = []
                        
                        for trk_idx in range(len(event_trk_counts)):
                            n_segs = event_trk_counts[trk_idx]
                            segment_scores = scores_flat[score_idx:score_idx + int(n_segs)]
                            passes = np.any(segment_scores > 0.6)
                            event_passes.append(passes)
                            score_idx += int(n_segs)
                        
                        mlp_pass_trk_list.append(event_passes)
                    
                    n_pass = np.sum([np.any(event) for event in mlp_pass_trk_list])
                    self.logger.log(f"MLP: {n_pass} events pass score threshold", "info")
                    
                    self.mlp_scores = scores_flat
                    self.mlp_pass_trk = mlp_pass_trk_list

                except Exception as e:
                    self.logger.log(f"Error in MLP scoring: {e}", "error")
                    self.logger.log(traceback.format_exc(), "error")

            self.logger.log("All cuts defined", "success")
            
        except Exception as e:
            self.logger.log(f"Error defining cuts: {e}", "error") 
            self.logger.log(traceback.format_exc(), "error")
            return None  


    def apply_cuts(self, data, cut_manager, group=None, active_only=True):

        ## data_cut needs to be an awkward array 

        """Apply all trk-level mask to the data

        Args:
            data: Data to apply cuts to
            mask: Mask to apply 
    
        Returns:
            ak.Array: Data after cuts applied
        """
        self.logger.log("Applying cuts to data", "info")

        try:
            #check mc truth codes before cuts
            #mc_parts = self.mc_pre_cuts(data)
    
            # Copy the array 
            # This is memory intensive but the easiest solution for what I'm trying to do
            data_cut = ak.copy(data) 
    
            # Combine cuts
            self.logger.log(f"Combining cuts", "info") 

            # Track-level mask
            trk_mask = cut_manager.combine_cuts(active_only=active_only)
    
            # Select tracks
            self.logger.log("Selecting tracks", "max")
            data_cut['trk'] = data_cut["trk"][trk_mask]
            data_cut['trkfit'] = data_cut['trkfit'][trk_mask]
            data_cut['trkmc'] = data_cut["trkmc"][trk_mask]

            # Then clean up events with no tracks after cuts
            self.logger.log(f"Cleaning up events with no tracks after cuts", "max") 
            data_cut = data_cut[ak.any(trk_mask, axis=-1)] 
    
            self.logger.log(f"Cuts applied successfully", "success")
    
            return data_cut
    
        except Exception as e:
            self.logger.log(f"Error applying cuts: {e}", "error") 
            self.logger.log(traceback.format_exc(), "error")
            return None


    def apply_mlp_score_filter(self, data, trk_mask=None):
        """Apply post-hoc MLP score-based filtering to data
        
        Args:
            data (ak.Array): Data to filter (already filtered by cuts)
            trk_mask (ak.Array): Original track-level mask from cuts (shape: original events × tracks)
            
        Returns:
            ak.Array: Filtered data containing only tracks passing MLP score threshold
        """
        if not hasattr(self, 'mlp_pass_trk') or self.mlp_pass_trk is None:
            self.logger.log("MLP track-level pass/fail data not available", "warning")
            return data
        
        try:
            self.logger.log(f"Applying MLP score filter (threshold=0.6)", "info")
            
            if trk_mask is None:
                self.logger.log("No track mask provided, skipping MLP filter", "warning")
                return data
            
            # Apply the same track filtering to mlp_pass_trk to align it with data_filtered
            # For each event, keep only the MLP pass/fail values for tracks that passed the cut
            mlp_pass_trk_filtered_by_cut = []
            
            for evt_idx in range(len(self.mlp_pass_trk)):
                if evt_idx < len(trk_mask):
                    evt_cut_mask = ak.to_numpy(trk_mask[evt_idx])
                    evt_mlp_passes = self.mlp_pass_trk[evt_idx]
                    
                    # Keep only MLP values for tracks that passed cuts
                    filtered_mlp = [evt_mlp_passes[trk_idx] 
                                   for trk_idx in range(len(evt_mlp_passes)) 
                                   if trk_idx < len(evt_cut_mask) and evt_cut_mask[trk_idx]]
                    
                    mlp_pass_trk_filtered_by_cut.append(np.array(filtered_mlp, dtype=bool))
            
            # Identify which events survived the cuts
            evt_survived = ak.any(trk_mask, axis=-1)
            evt_survived_np = ak.to_numpy(evt_survived)
            
            # Filter to only surviving events
            mlp_mask_list = [mlp_pass_trk_filtered_by_cut[i] 
                            for i in range(len(evt_survived_np)) 
                            if evt_survived_np[i]]
            
            mlp_mask = ak.Array(mlp_mask_list)
            
            # Sanity check shapes
            n_data_events = len(data)
            n_mlp_events = len(mlp_mask)
            if n_data_events != n_mlp_events:
                self.logger.log(f"Shape mismatch: data has {n_data_events} events, MLP mask has {n_mlp_events} events", "error")
                return data
            
            # Log mask statistics
            n_mask_true = int(ak.sum(ak.flatten(mlp_mask, axis=None)))
            n_mask_total = int(ak.sum(ak.num(mlp_mask, axis=1)))
            self.logger.log(f"MLP mask: {n_mask_true}/{n_mask_total} tracks pass threshold", "info")
            
            # Apply mask to data
            data_filtered = ak.copy(data)
            data_filtered['trk'] = data_filtered["trk"][mlp_mask]
            data_filtered['trkfit'] = data_filtered['trkfit'][mlp_mask]
            data_filtered['trkmc'] = data_filtered["trkmc"][mlp_mask]
            
            # Clean up events with no tracks after filtering
            data_filtered = data_filtered[ak.any(mlp_mask, axis=-1)]
            
            # Count tracks properly by summing track counts per event
            try:
                n_before = int(ak.sum(ak.num(data['trk'], axis=1)))
                n_after = int(ak.sum(ak.num(data_filtered['trk'], axis=1)))
                self.logger.log(f"MLP filter: {n_before} → {n_after} tracks ({100*n_after/n_before:.1f}%)", "info")
            except Exception:
                self.logger.log(f"MLP filter applied (track count unavailable)", "info")
            
            return data_filtered
            
        except Exception as e:
            self.logger.log(f"Error applying MLP score filter: {e}", "error")
            self.logger.log(traceback.format_exc(), "error")
            return data
    
    # Helper to convert the cut stats into a list 
    def get_stats_list(self, results):
        stats = [] 
        if isinstance(results, list): 
            for result in results: 
                if "cut_stats" in result: 
                    stats.append(result["cut_stats"])
        else: 
            stats.append(results["cut_stats"])
        return stats

    def plot_mlp_scores(self, data, data_cut, file_id):
        """Plot MLP score distribution."""
        try:
            if not hasattr(self, 'mlp_scores') or self.mlp_scores is None:
                self.logger.log("MLP scores not available for plotting", "info")
                return
            
            self.logger.log(f"Attempting to plot {len(self.mlp_scores)} total MLP scores", "info")
            
            # Filter out invalid scores (-999.0)
            valid_scores = self.mlp_scores[self.mlp_scores > -900]
            
            self.logger.log(f"Found {len(valid_scores)} valid MLP scores", "info")
            
            if len(valid_scores) == 0:
                self.logger.log("No valid MLP scores to plot", "warning")
                return
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(valid_scores, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
            ax.set_xlabel('MLP Score', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title('MLP Score Distribution', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            # Extract filename from full path and sanitize
            import os
            base_name = os.path.basename(file_id).replace('.root', '')
            plot_path = f"mlp_scores_{base_name}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            self.logger.log(f"MLP score histogram saved to {plot_path}", "info")
            plt.close()
            
        except Exception as e:
            self.logger.log(f"Error plotting MLP scores: {e}", "warning")
            self.logger.log(traceback.format_exc(), "warning")

    def execute(self, data, file_id,  inactive_cuts=None):
        """Perform complete analysis on an array
        Args:
            data: The data to analyse
            file_id: Identifier for the file
            cut_names: List of cuts to activate/deactivate
            active: activate/deactive cuts
        Returns:
            dict: Complete analysis results
        """
        self.logger.log(f"Beginning analysis execution for file: {file_id}", "info")
        try:

            # Create a unique cut manager for this file
            cut_manager = CutManager(verbosity=self.verbosity)

            self.logger.log("Defining cuts", "max")
            # Define cuts
            self.define_cuts(data, cut_manager)

            # Set activate cuts
            if inactive_cuts: 
                cut_manager.toggle_cut(inactive_cuts, active=False)
    
            # Calculate cut stats
            self.logger.log("Getting cut stats", "max")
            cut_stats = cut_manager.create_cut_flow(data)
        
            # Mark CE-like tracks (useful for debugging 
            data["CE_like"] = cut_manager.combine_cuts(active_only=True)
            
            # Get track-level mask for alignment with MLP filter
            trk_mask = cut_manager.combine_cuts(active_only=True)
            
            # Apply cuts
            data_CE = self.apply_cuts(data, cut_manager) # Just CE-like tracks 
            
            # Log track count before MLP filter
            try:
                n_before_mlp = int(ak.sum(ak.num(data_CE['trk'], axis=1)))
                self.logger.log(f"Before MLP filter: {n_before_mlp} tracks", "info")
            except Exception:
                pass
            
            # Apply MLP score filter if available (pass the mask to align shapes)
            # MLP filter is OFF by default - set apply_mlp=True to enable
            apply_mlp = False  # Set to True to enable MLP filtering
            if apply_mlp:
                data_CE_mlp = self.apply_mlp_score_filter(data_CE, trk_mask=trk_mask)
                
                # Log track count after MLP filter
                try:
                    n_after_mlp = int(ak.sum(ak.num(data_CE_mlp['trk'], axis=1)))
                    self.logger.log(f"After MLP filter: {n_after_mlp} tracks", "info")
                except Exception:
                    pass
            else:
                data_CE_mlp = data_CE
                self.logger.log("MLP filter disabled (apply_mlp=False)", "info")
            
            # Plot MLP scores if available (disabled by default)
            # self.plot_mlp_scores(data, data_CE, file_id)
            
            # Compile all results
            self.logger.log("Analysis completed", "success")


            result = {
                "cut_stats": cut_stats,
                "filtered_data": data_CE_mlp  # Return MLP-filtered data if available
            }

            return result
    
        except Exception as e:
            self.logger.log(f"Error during analysis execution: {e}", "error")  
            self.logger.log(traceback.format_exc(), "error")
            return None, None