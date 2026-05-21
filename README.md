# Run-1A Analysis

A Python-based analysis pipeline for processing and analyzing Mu2e Run 1A simulation and data. This codebase processes ROOT files from the Mu2e experiment, applies physics cuts, and performs statistical analyses on track and event information.

## Overview

This analysis framework processes ROOT files containing Mu2e detector events, applies a series of physics-motivated cuts, and generates distributions for background studies and signal optimization. The pipeline supports both electron and positron analysis with configurable cut selections.

## Project Structure

```
Run-1A-Analysis/
├── README.md                    # This file
├── analyze.py                   # Main analysis class with cut definitions
├── process.py                   # Data processing driver and file handler
├── compare.py                   # Comparison and plotting utilities
├── fits.py                      # Fitting routines (polynomial, landau, custom PDFs)
├── cosmics.py                   # Cosmic ray analysis and momentum fitting
├── CE/                          # Charged Endpoint (CE) configuration
│   ├── CeEnd.fcl               # FCL configuration for CE Endpoint
│   ├── CeLL.fcl                # FCL configuration for CE LL process
│   ├── CE_gen.txt              # CE generation parameters
│   └── plot_genE.py            # CE energy comparison plotting
├── file_lists/                  # Input file lists (text format)
│   ├── CeMLL_MDC2025an_best_nomix.txt
│   ├── DIOtail95_MDC2025an_best_nomix.txt
│   ├── ExtRPC_MDC2025an_nomix.txt
│   ├── IntRPC_MDC2025an_nomix.txt
│   ├── IPADIO_MDC2025an_nomix.txt
│   └── _processing_status.txt
└── cut_lists/                   # Output CSV files with cut statistics
    ├── eminus_CeMLL_cut_stats.csv
    └── eminus_DIO_cut_stats.csv
```

## Dependencies

This project requires the following Python packages:

- **Core Data Processing**: `uproot`, `awkward`, `numpy`, `pandas`
- **Machine Learning**: `xgboost`, `scikit-learn`
- **Plotting**: `matplotlib`
- **Physics Fitting**: `zfit` (with optional TensorFlow backend)
- **Custom Utilities**: `pyutils` module (not included; external dependency)

### Installation

```bash
pip install uproot awkward numpy pandas xgboost scikit-learn matplotlib zfit
```

## Main Components

### `analyze.py`
Defines the core analysis logic with the `Analyze` class:
- **Cut Definitions**: Physics-motivated selection criteria (21 different cuts)
- **Track Selection**: Surface-based track filtering (TT Front, ST Foils, OPA surfaces)
- **Signal/Background Separation**: Particle type identification using MC truth information
- **Cut Flow Management**: Tracks cut efficiencies and event flow through the selection

**Key Methods:**
- `has_trk_front_segment()`: Check if track intersects the TT Front surface
- `define_cuts()`: Apply all physics cuts to the data
- `execute()`: Run full analysis pipeline on a dataset

### `process.py`
Main processing driver (`AnaProcessor` class):
- **File I/O**: Reads ROOT files via UPRoot and extracts requested branches
- **Parallel Processing**: Multi-threaded file processing with configurable job count
- **Timeout Protection**: Prevents hanging on problematic files (60s timeout per file)
- **Data Aggregation**: Combines results from all files and generates summary statistics
- **Output Generation**: Creates CSV files with cut statistics for each process

**Key Methods:**
- `process_file()`: Extract and analyze a single ROOT file
- `postprocess()`: Combine results from all files into unified datasets
- `execute()`: Main entry point for running the full pipeline

### `compare.py`
Visualization and comparison utilities (`Compare` class):
- **Distribution Plotting**: Overlays histograms across different datasets/cuts
- **Resolution Studies**: Compares reconstructed vs true values
- **Particle Counting**: Visualizes different process yields
- **Publication-Quality Plots**: Uses serif fonts and professional styling

**Key Methods:**
- `plot_variable()`: Create overlaid histograms with custom formatting
- `compare_resolution()`: Compute resolution between reconstructed and true parameters
- `plot_particle_counts()`: Visualize MC truth particle type distributions

### `fits.py`
Advanced fitting functionality (`Fits` class):
- **Custom PDF Models**: Error function, Landau, and polynomial models
- **Extended Unbinned Fits**: Maximum likelihood fitting using zfit framework
- **Momentum Resolution**: Gaussian convolution with Landau energy loss
- **Parameter Extraction**: Extracts signal and background yields

**Supported Models:**
- Custom ERF PDF (error function based)
- Landau distribution (energy loss)
- Polynomial backgrounds

### `cosmics.py`
Cosmic ray and general momentum fitting (`Cosmics` class):
- **Momentum Distribution Fitting**: Fits reconstructed momentum spectra
- **Background Characterization**: Generates plots of cosmic ray data
- **Extended Fits**: Uses extended unbinned maximum likelihood

**Key Methods:**
- `fit_momentum()`: Fit momentum distributions with polynomial backgrounds

## Command Line Usage

### Basic Syntax

```bash
python process.py --file <file_list> [options]
```

### Required Arguments

- `--file`: Path to file list (text file with one ROOT file path per line) or individual ROOT file

### Optional Arguments

- `--sign`: Particle sign to analyze (`minus` or `plus`, default: `minus`)
- `--loc`: File location (`disk` or `tape`, default: `disk`)
- `--proctype`: Processing type (`ensemble` or `cosmics`, default: `ensemble`)
- `--jobs`: Number of parallel jobs (should equal number of files, default: 1)
- `--verbose`: Verbosity level (0-3, default: 1)

### Examples

```bash
# Analyze electron (minus) sample from file list
python process.py --file file_lists/DIOtail95_MDC2025an_best_nomix.txt --sign minus --jobs 5

# Analyze positron (plus) sample with verbose output
python process.py --file file_lists/CeMLL_MDC2025an_best_nomix.txt --sign plus --verbose 2

# Analyze cosmic ray events
python process.py --file file_lists/Cosimcs_MDC2025an_nomix.txt --proctype cosmics

# Single file analysis from disk
python process.py --file /path/to/simulation.root --sign minus --loc disk
```

## Input Data Format

### File Lists
Text files with one ROOT file path per line:
```
/pnfs/mu2e/path/to/file1.root
/pnfs/mu2e/path/to/file2.root
/pnfs/mu2e/path/to/file3.root
```

### ROOT File Structure
Expected branches in input ROOT files:
- `evt`: Event information (run, subrun, event)
- `trkfit`: Reconstructed track information with segments
- `trkmc`: MC truth track information
- `trk`: Track quality and PID information
- `crv`: Cosmic ray veto coincidence data

## Output Files

### Cut Statistics CSV
- **Filename**: `{eplus|eminus}_{proctype}_cut_stats.csv`
- **Format**: Cut name, events passing, cumulative efficiency
- **Example**: `eminus_ensemble_cut_stats.csv`

### Plots (from Compare class)
- Momentum distributions: `{prefix}_recomom.pdf`
- Track quality: `{prefix}_trkqual.pdf`
- Timing: `{prefix}_DT.pdf`, `{prefix}_time.pdf`
- Track geometry: `{prefix}_rmax.pdf`, `{prefix}_d0.pdf`
- Particle counts: `{prefix}_particle_counts.pdf`

### Fitted Data
- **Filename**: `output_data.csv`
- **Format**: Single column with reconstructed momenta in fit range
- **Format**: Used for external fitting tools (e.g., BAT)

## Physics Cuts

The analysis applies 21 configurable cuts:

0. Is reconstructed electron
1. Has downstream track
2. Track intersects TT Front surface
3. Good track quality and PID
4. Good track quality (plus only)
5. Good track PID (minus only)
6. Within T0 window
7. Within T0 error range
8. Has track hits
9. Within LHR maximum value
10. Within d0 range
11. Within pitch angle
12. Intersects straw tracker (ST)
13. Does not intersect OPA
14. CRV veto (no cosmic coincidences)
15. CRV quality cuts
16. CRV timing window
17. Pz/Pt ratio cut
18. Trigger selection
19. Within momentum-time correlation
20. Early time rejection
21. Reflection rejection

### Cut Configuration

Cut switches can be enabled/disabled by passing boolean arrays. Different cut sets are defined for `minus` vs `plus` particles to optimize for each signal type.

## Running the Full Analysis Workflow

### Step 1: Prepare File Lists
Create text files in `file_lists/` with full paths to ROOT files.

### Step 2: Run Analysis
```bash
python process.py --file file_lists/DIOtail95_MDC2025an_best_nomix.txt \
                  --sign minus \
                  --jobs 15 \
                  --verbose 1
                  --loc "disk"
                  --proctype "DIO"
```

### Step 3: Review Outputs
- Check `eminus_ensemble_cut_stats.csv` for cut statistics
- Review generated plots in current directory
- If using cosmics mode, check `output_data.csv` for fitted data

## Troubleshooting

### File Processing Timeouts
If individual files timeout (60s limit):
- Check file validity with `uproot`
- Files may be corrupted or too large
- Monitor `_processing_status.txt` for currently processing file

### Import Errors
Ensure `pyutils` is in your Python path:
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/pyutils
```

### Memory Issues
- Reduce `--jobs` parameter to process fewer files in parallel
- Split file lists into smaller batches
- Check available system RAM

### Missing Data or Branches
- Verify input ROOT files contain expected branches
- Check that files were produced with compatible Mu2e software version
- Review verbose output for specific branch extraction errors

## Physics Background

This analysis focuses on charged lepton searches in the Mu2e experiment, with emphasis on:
- **Signal** (Charge Exchange - CE): nn → pμ⁻ processes
- **Backgrounds**: DIO (Decay In Orbit), cosmic rays, RPCs (Radiative Pion Capture)

The analysis separates signal from background using:
- Kinematic cuts (momentum, timing, angle)
- Detector geometry constraints
- Machine learning (XGBoost) for advanced cuts

## Configuration Files

### FCL Files (in `CE/` directory)
FHICL configuration files for Mu2e simulation. These are used with the offline software (not this Python analysis) to generate simulated events.

## Future Improvements

Potential enhancements to this codebase:
- [ ] Configuration file support (YAML/TOML) replacing hardcoded cuts
- [ ] Systematic uncertainty propagation
- [ ] Alternative fitting backends (RooFit, scipy)
- [ ] Distributed processing (Spark, Dask)
- [ ] Real-time monitoring dashboard
- [ ] Unit tests and continuous integration

## Contact

For questions about this analysis, refer to the Mu2e collaboration documentation or contact the Run 1A analysis coordinators.

## License

This code is part of the Mu2e experiment software. Use according to Mu2e collaboration policies.
