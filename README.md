RAPTOR is a first-principles, high-performance computing (HPC) digital twin designed for advanced electromagnetic (EM) modeling, synthetic aperture radar (SAR) imaging, cognitive electronic warfare (EW), and volumetric physics simulation.
The architecture dynamically scales from phenomenological Range-Doppler and SAR calculations to fully volumetric, FP64 GPU-accelerated Shooting and Bouncing Rays (SBR) and Finite-Difference Time-Domain (FDTD) plasma simulations.
#-# 🛠 Compilation & Deployment
RAPTOR requires a modern C++17 compiler, CUDA Toolkit (for FP64 BVH Raytracing), LibTorch (for Cognitive AI/Reinforcement Learning agents), and OpenMP/MPI for distributed memory scaling.

bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/path/to/libtorch
make -j$(nproc)

#-# ⚙️ Command Line Interface Reference
RAPTOR utilizes a comprehensive flag structure to control physical, mathematical, and environmental parameters.

#-# First-Principles EM Solvers
These parameters control the highest fidelity physics engines, bypassing phenomenological approximations.

--solver [analytical|fdtd]` : Defines the core computational engine. `analytical` utilizes rigorous propagation models. `fdtd` engages the 3D volumetric space-time solver (Maxwell's equations) across OpenMPI nodes.
--cole-cole [f]` : Path to a configuration file defining Multi-pole Dispersive Media, allowing for complex permittivity simulations over ultra-wideband (UWB) pulses.
--sar-bio [float]` : Computes and outputs the Specific Absorption Rate (SAR) in W/kg for biological targets within the FDTD volumetric mesh.

#-# Distributed HPC Architecture
--mpi-nodes [N]` : Sets the number of distributed nodes for the OpenMPI communicator. Crucial for scaling the FDTD mesh or massively parallel Monte Carlo iterations.
--shm-zero-copy` : Enables Shared Memory IPC. Bypasses network overhead for GPU-to-CPU DSP acceleration (optimized for unified memory architectures).
--precision [fp64|fp32|bf16]` : Controls the dynamic precision casting of the analytical engine. (Note: GPU SBR raytracing strictly enforces FP64 to prevent high-frequency phase drift).

#-# Hardware-In-The-Loop (HITL)
--vita49-stream [ip:port]` : Encapsulates IQ data into the IEEE VITA 49 Radio Transport (VRT) standard and streams it via UDP for Hardware-in-the-Loop (HITL) integration.
--s-param-import [f]` : Ingests a Touchstone `.s2p` file to map physical hardware frequency responses (insertion loss, phase delay) directly onto the simulated antenna array.
--dac-spur [dBc]` : Injects Spurious Free Dynamic Range (SFDR) harmonics into the transmission waveform.

#-# Advanced Theoretical Physics & 6G Models
--rx-rydberg` : Simulates a Rydberg Atomic Quantum Receiver. Negates standard $kTB$ thermal noise calculations and clamps the noise floor to near-absolute zero ($1\times 10^{-25}$).
--oam-mode [N]` : Imprints an Orbital Angular Momentum (OAM) helical phase front onto the transmission array, calculating element phases using $\phi = \text{mode} \times \arctan(y/x)$.
--cell-free-aps [N]` : Distributes the array elements randomly across a massive 10km grid to simulate 6G Cell-Free Massive MIMO access points. Overrides the standard `--array` parameter.
--target-plasma-cloak [N]` : Envelopes the target in an Active Plasma Meta-Stealth shell. Utilizes Drude-Lorentz integration in FDTD or complex permittivity boundary reflection equations in the analytical solver.
--acoustic-vibration [Hz]` : Induces localized high-frequency spatial jitter (Acousto-EM spoofing) on target scatterers to generate micro-Doppler artifacts.
--isac-data [bps]` : Encodes a QPSK/PSK RadCom bitstream directly into the radar waveform for Integrated Sensing and Communications (ISAC).
--starlink-illuminate` : Simulates a passive bistatic radar setup utilizing a constellation of 10 moving LEO nodes as illuminators of opportunity.
--quantum-entangled [dB]` : Simulates Quantum Illumination (Quantum Chernoff Bound). Suppresses the effective thermal noise and jammer floor by the specified decibel amount.

#-# Advanced Mathematics & Signal Processing
--pa-memory [taps]` : Simulates Non-Linear Power Amplifier (PA) memory effects using a localized Volterra series history buffer.
--stap-smi [snapshots]` : Enables Space-Time Adaptive Processing (STAP) using Sample Matrix Inversion. Specifies the number of fast-time snapshots used to estimate the adaptive spatial covariance matrix.
--frft [order]` : Replaces the standard FFT in Range-Doppler processing with a Fractional Fourier Transform (FrFT), utilized for focusing accelerating targets.
--deramp` : Enables stretch processing (deramp on receive) instead of matched filtering.
--pfa-algo [t]` : Selects the SAR imaging algorithm (e.g., `pfa` for Polar Format Algorithm, `bp` for exact Time-Domain Backprojection).

#-# Hardware Imperfections & Rigorous Propagation
--amp-err [dB]` / `--phase-err [deg]` : Standard deviation of zero-mean Gaussian amplitude and phase errors applied per-element, per-pulse.
--subarray [N]` : Partitions the total array into $N$ subarrays for analog combining before digital sampling.
--adc-jitter [ps]` : Standard deviation of ADC aperture timing jitter, calculated dynamically into high-frequency phase noise based on the instantaneous shifted carrier frequency.
--thermal-drift [K/s]` : Injects phase drift across the antenna array proportional to elapsed simulation time and element index, simulating uneven array heating.
--allan-variance [ppb]` : Simulates local oscillator Random Walk phase noise over the Coherent Processing Interval (CPI).
--rx-recover [us]` : Blanking time. Eclipses target returns that arrive faster than the T/R switch recovery time.
--lo-leakage [dBm]` : Injects a static DC phase/amplitude offset into every receiver channel.
--multipath-diffuse [v]` : Introduces a random phase variance to the secondary (bounce) path in ground-reflection calculations.
--creeping-wave` : Simulates EM diffraction around the shadow boundary of spherical target nodes.

#-# Metamaterials & Artificial Intelligence
--ris [x:y:z]` / `--ris-elements [N]` : Positions a Reconfigurable Intelligent Surface (RIS) in the environment to act as an active multipath relay.
--ttd` : Enables True Time Delay lines instead of phase shifters to eliminate spatial beam squint in wideband signals.
--conformal-array [f]` : Loads a custom 3D element geometry file, overriding linear/planar array constraints.
--sbr [bounces]` : **GPU Accelerated.** Engages the FP64 CUDA Shooting and Bouncing Rays engine. Raytraces a highly dense 100x100 grid onto the target's rotating local coordinate system for exact multipath delay/RCS mapping.
  
--ram-tensor [f]` : Ingests a frequency-dependent Radar Absorbent Material (RAM) table to dynamically lower target RCS as a function of the instantaneous waveform frequency.
--atr-model [f]` : Loads a LibTorch/ONNX neural network `.pt`/`.onnx` file to perform real-time Automatic Target Recognition (ATR) classification on the generated RDM or SAR images.
--rl-adapt` : Engages the LibTorch Reinforcement Learning Agent (`cognitive_agent.hpp`) to autonomously mutate PRF and Bandwidth in response to detected jamming in the spectrum history.
--gym-env` : Wraps the simulation space in an OpenAI Gym / PettingZoo compliant API for external reinforcement learning training.

#-# Electronic Warfare (EW) & Decoys
--jam [az][dB]` : Azimuth and Jammer-to-Signal (J/S) power ratio.
--jam-type [type]` : Modulates the jammer (e.g., `noise`, `gan` - simulated adversarial learning).
--jam-freq [Hz]` : Frequency offset for deceptive jamming.
--drfm-latency [ns]` : Forces a nanosecond processing delay on the adversarial jammer, potentially exposing the true target skin return on the leading edge.
--drfm-bits [N]` : Bit depth of the adversarial Digital Radio Frequency Memory (DRFM) quantizer. Lower bits cause severe quantization harmonic jamming.
--jam-pol [type]` : Polarization vector of the adversarial jamming signal.
--chaff-dens [v]` / `--chaff-fall [v]` : Spawns a volumetric cloud of scatterers that slowly descents in the Z-axis, creating slow-moving, high-RCS clutter.
--crosseye-base [m]` : Spawns two massive artificial scatterers with inverse phase ($\pm 10.0$ RCS) separated by baseline $m$ on the target to break monopulse trackers.
--tow-decoy [m]` : Spawns an artificial high-RCS sphere trailing the target by $m$ meters along its negative velocity vector.
--lpi-opt` : Enables Low Probability of Intercept logic, reducing transmit power by 50% to minimize the intercept footprint.

#-# Target Physics & Kinematics
--target [type]` : Basic point-scatterer models (`f16`, `human`).
--rcs-file [f]` : Ingests a CSV/TXT list of predefined scatterer points and baseline RCS values.
--pos x:y:z` / `--vel x:y:z` : Target initial Cartesian position and velocity vectors.
--yaw-rate`, `--pitch-rate`, `--roll-rate [d/s]` : Degrees per second dynamic target rotation (Critical for SBR and ISAR modes).
--jem [rpm]` : Jet Engine Modulation micro-Doppler RPM rate.
--gait-vel [m/s]` : Human walking modulation velocity.
--dielectric [v]` : Target dielectric constant (scales internal reflectivity).
--swerling [0-4]` : Swerling Case 0-4 target fluctuation models (uses Exponential and Gamma distributions).
--drone-rotors [N]` : Automatically builds $N$ micro-Doppler rotor scatterers around the target center.
--rcs-bistatic` : Evaluates bistatic geometric scaling ($\cos(\beta/2)$).
--cavity-return [m]` : Adds a delayed, high-RCS scattering node receding from the target to simulate jet engine inlet cavity ring-down effects.
--blade-flex [mm]` : Adds aeroelastic non-linear sine-wave modulations to the $Z$-axis of rotor scatterers.
--glint` : Adds violent center-of-mass spatial jitter ($\pm 2.0$m) to the target to simulate complex angular tracking glint.
--exhaust-plume [K]` : Adds randomized micro-scatterers trailing the target velocity vector to simulate ionized exhaust reflections.
--wake-vortex [m]` : Simulates turbulent trailing wake radar returns.
--hypersonic [mach]` : Simulates dynamic plasma blackout based on atmospheric friction variables.

#-# Environmental & Atmospheric Modeling
--ducting`, `--duct-ht [m]` : Enables atmospheric evaporation ducting limits.
--rain [mm/h]`, `--turb [rad]`, `--plasma [dens]`, `--iono [S4]` : Core attenuation and phase-scrambling atmospheric factors.
--rain-volumetric` : Defines 3D spatial boundaries for the `--rain` attenuation metric (limits storm cell size).
--multipath` : Assumes a perfectly conducting flat-earth bounce path.
--scintillation [S4]` : Ionospheric fading depth, modulating path loss per pulse based on an S4 index.
--veg-atten [dB/m]`, `--dust-storm [vis]`, `--snow-rate [mm/h]`, `--aurora-abs [dB]` : Volumetric attenuation factors integrated into the two-way radar equation.

#-# Clutter Models
--clutter-dist [t]` : Statistical model for sea/land clutter (`rayleigh`, `weibull`, `lognormal`).
--sea-state [0-9]` : Scales clutter amplitude logarithmically based on Douglas Sea State.
--clutter-vel [v]` : Base radial velocity of moving clutter.
--clutter-tex [v]` : Temporal correlation/texture factor applied via recursive filtering to previous pulses.
--surface-roughness [cm]` : Computes the Beckmann physical clutter model for grazing angle sea/land reflections.
--wind-shear [m/s/km]` : Introduces an altitude-dependent Doppler gradient to the clutter model.
--bio-clutter [dens]` : Injects a volumetric swarm of point targets with randomized flapping modulation (simulating bats or birds).

#-# Antenna & Array Configuration
--array [N]` : Number of phased array elements.
--spacing [m]` : Distance between array elements.
--mvdr` : Minimum Variance Distortionless Response adaptive beamforming.
--stap` : Space-Time Adaptive Processing (uses spatial approximation if SMI is not set).
--insar [m]` : Sets up a secondary vertically displaced receiving channel to measure interferometric phase gradients.
--pol [HH|VV]` : Primary transmission polarization.
--pol-mode [std|full]` : `full` tracks cross-pol (HV/VH) scattering matrices alongside co-pol.
--array-fail [0-1]` : Probability of individual element dead-state failure at initialization.
--elem-pattern [t]` : Element factor pattern applied (e.g., `cos` applies a $\cos(\theta)$ roll-off).
--mutual-coupling` : Configures inter-element EM coupling matrix $C$.

#-# Radar Parameters
--fc [Hz]` : Carrier Frequency (Default: 10 GHz).
--bw [Hz]` : Bandwidth (Default: 1 MHz).
--prf [Hz]` : Pulse Repetition Frequency (Default: 1000 Hz).
--pulses [N]` : Pulses per Coherent Processing Interval (CPI).
--bits [N]` : ADC Bit depth (Triggers the `quantize()` routine).
--n-fft [N]` : Base fast-time samples (Must be power of 2 for FFTs).
--tx-pwr [W]` : Transmit power in Watts.
--gain [dBi]` : Antenna isotropic gain.
--nf [dB]` : Receiver Noise Figure.
--loss [dB]` : Systematic hardware losses.
--temp [K]` : System thermal noise temperature.
--pw [s]` : Pulse Width.
--adc-dnl [LSB]` / `--adc-inl [LSB]` : ADC Differential and Integral Non-Linearity errors.
--pn-slope [dB/dec]` : Phase noise slope.
--cryo` : Assumes 4K cryogenic cooling, overriding system temperature.
--gas-atten` : Enables atmospheric O2/H2O absorption modeling.

#-# Platform & Geometry
--tx x:y:z` / `--rx x:y:z` : Global Cartesian coordinates of the transmitter and receiver.
--radar-vel x:y:z` : Velocity vector of the radar platform.
--orbit [km]` : Hardcodes Tx/Rx positions and calculates exact Keplerian velocity for a LEO altitude.
--vibration [g]` : Platform vibration RMS applied to phase center.
--earth-k [val]` : 4/3 Earth radius equivalent factor for horizon calculations.
--space-debris [N]` : Spawns orbital micro-scatterers moving at Keplerian velocities.

#-# Waveform Generation
--wave [type]` : Selects base waveform (`lfm`, `ofdm`).
--poly-code [type]` : Applies polyphase coding (`barker`, `frank`).
--rfi [type]` : Adds deterministic RFI (`lte`, `wifi`) to specific frequency bins.
--illuminator [f]` : Defines a static passive source.
--nlfm [alpha]` : Non-Linear FM chirp shaping coefficient.
--quantum` : Transmits an entangled Gaussian noise sequence rather than a deterministic pulse.
--freq-hop [type]` : Strategy for frequency agility. Supports `random` (within 100MHz) or `costas` (deterministic Costas array logic).

#-# Detection & Tracking
--pfa [float]` : Desired Probability of False Alarm for CFAR algorithms.
--cfar-type [str]` : Constant False Alarm Rate algorithm (`ca` = Cell Averaging, `go` = Greatest Of, `so` = Smallest Of, `os` = Ordered Statistic).
--cfar-guard [N]` / `--cfar-ref [N]` : Number of CFAR guard and reference cells.
--mti-order [N]` : Moving Target Indication delay-line canceller order.
--tracker [type]` : Engages backend tracker (`kalman` or `ab`).
--tws-revisit [ms]` : Track-While-Scan revisit interval; restricts Kalman filter measurement updates to this specific temporal cadence.

#-# Modes & Algorithms
--mode [isar|rdm]` : Visualizer control. `rdm` yields Range-Doppler matrices; `isar` routes to SAR backprojection.
--algo [bp|csa]` : SAR Algorithm (Backprojection vs. Chirp Scaling).
--tbd` : Track-Before-Detect logic processing (bypass hard CFAR thresholds).
--blind-zones` : Enforces strict Pulse Doppler eclipsing.
--pc` : Pulse Compression (Matched Filter implementation).
--sigint` : Mutes the primary transmitter to passively listen to environment sources (Jammers/Illuminators).

#-# Input/Output & Data Streaming
--res [N]` : Target resolution of the output SAR/RDM image grid.
--span [m]` : Spatial width of the imaging grid.
--dyn [dB]` : Dynamic range floor for the CLI visualizer output.
--format [csv|bin]` : Output file format.
--out [f]` : Path for the output file.
--udp-host [ip]` / `--udp-port [p]` : Target for real-time IQ UDP streaming.
--meta-file [f]` : Exports simulation parameters to JSON.
--gpu` : Generic flag to enable/enforce GPU pipelines (SBR inherently requires this).# Cpp---Advanced-Radar-Simulator

Advanced CUDA enabled Radar Simulator.  Complete CLI and ASCII.  It's WIP
