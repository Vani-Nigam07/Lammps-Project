# autoMD: Autonomous Simulation Control via Multi-Agent Systems

An industrial-grade agentic framework built on the **Model Context Protocol (MCP)** that automates end-to-end molecular dynamics (MD) workflows in LAMMPS. The system replaces traditional, brittle trial-and-error simulation loops with a tool-using AI controller backed by a deterministic auditor agent.

---

## Architecture Overview

`autoMD` uses a decoupled multi-server architecture powered by `FastMCP` that communicates over a shared file system.

### 1. Pre-Processing Server

*  **Purpose:** Topology manipulation, structure preparation, and dynamic input generation.


* **Key Capabilities:** Parses LAMMPS data files, performs atom deletion/renumbering for custom pore geometries, and injects runtime parameters into simulation templates.


* **Supported Material Libraries:** Includes literature-grounded and first-principles Lennard-Jones parameters for:
*  **Graphene** (carbon-water interactions) 
*  **h-BN** (DREIDING-A force field parameterization) 
*  **$\text{MoS}_2$** (UFF parameters for metal sites + alternating hourglass chemistry) 
*  **$\text{Ti}_2\text{C}$ MXene** 


### 2. LAMMPS Runner Server


#### **Purpose:** Asynchronous simulation execution, hardware management, and process isolation.


*  **Key Capabilities:** Launches detached LAMMPS jobs. Auto-detects local hardware acceleration options (MPI parallelization on CPU or Kokkos GPU targets) to optimize node footprint.
*  **State Protection:** Operates with an independent **Auditor Agent** that intercepts tool commands to enforce hard safety boundaries (e.g., preventing timestep overreach, unphysical pressure ceilings, and simulation NaN/runaway instabilities).


### 3. Post-Processing Server

#### **Purpose:** Domain-specific calculations to evaluate reverse-osmosis (RO) performance.

* **Key Capabilities:** Streams/tails active simulation log files to parse thermodynamic streams, calculate real-time ion rejection rates , and compute total water flux curves ($L \cdot cm^{-2} \cdot day^{-1} \cdot MPa^{-1}$).



---

## Core Tool Pipeline

| Server | Core Tool | Functional Description |
| --- | --- | --- |
| **Pre-Processing** | `delete_atoms_and_rewrite` | Modifies active coordinates and re-maps consistent topologies.|
| | `reconstruct_full_filter` | Rebuilds symmetric filter blocks from piston architectures. |
| **LAMMPS Runner** | `start_lammps_detached` | Launches simulations asynchronously to decouple LLM token life from execution times.|
| | `get_lammps_status` | Tracks process IDs and scans outputs for duplication or failure loops. |
| **Post-Processing** | `desalination_water_flux` | Extracts water permeation metrics from underlying trajectories.|
| | `desalination_ion_rejection` | Measures ion leakage behavior on the downstream membrane side. |

---

## Production Deployment & Validation

- in progress

---

## Future Roadmap: Constraint-First Optimization

The project is scaling toward an autonomous discovery system for advanced membrane architectures. 
I want to a orchestrate a **Constraint-First Optimization** layer:

* **Hard-Boundary Evaluation:** Treats ion rejection percentage as a non-negotiable threshold parameter, rather than a soft weight in a loss function.

* **Two-Phase Search Strategy:**
1. **Space-Filling Exploration:** Initial broad sampling of the multi-material design space to identify stable operational baseline zones.

2. **Bandit/Bayesian Refinement:** Downstream optimization loops that maximize water flux within the validated structural safety bounds discovered in Phase 1.


