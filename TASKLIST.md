
### Release Notes:
* v0.0.1
  * dynamic version
  * fixes to packaging and dependencies
  * upgrade IO support to xarray 2023.9
  * clean-up of main script `CS_calculations.py` (exposed as a cli)

### Task List:
  * **Documentation** -- comlete coverage
  * **Unit testing** -- increase coverage
  * Add scripts, CI programmes and notebook examples

  * IO
    * ~~ UM ~~
    * MONC
    * PTerodaCTILES ??
    * ??

  * Geometry
    * Integrate staggered and non-staggered grid interfaces
    * Implement gradients as methods on grids

  * Physics
    * Integrate with [MONC_utils](https://github.com/ParaConUK/monc_utils)
    * Integrate  SGS Fluxes analysis from P. Burns

  * Analysis tools
    * Add spectra calculation
    * Add Histograms/PDFs and violin calculators
    * Add tensor invariant calculators
    * Add quadrant analysis support

  * SGS
    * Add a mixed and nonlinear model extensions
    * Generalise dynamic coefficient calculations

      * Optimize simultaneous calculation at multiple scales or contraction dims
