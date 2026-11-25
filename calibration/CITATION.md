# Citing the calibration code

Suggested software reference, using the author's GitHub account name:

Mjgenoguin. (2026). *SWOT-DA: NSGA-II calibration of RRI parameters and river geometry* (source snapshots published 25 November 2026) [Source code]. GitHub. https://github.com/Mjgenoguin/SWOT-DA

For the submitted paper, replace the repository URL with the immutable calibration-directory URL for the commit used:

```text
https://github.com/Mjgenoguin/SWOT-DA/tree/<full-commit-sha>/calibration
```

Obtain that identifier with `git rev-parse HEAD` from the checked-out revision. The commit identifies both calibration drivers and their documentation; this publication does not assign a DOI.

Suggested availability sentence:

> The retained NSGA-II drivers for hydrological-parameter and river-profile calibration are available in SWOT-DA at the cited commit. The first-stage source is a recovered snapshot whose complete historical optimizer configuration could not be verified; execution requires separately obtained model inputs and observations.

Cite the underlying optimization method separately:

Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation, 6*(2), 182–197. https://doi.org/10.1109/4235.996017
