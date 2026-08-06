The upcoming refactor has several excellent targets:

Establish one naming policy: files, classes, methods, configs, results.

Define explicit interfaces/protocols for Construction, Action, Policy, Objective, Memory, Search, Archive, Experiment.

Identify dependency direction and eliminate backwards/circular dependencies.

Separate public API from private implementation (construct() versus _refresh_queue()).

Move reusable experiment orchestration out of notebooks.

Separate configuration from execution.

Standardize results/metrics so experiments are directly comparable.

Make experiment definitions declarative and tiny.

Keep specialized implementations interchangeable.