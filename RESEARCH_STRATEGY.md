# RamseyNumber Research Strategy

## Purpose

RamseyNumber is not built around the assumption that one search algorithm,
one neural network, or one heuristic will solve the search problem by itself.
The project is designed as an experimental framework in which independent
parts of the search process can be changed, measured, combined, and improved.

The central research principle is:

> When a method reaches a plateau, change one of the factors that determines
> the search landscape, the available moves, the information presented to the
> searcher, or the way the landscape is explored.

For the current `R(5,5)` work on `K_43`, the canonical goal remains exact:
minimize the number of monochromatic `K_5` subgraphs, ultimately reaching
score zero. All auxiliary objectives, learned models, heuristics, and search
methods exist to help reach that exact goal.

The major control surfaces are:

| Lever | Question we can change |
|---|---|
| Construction | Where does a search begin? |
| Representation | How is a coloring or state represented? |
| Encoding | What information is explicitly presented to an algorithm? |
| Embedding | How does a learned model internally represent that information? |
| Action | What transformations are permitted in one step? |
| Objective / scoring | What does "better" mean during search? |
| Reward | What feedback does a learner receive? |
| Memory | What past information influences future moves? |
| Policy | How is the next action selected? |
| Search | How are sequences and alternatives explored? |
| Model | What machinery learns a policy or representation? |
| Learning algorithm | How are model parameters updated? |
| Curriculum | Which states and difficulty levels are used for training? |
| Archive / population | Which discoveries are preserved and reused? |
| Symmetry | Which apparently different states are equivalent? |
| Diversity | Are we exploring genuinely different structures? |
| Computation | How much useful search can we afford? |
| Evaluation | How do we determine whether an idea actually helped? |

These dimensions should remain as independent and interchangeable as
practical. The strength of the project comes from being able to alter one of
them without rewriting the rest of the system.

## 1. Construction

Construction determines the initial state of a search episode.

Current and natural construction families include:

- random colorings;
- archived colorings;
- mixtures of construction methods;
- cyclic constructions;
- Exoo-style constructions;
- fixed or manually supplied colorings;
- perturbations of elite archived colorings;
- modular, residue, or difference-set constructions;
- recombinations of strong colorings;
- constructions designed to reproduce statistics of elite graphs;
- learned construction generators.

Construction asks:

> Where in the enormous coloring space should the search begin?

A sufficiently powerful local search can reduce the importance of seed
quality in easy score regimes. Seed structure is expected to matter more as
the search approaches increasingly difficult low-score regions.

## 2. Representation

For two-color `K_43`, the canonical coloring is the 903-edge binary coloring.
It should remain the source of truth.

Other representations are derived views that may make particular algorithms
more effective:

- a `43 x 43` symmetric color matrix;
- an edge list;
- vertex neighborhoods;
- clique-incidence structures;
- motif counts;
- local structural summaries;
- spectral representations;
- graph tensors.

Keeping a minimal canonical representation while allowing specialized derived
views prevents algorithm-specific representations from becoming the truth of
the system.

## 3. Encoding

Encoding determines which information is explicitly presented to a learned
model or other decision procedure.

The current neural encoding is intentionally sparse. Its pairwise channels
represent:

- edge color;
- diagonal position;
- action availability.

Possible future features include:

- red and blue degree of endpoint vertices;
- number of monochromatic `K_5`s containing an edge;
- number of nearly monochromatic `K_5`s containing an edge;
- triangle or other motif information;
- common-neighbor statistics;
- local danger values;
- tabu age;
- recent action history;
- changes from the previous state.

There is an important tradeoff. Too little explicit information forces a
model to rediscover inexpensive and obvious mathematics. Too much hand-built
information may bias the learner toward existing human heuristics and obscure
features it might otherwise discover for itself.

Encoding should therefore be treated as an experimental variable.

## 4. Embedding

Encoding is the information supplied to a neural model. Embedding is the
learned internal representation constructed from that information.

The current model projects a small explicit feature vector into a wider hidden
representation. Future experiments may vary or introduce:

- embedding width;
- edge embeddings;
- vertex embeddings;
- graph-level embeddings;
- structural embeddings;
- multiple interacting embedding spaces.

Learned embeddings are especially interesting as analysis tools. Low-scoring
colorings may form clusters in learned representation space corresponding to
structural classes for which we do not yet have mathematical names.

## 5. Action

Actions define the transformations available to a search algorithm. They
therefore define the connectivity and effective geometry of the search space.

The initial action vocabulary contains a single operation:

> Flip one edge color.

Future action families can include:

- single-edge flip;
- red/blue edge swap;
- arbitrary double-edge flip;
- adjacent double-edge flip;
- triangle modification;
- `K_5`-targeted modification;
- neighborhood perturbation;
- structured multi-edge moves;
- learned macro-actions.

This is a particularly important research direction near local minima. A
state that is separated from an improvement by several individually bad
single-edge flips may be one atomic action away when using a richer move set.

For example, swapping one red and one blue edge preserves the total number of
edges of each color while changing their arrangement. Although such a swap can
be expressed as two individual flips, a greedy single-flip search may reject
the first move because its intermediate state is worse. An atomic swap can
cross that barrier directly.

Mixed and hierarchical action policies are also possible. A policy may first
choose an action type and then choose the edges or vertices participating in
that action.

## 6. Objective and Scoring

The true Ramsey objective is fixed:

> The exact score is the number of forbidden monochromatic cliques.

For the current symmetric `R(5,5)` search, score zero is the definitive target.

Search algorithms need not, however, rank intermediate states using only that
single number. Useful auxiliary objectives may include:

- exact monochromatic count;
- weighted near-monochromatic counts;
- nonlinear danger penalties;
- concentration of violations;
- degree imbalance;
- motif distributions;
- diversity or novelty measures;
- adaptive combinations of objectives.

Two colorings can have the same exact Ramsey score while having very different
prospects for future improvement. A richer objective can distinguish those
states while exact scoring remains the final authority.

## 7. Reward

An objective measures a state. Reward tells a learner how to interpret a
transition between states.

Possible reward signals include:

- immediate exact-score reduction;
- danger-energy reduction;
- improvement over the best state seen in the episode;
- delayed multi-step improvement;
- terminal or milestone bonuses;
- plateau-escape rewards;
- novelty rewards;
- cycling penalties.

Reward design affects credit assignment: which earlier choices receive credit
or blame for improvements that appear later. This becomes increasingly
important when useful moves temporarily make the graph worse before enabling a
larger improvement.

## 8. Memory

Search memory is explicit algorithmic history, distinct from neural model
parameters.

Current tabu memory already uses recent actions and visited states. Future
memory strategies may include:

- fixed edge tenure;
- adaptive tenure;
- state tenure;
- recently modified vertex sets;
- repeated-motif avoidance;
- long-term action frequency;
- novelty memory;
- elite-state memory.

Short-term memory prevents immediate cycling. Long-term memory can encourage
the search to leave regions it has explored too heavily.

## 9. Policy

A policy answers:

> Given the current state and available actions, which action should be taken?

Policies can include:

- random;
- greedy;
- stochastic greedy;
- neural;
- epsilon-greedy;
- softmax sampling;
- simulated-annealing rules;
- hand-built heuristics;
- greedy/neural hybrids;
- ensembles;
- hierarchical policies.

Different score regimes may favor different policies. A hybrid policy could,
for example, use greedy selection while clearly productive local moves exist
and defer to a learned policy when the local landscape becomes ambiguous.

## 10. Search

A policy chooses a move. A search algorithm determines how possible futures
are explored across many moves.

Search strategies include:

- a single local-search trajectory;
- repeated restarts;
- beam search;
- tree search;
- Monte Carlo tree search;
- simulated annealing;
- evolutionary search;
- population search;
- large-neighborhood search.

This changes the question from:

> What is the best next action?

to potentially:

> Which sequence of actions leads to the best reachable state?

Multi-step reasoning becomes increasingly important when every immediate move
from a strong coloring looks neutral or harmful.

## 11. Neural Architecture

The current pairwise message-passing policy/value network is one architecture,
not an assumption of the project.

Possible model experiments include:

- wider hidden representations;
- deeper message passing;
- graph attention;
- graph neural networks with separate vertex and edge streams;
- transformer-style architectures;
- residual architectures;
- global graph representations;
- multiple policy heads;
- action-type heads;
- recurrent memory.

Architecture should evolve together with the information and action spaces.
Increasing model complexity before determining what the model must represent
or decide is unlikely to be efficient.

## 12. Learning Algorithm

PPO is one learning algorithm, not a synonym for reinforcement learning.

Other approaches worth supporting or testing include:

- PPO and other actor-critic methods;
- Q-learning and action-value methods;
- imitation learning;
- supervised learning from strong heuristics;
- offline reinforcement learning from archived trajectories;
- preference-based learning;
- evolutionary optimization of model parameters;
- curriculum learning;
- iterative self-improvement from archived discoveries.

The optimized greedy search can itself become a teacher. Large numbers of
`(state, greedy action)` examples could pretrain a neural model to understand
the behavior of the greedy heuristic. Reinforcement learning could then focus
on discovering when and how to depart from greedy behavior rather than
spending training capacity rediscovering it from scratch.

## 13. Curriculum

The project has already demonstrated the usefulness of dividing training by
difficulty.

Conceptually:

```text
Phase 1: random / high-score states -> competent local improvement
Phase 2: stronger archived states   -> lower-score specialization
Phase 3: greedy-plateau states      -> learn to escape local structure
```

Future curricula can use explicit score bands such as:

```text
1000-2000
600-1000
400-600
300-400
200-300
100-200
50-100
...
```

Score bands need not be the only measure of difficulty. A useful curriculum
may eventually use whether a particular baseline algorithm can improve the
state. A score-350 coloring that greedy solves easily is a different training
example from a score-350 coloring on which greedy is completely trapped.

## 14. Archive and Population

The persistent coloring archive is more than output storage. It can become a
dataset describing the search landscape.

Useful archived metadata can include:

- exact coloring;
- exact score;
- clique histogram;
- construction source;
- parent coloring;
- action sequence or trajectory identifier;
- search algorithm and policy;
- model checkpoint;
- runtime;
- structural statistics;
- whether a baseline method could improve the coloring;
- plateau depth or other difficulty measurements.

This enables questions such as:

> What distinguishes a coloring trapped near score 300 from one that can be
> pushed to 250?

Those archived examples can then become training data for subsequent models
and search methods.

## 15. Symmetry

Ramsey colorings contain large amounts of representational redundancy.

Relevant symmetries include:

- vertex relabeling;
- global color exchange;
- graph isomorphism;
- equivalent action transformations.

The current neural architecture already incorporates vertex-renaming symmetry
through permutation-equivariant computation. Additional opportunities include:

- symmetry-aware data augmentation;
- canonical graph labeling;
- symmetry-aware archive deduplication;
- symmetry-aware action generation.

Mathematically equivalent states should not consume unnecessary search or
learning capacity merely because they have different vertex names.

## 16. Population Diversity

Score diversity is not the same as structural diversity.

An archive containing hundreds of score-350 colorings may still represent only
one narrow family of structures. A strong research population should preserve
meaningfully different states, potentially measured by:

- coloring Hamming distance;
- motif distributions;
- local violation structure;
- construction ancestry;
- graph invariants;
- learned embedding distance.

Diversity becomes particularly important for training because a learner can
otherwise over-specialize to one family of low-scoring colorings.

## 17. Computation and Performance

Runtime optimization is part of the research strategy because computational
efficiency directly determines how many hypotheses can be tested.

Important performance areas include:

- incremental scoring;
- incremental action analysis;
- action generation;
- memory and allocation behavior;
- batching;
- CPU parallelism;
- GPU utilization;
- archive access;
- model inference cost.

An exact optimization that makes a search 10 or 20 times faster effectively
multiplies the experimental budget by the same factor. Performance work should
therefore be judged by how much additional useful search it enables, not merely
as software polish.

Optimizations must remain independently verifiable. Whenever an incremental or
cached calculation replaces a complete recomputation, tests should compare it
against the simple complete calculation over long mutation sequences.

## 18. Evaluation

Every search innovation needs a stable scientific referee.

New methods should be evaluated on fixed benchmark pools using consistent
budgets. A Phase 3 benchmark might contain a structurally diverse collection
of strong colorings and compare methods such as:

| Method | Mean best | Median best | Record | Success below target | Runtime |
|---|---:|---:|---:|---:|---:|
| Greedy + flip | | | | | |
| Greedy + swap | | | | | |
| Greedy + mixed actions | | | | | |
| Neural + flip | | | | | |
| Neural + mixed actions | | | | | |

Evaluation should distinguish at least:

- average performance;
- median performance;
- best discovered score;
- success rate at meaningful thresholds;
- improvement relative to the seed;
- runtime or compute budget;
- structural diversity of successes.

One spectacular run is a discovery, but it is not evidence that one method is
generally stronger than another.

## Research Loop

The project should continually repeat the following process:

1. Establish a reproducible baseline.
2. Identify the current performance plateau.
3. Choose one control surface to change.
4. Form a concrete hypothesis about why the change could help.
5. Benchmark the change against the same states and compute budget.
6. Preserve successful colorings and experiment metadata.
7. Keep improvements that reproduce.
8. Use the resulting stronger states to define the next problem regime.
9. Repeat.

The broad search program can be summarized as:

```text
Construction
    x Representation
    x Information / Encoding
    x Action
    x Objective / Reward
    x Memory
    x Policy
    x Search
    x Model / Learning
```

The archive closes the loop. Successful combinations produce stronger states,
and those stronger states become the starting material for the next generation
of methods.

The project therefore does not depend on discovering one brilliant algorithm.
Its goal is to build a framework in which assumptions can continually be
changed one at a time, results can be measured honestly, successful ideas can
be composed, and each new plateau can be attacked from a different angle.
