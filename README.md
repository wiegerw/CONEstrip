# The CONEstrip library
This repository contains a Python framework for reasoning with uncertainty.
This is joint work with Erik Quaeghebeur. N.B. This is ongoing work, so not
everything is finished yet.

## The CONEstrip algorithm
In the paper [The CONEstrip algorithm](https://doi.org/10.1007/978-3-642-33042-1_6)
(SMPS 2012), an algorithm is defined that determines whether a finitely generated
general convex cone contains the origin.
> Uncertainty models such as sets of desirable gambles and (conditional)
lower previsions can be represented as convex cones. Checking the consistency of and
drawing inferences from such models requires solving feasibility and optimization
problems. We consider finitely generated such models. For closed cones, we can use
linear programming; for conditional lower prevision-based cones, there is an efficient
algorithm using an iteration of linear programs. We present an efficient algorithm for
general cones that also uses an iteration of linear programs.

## The Propositional CONEstrip algorithm
In the paper [A propositional CONEstrip algorithm](https://doi.org/10.1007/978-3-319-08852-5_48)
(IPMU 2014) a generalization of the CONEstrip algorithm is presented.

>We present a variant of the CONEstrip algorithm for checking whether the origin lies in a finitely generated convex cone that can be
open, closed, or neither. This variant is designed to deal efficiently with
problems where the rays defining the cone are specified as linear combinations of propositional sentences. The variant differs from the original
algorithm in that we apply row generation techniques. The generator
problem is WPMaxSAT, an optimization variant of SAT; both can be
solved with specialized solvers or integer linear programming techniques.
We additionally show how optimization problems over the cone can be
solved by using our propositional CONEstrip algorithm as a preprocessor.
The algorithm is designed to support consistency and inference computations within the theory of sets of desirable gambles. We also make a
link to similar computations in probabilistic logic, conditional probability
assessments, and imprecise probability theory.

## Implementation

In the CONEstrip algorithm several linear programming problems need to be solved.
This has been implemented using [Z3](https://github.com/Z3Prover/z3).
Note that it is also possible to solve them with a library like
[cddlib](https://people.inf.ethz.ch/fukudak/cdd_home/).

The propositional CONEstrip algorithm is defined in terms of WPMaxSAT problems. We
note that WPMaxSAT problems can be conveniently expressed using the propositional
logic of Z3. Hence, also this algorithm has been implemented using
[Z3](https://github.com/Z3Prover/z3).

## Documentation

A detailed specification of the implementation can be found in [CONEstrip.pdf](https://github.com/wiegerw/CONEstrip/blob/main/doc/CONEstrip.pdf).
Note that this is not a tutorial, but rather a precise description of the algorithms
that were implemented.

## Installation

The code is available as the PyPI package [CONEstrip](https://pypi.org/project/CONEstrip/).
It can be installed using

```
pip install CONEstrip
```

## Licensing

The code is available under the [Boost Software License 1.0](http://www.boost.org/LICENSE_1_0.txt).
A [local copy](https://github.com/wiegerw/CONEstrip/blob/main/LICENSE) is included in the repository.

For testing the [Toy Parser Generator](https://github.com/CDSoft/tpg) package is used.
This package is available under the
[GNU Lesser General Public License v2.1](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html) license.
A [local copy](https://github.com/wiegerw/CONEstrip/blob/main/LGPL-2.1.txt) is included in the repository.
