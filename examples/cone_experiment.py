# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import argparse
import json

from pathlib import Path

from conestrip.cones import print_gamble, linear_combination, parse_general_cone, parse_gamble
from conestrip.random_cones import add_random_border_cones, random_border_point, random_inside_point, random_general_cone
from conestrip.conestrip import is_in_general_cone, is_in_cone_generator, is_in_cone_generator_border, random_between_point, simplified_linear_combination, conestrip1_solution, conestrip2_solution, conestrip3_solution, conestrip_algorithm
from conestrip.conestrip_cdd import conestrip_cdd_algorithm
from conestrip.utility import StopWatch


def print_experiment(data):
    cone_size = data["cone-size"]
    gamble_size = data["gamble-size"]
    generator_size = data["generator-size"]
    coordinate_bound = data["coordinate-bound"]
    border_count = data["border-count"]
    print(f"cone-size: {cone_size}")
    print(f"gamble-size: {gamble_size}")
    print(f"generator-size: {generator_size}")
    print(f"coordinate-bound: {coordinate_bound}")
    print(f"border-count: {border_count}")

    cone = parse_general_cone(data["extended-cone"])

    experiments = data["experiments"]
    for experiment in experiments:
        r = cone.generators[experiment["cone-index"]]
        r_parent = cone.generators[experiment["parent-index"]]
        # facet_index = data["parent-facet-index"]
        print('r_parent =\n', r_parent, '\n')
        print('r =\n', r, '\n')

        x1 = parse_gamble(experiment["x1"])
        x2 = parse_gamble(experiment["x2"])
        x3 = parse_gamble(experiment["x3"])
        lambda1 = parse_gamble(experiment["lambda1"])
        lambda2 = parse_gamble(experiment["lambda2"])
        lambda3 = parse_gamble(experiment["lambda3"])

        print('x1 =', print_gamble(x1), 'lambda1 =', print_gamble(lambda1))
        print('x2 =', print_gamble(x2), 'lambda2 =', print_gamble(lambda2))
        print('x3 =', print_gamble(x3), 'lambda3 =', print_gamble(lambda3))
        print()

        for exp, t in experiment["timings"]:
            print(f'{exp}: {t}')


def run_experiment(cone_size, generator_size, gamble_size, coordinate_bound, border_count):
    data = {
        "cone-size": cone_size,
        "gamble-size": gamble_size,
        "generator-size": generator_size,
        "coordinate-bound": coordinate_bound,
        "border-count": border_count
    }

    R = random_general_cone(cone_size, gamble_size, generator_size, coordinate_bound)
    add_random_border_cones(R, border_count, False)
    data["extended-cone"] = str(R)

    experiments = []
    experiment = {}
    for index, r in enumerate(R.generators):
        if not r.parent:
            continue

        # Run an experiment with cone generator r and its parent
        r_parent, facet_index = r.parent
        r_parent_index = R.generators.index(r_parent)
        experiment["cone-index"] = index
        experiment["parent-index"] = r_parent_index
        experiment["parent-facet-index"] = facet_index

        x1, lambda1 = random_inside_point(r)
        x2, lambda2 = random_border_point(r)
        x3, lambda3 = random_between_point(r)

        experiment["x1"] = print_gamble(x1)  # N.B. json does not support writing decimals
        experiment["x2"] = print_gamble(x2)
        experiment["x3"] = print_gamble(x3)
        experiment["lambda1"] = print_gamble(lambda1)
        experiment["lambda2"] = print_gamble(lambda2)
        experiment["lambda3"] = print_gamble(lambda3)

        assert x1 == linear_combination(lambda1, r.vertices)
        assert x2 == linear_combination(lambda2, r.vertices)
        assert x3 == simplified_linear_combination(lambda3, r_parent.vertices)

        timings = []

        watch = StopWatch()
        assert is_in_cone_generator(r, x1)
        timings.append(('is_in_cone_generator(r, x1)', watch.seconds()))

        watch.restart()
        assert is_in_cone_generator_border(r, x2)
        timings.append(('is_in_cone_generator_border(r, x2)', watch.seconds()))

        watch.restart()
        assert not is_in_cone_generator(r, x3)
        timings.append(('is_in_cone_generator(r, x3)', watch.seconds()))

        watch.restart()
        assert not is_in_cone_generator(r, x3, with_border=True)
        timings.append(('is_in_cone_generator(r, x3, with_border=True)', watch.seconds()))

        watch.restart()
        assert is_in_cone_generator(r_parent, x3)
        timings.append(('is_in_cone_generator(r_parent, x3)', watch.seconds()))

        watch.restart()
        assert is_in_general_cone(R, x1, solver=conestrip1_solution)
        timings.append(('is_in_general_cone(R, x1, solver=conestrip1_solution)', watch.seconds()))

        watch.restart()
        assert is_in_general_cone(R, x1, solver=conestrip2_solution)
        timings.append(('is_in_general_cone(R, x1, solver=conestrip2_solution)', watch.seconds()))

        watch.restart()
        assert is_in_general_cone(R, x1, solver=conestrip3_solution)
        timings.append(('is_in_general_cone(R, x1, solver=conestrip1_solution)', watch.seconds()))

        watch.restart()
        assert is_in_general_cone(R, x1, solver=conestrip_algorithm)
        timings.append(('is_in_general_cone(R, x1, solver=conestrip_algorithm)', watch.seconds()))

        watch.restart()
        assert is_in_general_cone(R, x1, solver=conestrip_cdd_algorithm)
        timings.append(('is_in_general_cone(R, x1, solver=conestrip_cdd_algorithm)', watch.seconds()))

        experiment['timings'] = timings
        experiments.append(experiment)

    data["experiments"] = experiments
    print_experiment(data)
    return data


def main():
    cmdline_parser = argparse.ArgumentParser()
    cmdline_parser.add_argument('--cone-size', type=int, default=1, help='the number of cone generators in the initial general cone')
    cmdline_parser.add_argument('--generator-size', type=int, default=3, help='the number of gambles in the cone generators of the initial general cone')
    cmdline_parser.add_argument('--gamble-size', type=int, default=3, help='the number of elements of the gambles in the initial general cone')
    cmdline_parser.add_argument('--coordinate-bound', type=int, default=10, help='the maximum absolute value of the coordinates')
    cmdline_parser.add_argument('--border-count', type=int, default=10, help='the number of border facets that is added to the initial general cone')
    cmdline_parser.add_argument('--count', type=int, default=1000, help='the number of times the experiment is repeated')
    cmdline_parser.add_argument('--output', '-o', metavar='FILE', default='cone-experiment.json', help='the output file')
    args = cmdline_parser.parse_args()
    runs = [run_experiment(args.cone_size, args.generator_size, args.gamble_size, args.coordinate_bound, args.border_count) for _ in range(args.count)]

    output_file = args.output
    text = json.dumps(runs)
    Path(output_file).write_text(text)
    print(f'Saved output in file {args.output}')


if __name__ == '__main__':
    main()
