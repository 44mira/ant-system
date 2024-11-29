# Legolas Tyrael B. Lada
# 2022-04734 | 3 - BSCS

from typing import TypeAlias, TypeVar
from collections import namedtuple
from functools import total_ordering
from random import randint
from sys import argv

# [[ Variables ]] {{{

T = TypeVar("T")
matrix: TypeAlias = list[list[T]]
City = namedtuple("City", ["city", "probability"])

iterations: int = 1

# delta
input_data: matrix[int] = [
    [0, 5, 7, 2, 1],  # 1
    [5, 0, 8, 5, 3],  # 2
    [7, 8, 0, 10, 3],  # 3
    [2, 5, 10, 0, 4],  # 4
    [1, 3, 3, 4, 0],  # 5
]
dimensions: int = len(input_data)

ant_count: int = 3
tau: matrix[float] = [[1] * dimensions for _ in range(dimensions)]

# special instruction for determining the alpha value
alpha = 0.71
beta = 2

# }}}

# [[ Ant Class ]] {{{


@total_ordering
class Ant:
    def __init__(
        self,
        start_node: int = -1,
        *,
        dummy: bool = False,
    ) -> None:
        """
        Ant object representing the state of an individual ant in the system.

        :param start_node: The node where the ant starts pathfinding
        :param dummy: Whether the Ant is just made for the initial comparison

        :attr path: the path taken by the ant
        :attr __to_visit: private set for determining the path,
                          also known as set J_k
        :attr tour_length: the length of the tour
        :attr path_pairs: the path but in pairs, for ease of checking in
                          pheromone updating
        """
        if not dummy and start_node >= 0:
            self.path: list[int] = [start_node]
            self.__to_visit: set[int] = set(range(len(input_data))) - set(
                self.path
            )

            # this solves for tour_length and sets the final path
            self.tour_length: float = self.__calculate_tour()

            # for pheromone updating
            self.path_pairs: set[tuple[int, int]] = {
                (self.path[i], self.path[i + 1]) for i in range(dimensions)
            }
            return

        self.tour_length = 1000  # for initial comparison

    def __calculate_tour(self) -> float:
        # state transition rule
        while self.__to_visit:
            best: City = City(-1, -1.0)  # dummy values
            denominator: float = sum(
                self.__calculate_term(n) for n in self.__to_visit
            )

            # find the best next city
            for next_city in self.__to_visit:
                numerator: float = self.__calculate_term(next_city)

                if (numerator / denominator) > best.probability:
                    best = City(next_city, numerator / denominator)

            # move to the next city
            self.__to_visit.remove(best.city)
            self.path.append(best.city)

        # return back to the original city
        self.path.append(self.path[0])

        # sum all of the path pairss
        return sum(
            input_data[self.path[i]][self.path[i + 1]]
            for i in range(dimensions)
        )

    def __calculate_term(self, next_city: int) -> float:
        """
        This solves for the `tau(r, n) * eta(r, n) ^ beta`

        Recall that `eta(r, n)` is just `1 / delta(r, n)`

        :param next_city: the next city to go to from current_city
        :return: the result of the term
        """
        current_city: int = self.path[-1]
        eta: float = 1 / input_data[current_city][next_city]

        return tau[current_city][next_city] * eta**beta

    def __lt__(self, value: object, /) -> bool:
        assert isinstance(value, Ant), "comparison only works for Ant to Ant"
        return self.tour_length < value.tour_length

    def __eq__(self, value: object, /) -> bool:
        assert isinstance(value, Ant), "comparison only works for Ant to Ant"
        return self.tour_length == value.tour_length

    def __repr__(self) -> str:
        display_path = [city + 1 for city in self.path]
        return f"{display_path} : {self.tour_length}"


# }}}

# [[ Functions ]] {{{


def iteration(n: int, /) -> Ant:
    """
    A single iteration of the Ant System algorithm.

    :param n: the iteration count
    :return: the best Ant of the iteration
    """

    # hashset to ensure to repetition of starting city unless
    # all cities are already occupied
    starting_cities: set[int] = set()
    ants: list[Ant] = []

    # the ants go on their tours
    for _ in range(ant_count):
        # start at a vacant city
        while True:
            starting_city = randint(0, 4)
            if (
                starting_city not in starting_cities  # if vacant city
                or len(starting_cities) == dimensions  # or if no vacant cities
            ):
                starting_cities.add(starting_city)
                break
        ants.append(Ant(starting_city))

    # the ants leave behind their pheromones
    # (global updating rule)
    for m in range(dimensions):
        for n in range(dimensions):
            # rate at which the pheromone is decayed from the path
            first_term = (1 - alpha) * tau[m][n]

            # sum all pheromones deposited by the ants on their tour
            second_term = sum(
                1 / input_data[m][n]
                for ant in ants
                if (m, n) in ant.path_pairs
            )

            tau[m][n] = first_term + second_term

    return min(ants)


# }}}


def main():
    global_best: Ant = Ant(dummy=True)
    for i in range(iterations):
        current = iteration(i)
        print(f"Iteration {i+1} best: {current}")
        global_best = min(current, global_best)

    assert global_best.path, "global_best path should not be None"

    print("---" * 15)
    print(f"{'TEST PATH': <16}: BEST TOUR LENGTH")
    print(global_best)


if __name__ == "__main__":
    if len(argv) == 2:
        iterations = int(argv[1])
    main()
