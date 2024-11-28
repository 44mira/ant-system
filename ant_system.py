# Legolas Tyrael B. Lada
# 2022-04734 | 3 - BSCS

from typing import TypeAlias, TypeVar
from collections import namedtuple
from functools import total_ordering
from random import randint

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
        start_node: int | None = None,
        *,
        dummy: bool = False,
    ) -> None:
        """
        :param start_node: The node where the ant starts pathfinding
        :param dummy: Whether the Ant is just made for the initial comparison
        """
        if not dummy and start_node:
            self.path: list[int] = [start_node]
            self.to_visit: set[int] = set(range(len(input_data))) - set(
                self.path
            )
            self.tour_length: float = self.__calculate_tour()
            return

        self.tour_length = 1000  # for initial comparison

    def __calculate_tour(self) -> float:
        while self.to_visit:
            best: City = City(-1, -1.0)  # dummy values
            denominator: float = sum(
                self.__calculate_term(n) for n in self.to_visit
            )

            # find the best next city
            for next_city in self.to_visit:
                numerator: float = self.__calculate_term(next_city)

                if (numerator / denominator) > best.probability:
                    best = City(next_city, numerator / denominator)

            # move to the next city
            self.to_visit.remove(best.city)
            self.path.append(best.city)

        return sum(
            input_data[self.path[i]][self.path[i + 1]]
            for i in range(dimensions - 1)
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
    starting_cities = set()

    for i in range(ant_count):
        # start at a vacant city
        while True:
            starting_city = randint(0, 4)
            if (
                starting_city not in starting_cities  # if vacant city
                or len(starting_cities) == dimensions  # or if no vacant cities
            ):
                starting_cities.add(starting_city)
                break

        current_ant = Ant(starting_city)

    return Ant(dummy=True)


# }}}


def main():
    global_best: Ant = Ant(dummy=True)
    for i in range(iterations):
        global_best = min(iteration(i), global_best)

    assert global_best.path, "global_best path should not be None"


if __name__ == "__main__":
    main()
