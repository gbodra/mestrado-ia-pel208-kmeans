import logging
import sys
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
import random


def vector_add(v, w):
    return [v_i + w_i for v_i, w_i in zip(v, w)]


def vector_subtract(v, w):
    return [v_i - w_i for v_i, w_i in zip(v, w)]


def vector_sum(vectors):
    return reduce(vector_add, vectors)


def scalar_multiply(c, v):
    return [c * v_i for v_i in v]


def vector_mean(vectors):
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))


def dot(v, w):
    return sum(v_i * w_i for v_i, w_i in zip(v, w))


def sum_of_squares(v):
    return dot(v, v)


def squared_distance(v, w):
    return sum_of_squares(vector_subtract(v, w))


class KMeans:
    def __init__(self, k):
        self.k = k
        self.means = None

    def classify(self, input):
        return min(range(self.k), key = lambda i: squared_distance(input, self.means[i]))

    def train(self, inputs):
        self.means = random.sample(inputs, self.k)
        assignments = None

        while True:
            new_assignments = list(map(self.classify, inputs))

            if assignments == new_assignments:
                return

            assignments = new_assignments

            for i in range(self.k):
                i_points = [p for p, a in zip(inputs, assignments) if a == i]

                if i_points:
                    self.means[i] = vector_mean(i_points)


logging.basicConfig(stream=sys.stderr, level=logging.INFO)

df = pd.read_csv("./data/aula.txt", header=None)
df.columns = ["X", "Y"]

random.seed(0)
inputs = df.values.tolist()
clusterer = KMeans(3)
clusterer.train(inputs)
print("Means: ", clusterer.means)

plt.plot(df["X"], df["Y"], 'o')
plt.plot([x[0] for x in clusterer.means],
         [x[1] for x in clusterer.means], 'x')

plt.title("Dados brutos")
plt.show()
