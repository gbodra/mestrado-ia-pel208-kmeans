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
    return scalar_multiply(1 / n, vector_sum(vectors))


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
        return min(range(self.k), key=lambda i: squared_distance(input, self.means[i]))

    def train(self, inputs):
        self.means = random.sample(inputs, self.k)
        assignments = None

        while True:
            new_assignments = list(map(self.classify, inputs))

            if assignments == new_assignments:
                return assignments

            assignments = new_assignments

            for i in range(self.k):
                i_points = [p for p, a in zip(inputs, assignments) if a == i]

                if i_points:
                    self.means[i] = vector_mean(i_points)


def test_class_examples():
    df = pd.read_csv("./data/aula.txt", header=None)
    df.columns = ["X", "Y"]

    random.seed(0)
    inputs = df.values.tolist()
    clusterer = KMeans(3)
    result = clusterer.train(inputs)
    print("Means: ", clusterer.means)

    plt.scatter(df["X"], df["Y"], c=result, marker="o")
    plt.scatter([x[0] for x in clusterer.means],
                [x[1] for x in clusterer.means], marker="x", c="red")

    plt.title("Dataset: Slides da Aula")
    plt.show()


def test_iris():
    df = pd.read_csv("./data/iris.txt", header=None)
    df.columns = ["Sepal_Lenght", "Sepal_Width", "Petal_Lenght", "Petal_Width", "Class"]
    df.drop(inplace=True, columns=["Class"])

    random.seed(0)
    inputs = df.values.tolist()
    clusterer = KMeans(3)
    result = clusterer.train(inputs)
    # Multiplica por 30 somente para melhorar a visualização no gráfico
    df["Petal_Width"] = df["Petal_Width"] * 30
    print("Means: ", clusterer.means)

    fig = plt.figure(figsize=[10, 10])
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(df["Sepal_Lenght"], df["Sepal_Width"], df["Petal_Lenght"],
               s=df["Petal_Width"], c=result, marker="o")
    ax.scatter([x[0] for x in clusterer.means],
               [x[1] for x in clusterer.means],
               [x[2] for x in clusterer.means],
               s=[x[3] * 30 for x in clusterer.means], marker="x", c="red")

    plt.title("Dataset: Iris")
    plt.show()


def test_haberman():
    df = pd.read_csv("./data/haberman.data", header=None)
    df.columns = ["Age", "Year_Operation", "Positive_Axilary_Nodes", "Class"]
    df.drop(inplace=True, columns=["Class"])

    random.seed(0)
    inputs = df.values.tolist()
    clusterer = KMeans(2)
    result = clusterer.train(inputs)
    print("Means: ", clusterer.means)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(df["Age"], df["Year_Operation"], df["Positive_Axilary_Nodes"],
               c=result, marker="o")
    ax.scatter([x[0] for x in clusterer.means],
               [x[1] for x in clusterer.means],
               [x[2] for x in clusterer.means], marker="x", c="red")

    plt.title("Dataset: Haberman")
    plt.show()


def test_container_crane():
    df = pd.read_csv("./data/Container_Crane_Controller_Data_Set.csv", sep=";")
    df.drop(inplace=True, columns=["Power"])

    random.seed(0)
    inputs = df.values.tolist()
    clusterer = KMeans(3)
    result = clusterer.train(inputs)
    print("Means: ", clusterer.means)

    plt.scatter(df["Speed"], df["Angle"], c=result, marker="o")
    plt.scatter([x[0] for x in clusterer.means],
                [x[1] for x in clusterer.means], marker="x", c="red")

    plt.title("Dataset: Container Crane")
    plt.show()


logging.basicConfig(stream=sys.stderr, level=logging.INFO)

test_class_examples()

test_iris()

test_haberman()

test_container_crane()
