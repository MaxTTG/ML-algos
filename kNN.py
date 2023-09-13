import random
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mpl_toolkits import mplot3d
import random
import copy


# 1) Написать функцию для создания n кластеров
#
# В функцию передаётся:
# center[cluster_num][dims] - [x,y] центры кластеров по измерениям
# dims                      - размерность пространства
# sigma[cluster_num][dims]  - массив сигм по каждому измерению
# n                         - количество точек
# cluster_num               - количество кластеров
#
# На выходе молучаем массив arr[cluster_num][n][dims]


def MakeCluster(center, dims, sigma, cluster_num, n):
    arr = np.zeros((cluster_num, n, dims))
    for i in range(cluster_num):
        for j in range(n):
            for k in range(dims):
                arr[i, j, k] = np.random.normal(center[i, k], sigma[i, k])
    return arr


"""
Проверка работы MakeClusters

dots = MakeCluster(centers, dim, sigmas, cluster_num, 100)
plt.plot(dots[0, :, 0], dots[0, :, 1], 'o')
plt.plot(dots[1, :, 0], dots[1, :, 1], 'o')
plt.show()
"""


# 2) Написать функцию, реализующую метод k-средних.
# Реализовать функцию искажения J для оценки качества кластеризации.
#
# k - количество кластеров
# dim - количество измерений
# points[lenght, dim]
def k_mean(k, dim, points):
    print(f"0) {points[0]}\n")
    n = len(points)
    cluster_coords = GetStartClusterCoords(k, dim)
    new_cluster_coords = np.zeros((k, dim))
    color = np.zeros((n, 2))  # [какой цвет, расстояние до кластера]

    while True:
        print(f"1) {points[0]}\n")
        for i in range(n):
            color[i, 0] = 0
            color[i, 1] = dist(points[i], cluster_coords[0])
            for j in range(k):
                a = dist(points[i], cluster_coords[j])  # новое расстояние до кластера
                if a <= color[i, 1]:
                    color[i, 0] = j
                    color[i, 1] = a
        # Получаем новые координаты
        print(f"2) {points[0]}\n")
        for i in range(k):  # Для кластера i
            # new_cluster_coords[i, :] = np.mean(points[color[:,0]==i], axis=1)
            for j in range(dim):  # Для измерения j
                mid = 0
                num = 0
                for m in range(n):  # Считаем точку k
                    if int(color[m, 0]) == i:
                        mid += points[m, j]
                        num += 1
                if num == 0:
                    qwer = random.randint(5, 10)
                else:
                    new_cluster_coords[i, j] = mid / num

        flag = True
        print(f"3) {points[0]}\n")
        for i in range(k):
            qwe = cluster_coords[i]
            asd = new_cluster_coords[i]
            distantion = dist(qwe, asd)
            print(f"cl_coord) {cluster_coords}")
            print(f"cl_coord_new) {new_cluster_coords}")
            print(f"distantion) {distantion}\n")
            flag = flag and (distantion < 0.1)
            cluster_coords[i] = new_cluster_coords[i]
        print(f"4) {points[0]}\n")
        if flag:
            break

    return color, cluster_coords


# Функция определения координат кластера, выполняется через средние
# k - количество кластеров
# dim - количество измерений
# min_max[k, dim, 2] - наименьшее и наибольшее стартовые значения для центра кластера
#
# Возвращает arr[k, dim]
def GetStartClusterCoords(k, dim, min_max=0):
    # return np.array([[random.randint(0, 20) for _ in range(dim)]for _ in range(k)])
    arr = np.zeros((k, dim))
    for i in range(k):
        for j in range(dim):
            arr[i, j] = random.randint(0, 20)
    return arr


# Функция расстояния точки по Евклиду
def dist(a: list, b: list) -> float:
    return math.sqrt(sum(((item_1 - item_2) ** 2 for item_1, item_2 in zip(a, b))))
    # sum = 0
    # for i in range(min(len(a), len(b))):
    #    sum += (a[i] - b[i]) ** 2
    # return math.sqrt(sum)


def get_centers(cluster_num, dimб, a=0, b=10):
    centers = np.zeros((cluster_num, dim))
    for i in range(cluster_num):
        for j in range(dim):
            centers[i, j] = random.randint(a, b)
    return centers


def get_sigmas(cluster_num, dim, a=1, b=5):
    sigmas = np.zeros((cluster_num, dim))
    for i in range(cluster_num):
        for j in range(dim):
            sigmas[i, j] = random.randint(a, b)
    return sigmas


# Выводит график зависимости суммы квадратов расстояний от количества кластеров
# На вход получает массив всех точек
def Lokot(df):
    X_std = StandardScaler().fit_transform(df)
    sse = []
    list_k = list(range(2, 7))

    for k in list_k:
        km = KMeans(n_clusters=k)
        km.fit(X_std)
        sse.append(km.inertia_)

    plt.figure(figsize=(6, 6))
    plt.plot(list_k, sse, '-o')
    plt.xlabel(r'Number of clusters')
    plt.ylabel('Sum of squared distance')


def Sil_a(cluster, point):
    d = 0
    for i in range(len(cluster)):
        d += dist(point, cluster[i])
    d /= (len(cluster) - 1)
    return d


def Silyet(Points, cluster_nums, colors, dim):
    B = []
    offset = 0
    cluster_content = [[] for i in range(cluster_nums)]
    points_in_cluster = np.zeros(cluster_nums)
    for i in range(len(Points)):
        points_in_cluster[int(colors[i, 0])] += 1

    for j in range(len(Points)):
        cluster_content[int(colors[j, 0])].append(Points[j])

    for i in range(cluster_nums):
        koef_sil = np.zeros(len(cluster_content[i]))
        for j in range(len(cluster_content[i])):
            a = Sil_a(cluster_content[i], cluster_content[i][j])
            for q in range(cluster_nums):
                if q != i:
                    B.append(Sil_a(cluster_content[q], cluster_content[i][j]))
            b = np.amin(B)
            koef_sil[j] = (b - a) / max(b, a)
        koef_sil.sort()
        plt.bar(np.array(range(len(cluster_content[i]))) + offset, koef_sil)
        offset += len(cluster_content[i])
    return koef_sil


if __name__ == "__main__":
    cluster_num = 3
    colors = ["red", "blue", "green", "purple", "orange", "yellow", "pink", "aqua"]
    dim = 3
    point_in_cluster = 100

    centers = get_centers(cluster_num, dim, 0, 10)
    print(f"centers:\n{centers}\n")

    sigmas = get_sigmas(cluster_num, dim, 1, 2)
    print(f"sigmas:\n{sigmas}\n")

    dots = MakeCluster(centers, dim, sigmas, cluster_num, point_in_cluster)

    points = np.zeros((point_in_cluster * cluster_num, dim))
    for i in range(cluster_num):
        for j in range(point_in_cluster):
            for k in range(dim):
                points[i * point_in_cluster + j, k] += dots[i, j, k]

    print(f"-1) {points[0]}\n")

    fig = plt.figure()
    ax_3d = Axes3D(fig)
    for i in range(cluster_num):
        ax_3d.plot(dots[i, :, 0], dots[i, :, 1], dots[i, :, 2] if dim > 2 else 0, 'o')

    colour, cl_coord = k_mean(cluster_num, dim, points)

    fig = plt.figure()
    ax_3d = Axes3D(fig)
    for i in range(point_in_cluster * cluster_num):
        ax_3d.plot(points[i, 0], points[i, 1], points[i, 2] if dim > 2 else 0, 'o', color=colors[int(colour[i, 0])])

    fig = plt.figure()
    ax_3d = Axes3D(fig)
    # Вывод k-mean sklern
    kmeans = KMeans(n_clusters=cluster_num)
    kmeans.fit(points)
    ax_3d.scatter(points[:, 0], points[:, 1], points[:, 2] if dim > 2 else 0, c=kmeans.labels_, cmap='rainbow')

    print(f"Центры по гауссовым распределениям: {centers}")
    print(f"Центры найденных кластеров: {cl_coord}")
    print(f"Центры найденных кластеров (sklearn): {kmeans.cluster_centers_}")

    plt.show()

    Silyet(points, cluster_num, colour, dim)
    plt.show()
    Lokot(points)
    plt.show()

'''
    # Вывод начальных точек
    fig, axs = plt.subplots(nrows=2, ncols=2)

    for i in range(cluster_num):
        axs[0, 0].plot(dots[i, :, 0], dots[i, :, 1], 'o')

    # Вывод моего k-mean кластеризации
    print(f"point_in_cluster = {point_in_cluster}, cluster_num = {cluster_num}")
    for i in range(point_in_cluster * cluster_num):
        print(f"i = {i}")
        axs[1, 0].plot(points[i, 0], points[i, 1], 'o', color=colors[int(colour[i, 0])])

    # Вывод k-mean sklern
    kmeans = KMeans(n_clusters=cluster_num)
    kmeans.fit(points)
    axs[1, 1].scatter(points[:, 0], points[:, 1], c=kmeans.labels_, cmap='rainbow')

    plt.show()
'''
