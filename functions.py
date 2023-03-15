import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statistics import mean, stdev, median, variance
import math

def guess_centroid(distances, speeds):
    plt.scatter(distances, speeds)
    #plt.show()
    centroid_input = input('Input Centroid Locations: ')
    centroids = eval(centroid_input)
    return centroids

def dist(distances, speeds, centroids=None):
    if centroids is None:
        centroids = guess_centroid(distances, speeds)


    clusters = {}
    for centroid in centroids:
        clusters[centroid] = []


    for i in range(len(distances)):
        distances_to_centroids = []
        for centroid in centroids:
            distance_to_centroid = ((centroid[0] - distances[i])**2) + ((centroid[1] - speeds[i])**2)
            distances_to_centroids.append(distance_to_centroid)
        closest_centroid = centroids[distances_to_centroids.index(min(distances_to_centroids))]
        clusters[closest_centroid].append(i)


    for centroid in centroids:
        x = [distances[k] for k in clusters[centroid]]
        y = [speeds[k] for k in clusters[centroid]]
        plt.scatter(x, y)
        plt.plot([centroid[0]], [centroid[1]], 'k', marker='D')

    plt.show()
    return clusters


def calculate_new_centroids(distances, speeds, clusters):
    new_centroids = []
    for centroid in clusters:
        x_mean = mean([distances[k] for k in clusters[centroid]])
        y_mean = mean([speeds[k] for k in clusters[centroid]])
        new_centroids.append((x_mean, y_mean))
    return new_centroids


def clusters(distances, speeds, centroids=None, repetitions=4):
    for n in range(repetitions):
        cluster = dist(distances, speeds, centroids)
        centroids = calculate_new_centroids(distances, speeds, cluster)
    return cluster

def elem_stats(data):
    new_dict = {
        'mean': mean(data),
        'median': median(data),
        'variance': variance(data),
        'std': stdev(data),
        'min': min(data),
        'max': max(data)
    }
    return new_dict

def correlation(x, y):
  multi_sum = []
  x_sqrt = []
  y_sqrt = []
  for value, num in zip(x, y):
    multi_sum.append(value*num)
  numerator = int(len(x))*sum(multi_sum)-sum(x)*sum(y)
  for n in x:
    x_sqrt.append(n**2)
  for n in y:
    y_sqrt.append(n**2)
  denominator_1 = len(x)*sum(x_sqrt)-sum(x)**2
  denominator_2 = len(x)*sum(y_sqrt)-sum(y)**2
  denominator = math.sqrt(denominator_1*denominator_2)
  r = numerator/denominator
  #print(sum(multi_sum))
  return r

def line_best_fit(x, y):
  x_sqrt = []
  multi_sum = []
  for n in x:
    x_sqrt.append(n**2)
  a = np.array([[sum(x_sqrt), sum(x)],[sum(x), len(x)]])
  inv_a = np.linalg.inv(a)
  for value, num in zip(x, y):
    multi_sum.append(value*num)
  b = np.array([sum(multi_sum), sum(y)])
  final = np.dot(b,inv_a)
  print(final, "hi")
  return final

def sigma_xy(xd, yd):
    nlist = []
    for i in range(len(xd)):
        nlist.append((xd[i]*yd[i]))
    return sum(nlist)

def least_sqrs(xd, yd):
    rix1 = [[sum(val**2 for val in xd), sum(xd)], [sum(xd), len(xd)]]
    rix2 = [sigma_xy(xd, yd), sum(yd)]
    ray1 = np.array(rix1)
    ray2 = np.array(rix2)
    invarray = np.linalg.inv(ray1)
    solution = np.dot(invarray, ray2)
    return solution