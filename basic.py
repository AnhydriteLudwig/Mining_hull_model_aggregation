import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import alphashape
from math import sqrt, pow
from descartes import PolygonPatch
from shapely.geometry import MultiPolygon


"""
Part B & C written by Xiaoteng Zhou (a1783926)
Part B using Alpha Shape algorithm
Part C using limiting vertical distance algorithm
"""


def obj_loader(filename):
    """
    obj file loader
    read file and input all vertices in it into a list
    """
    vertices = []
    try:
        _f = open(filename)
        for line in _f:
            if line[:2] == "v ":
                index1 = line.find(" ") + 1
                index2 = line.find(" ", index1 + 1)
                index3 = line.find(" ", index2 + 1)
                vertex = (float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1]))
                vertex = (round(vertex[0], 2), round(vertex[1], 2), round(vertex[2], 2))
                vertices.append(vertex)
        _f.close()
        return vertices
    except IOError:
        print(".obj file not found.")


def get_2d_points(path):
    """
    set all 3d points read from obj file to 2d points
    by multiplying an matrix that doesn't contain z axis value
    then all points will be set to z axis view projection
    :param path:
    :return:
    """
    data = obj_loader(path)
    projection_matrix = np.matrix([
        [1, 0, 0],
        [0, 1, 0],
    ])
    width, height = 100, 100
    scale = 0.1
    circle_pos = [width, height]
    _points = []
    for point in data:
        point = np.matrix(np.asarray(point))
        _points.append(point)
    single_points2d = []
    for point in _points:
        projected2d = np.dot(projection_matrix, point.reshape((3, 1)))
        _x = float(projected2d[0][0] * scale) + circle_pos[0]
        _y = float(projected2d[1][0] * scale) + circle_pos[1]
        single_points2d.append((_x, _y))
    return single_points2d


def read_file(_path):
    """
    going through all obj files in folder and read them data together
    :param path:
    :return:
    """
    points2D = []
    for f in listdir(_path):
        file_path = join(_path, f)
        if isfile(file_path):
            single_points2D = get_2d_points(file_path)
            for sin in single_points2D:
                points2D.append(sin)
    return points2D


def point_2_line_distance(point_a, point_b, point_c):
    """
    计算点a到点b c所在直线的距离
    :param point_a:
    :param point_b:
    :param point_c:
    :return:
    """
    # 首先计算b c 所在直线的斜率和截距
    if point_b[0] == point_c[0]:
        return 9999999
    slope = (point_b[1] - point_c[1]) / (point_b[0] - point_c[0])
    intercept = point_b[1] - slope * point_b[0]

    # 计算点a到b c所在直线的距离
    distance = abs(slope * point_a[0] - point_a[1] + intercept) / sqrt(1 + pow(slope, 2))
    return distance


def diluting(point_list, threshold):
    """
    抽稀
    :param point_list:二维点列表, threshold:阈值
    :return:
    """
    _final_list = [point_list[0]]
    check_index = 1
    while check_index < len(point_list) - 1:
        distance = point_2_line_distance(point_list[check_index], _final_list[-1], point_list[check_index + 1])
        if distance < threshold:
            check_index += 1
        else:
            _final_list.append(point_list[check_index])
            check_index += 1
    _final_list.append(point_list[0])
    return _final_list


def show_result(_points, _points_s):
    """
    plot all graph
    :param _points:
    :param _points_s:
    :return:
    """
    _x_list = []
    _y_list = []
    for item in _points:
        _a, _b = item
        _x_list.append(_a)
        _y_list.append(_b)
    _x_list_s = []
    _y_list_s = []
    for item_s in _points_s:
        _a_s, _b_s = item_s
        _x_list_s.append(_a_s)
        _y_list_s.append(_b_s)
    plt.subplot(2, 2, 1)
    plt.plot(_x_list, _y_list, color='g', marker='.', linestyle='solid')
    plt.title("PART B with points")
    plt.subplot(2, 2, 2)
    plt.plot(_x_list, _y_list, color='g', marker=',', linestyle='solid')
    plt.title("PART B without points")
    plt.subplot(2, 2, 3)
    plt.plot(_x_list_s, _y_list_s, color='g', marker='.', linestyle='solid')
    plt.title("PART C with points", y=-0.3)
    plt.subplot(2, 2, 4)
    plt.plot(_x_list_s, _y_list_s, color='g', marker=',', linestyle='solid')
    plt.title("PART C without points", y=-0.3)
    plt.show()


if __name__ == "__main__":
    threshold = 0.4  # 阈值
    folder = 'Model/440'
    points = read_file(folder)
    alpha_shape = alphashape.alphashape(points, 0.25)
    if isinstance(alpha_shape, MultiPolygon):
        alpha_shape = alphashape.alphashape(points, 0.25)
        x, y = alpha_shape[-1].exterior.coords.xy
        border_points = list(zip(x, y))
        border_points_simplify = list(zip(x, y))
        for c in border_points:
            border_points[border_points.index(c)] = list(c)
        border_points_simplify = diluting(border_points, threshold)
        show_result(border_points, border_points_simplify)
    else:
        alpha_shape = alphashape.alphashape(points, 0.5)
        x, y = alpha_shape.exterior.coords.xy
        border_points = list(zip(x, y))
        border_points_simplify = list(zip(x, y))
        for c in border_points:
            border_points[border_points.index(c)] = list(c)
        border_points_simplify = diluting(border_points, threshold)
        show_result(border_points, border_points_simplify)

