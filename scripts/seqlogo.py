from collections import Counter
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from svgpath2mpl import parse_path
import matplotlib
matplotlib.use('Agg')  # 或者 'Qt5Agg'，'MacOSX'，'TkAgg'，'GTK3Agg'，'WXAgg'，'Qt4Agg'，'GTKAgg' 等


# 两种序列
aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']
dna_list = ['A', 'T', 'C', 'G', '-']
# 默认颜色
default_aa_colors = {
    'GAPVLIMFW': 'blue',
    'YSTCN': 'black',
    'KHR': 'green',
    'DE': 'purple'
}
default_dna_colors = {
    'orange': 'G',
    'red': 'T',
    'blue': 'C',
    'green': 'A'
}
# 颜色的RGB
colors = {
    'black': '#000000',
    'red': '#cd0000',
    'green': '#00cd00',
    'blue': '#0000cd',
    'orange': '#ff8040',
    'purple': '#cd00cd',
    'yellow': '#cdcd00'
}
# 字母的矢量图
letter_vector = {
    'A': '''M495.19,594.93h-108.4l-43.09-112.76H146.44l-40.73,112.76H0L192.22,98.54h105.37l197.61,496.4Zm-183.47-196.39l-68-184.2-66.65,184.2h134.65Z''',
    'C': '''M423.33,398.45l111.77,29.77c-17.14,52.34-45.63,91.22-85.48,116.62-39.86,25.41-90.42,38.11-151.69,38.11-75.82,0-138.13-21.75-186.94-65.26-48.82-43.51-73.22-102.99-73.22-178.45,0-79.82,24.54-141.81,73.61-185.98s113.59-66.25,193.56-66.25c69.84,0,126.57,17.34,170.19,52.01,25.96,20.5,45.43,49.94,58.42,88.33l-114.11,22.9c-6.75-24.86-20.84-44.49-42.26-58.88-21.42-14.39-47.45-21.59-78.08-21.59-42.32,0-76.66,12.76-103.01,38.27-26.35,25.52-39.53,66.85-39.53,123.98,0,60.63,12.98,103.81,38.95,129.55,25.96,25.74,59.71,38.6,101.26,38.6,30.63,0,56.99-8.18,79.06-24.54,22.06-16.36,37.9-42.09,47.51-77.2Z''',
    'D': '''M59.74,98.53h218.38c49.25,0,86.79,3.16,112.62,9.48,34.71,8.58,64.45,23.82,89.21,45.71,24.76,21.9,43.59,48.71,56.51,80.42,12.92,31.72,19.38,70.83,19.38,117.33,0,40.86-6.05,76.08-18.16,105.65-14.8,36.12-35.92,65.35-63.37,87.7-20.73,16.93-48.71,30.14-83.96,39.62-26.38,7-61.63,10.5-105.76,10.5H59.74V98.53Zm119.48,83.97V511.3h89.21c33.36,0,57.45-1.58,72.25-4.74,19.38-4.06,35.45-10.95,48.24-20.65,12.78-9.7,23.21-25.68,31.28-47.91,8.07-22.23,12.11-52.54,12.11-90.92s-4.04-67.83-12.11-88.38c-8.07-20.54-19.38-36.57-33.91-48.08s-32.97-19.3-55.3-23.36c-16.69-3.16-49.38-4.74-98.09-4.74h-53.69Z''',
    'E': '''M66.43,594.93V98.53h484.61v83.97H198.39v110.05h328.13v83.64H198.39v135.1h365.13v83.64H66.43Z''',
    'F': '''M74,594.93V98.93h496v84H220v117h302v84H220v211H74Z''',
    'G': '''M300.47,398.93v-80.9h230.69v191.28c-22.42,19.65-54.9,36.96-97.45,51.91-42.55,14.95-85.64,22.44-129.27,22.44-55.45,0-103.77-10.53-144.99-31.61-41.22-21.07-72.2-51.2-92.93-90.4-20.73-39.19-31.1-81.83-31.1-127.9,0-50,11.57-94.44,34.71-133.31,23.14-38.86,57.01-68.67,101.6-89.42,33.99-15.94,76.29-23.91,126.92-23.91,65.81,0,117.21,12.5,154.21,37.5,37,25,60.8,59.56,71.41,103.66l-106.3,18.01c-7.47-23.58-21.51-42.2-42.12-55.84-20.61-13.64-46.34-20.47-77.2-20.47-46.77,0-83.95,13.43-111.55,40.29-27.6,26.86-41.4,66.71-41.4,119.55,0,56.99,13.98,99.73,41.94,128.23,27.96,28.5,64.6,42.74,109.92,42.74,22.42,0,44.89-3.98,67.44-11.96,22.54-7.97,41.88-17.63,58.03-28.99v-60.92h-122.58Z''',
    'H': '''M63.29,594.93V98.53h124.88v195.38h244.71V98.53h124.88v496.4h-124.88v-217.05H188.17v217.05H63.29Z''',
    'I': '''M59.07,594.93V98.53h124.88v496.4H59.07Z''',
    'K': '''M57.52,594.93V98.53h111.28v220.43L393.6,98.53h149.62l-207.51,193.34,218.79,303.05h-143.98l-151.5-232.96-90.22,82.96v150H57.52Z''',
    'L': '''M75.4,599.8V103.44H217.54v412.04h353.45v84.32H75.4Z''',
    'M': '''M50.83,595.63V98.65H206.11l93.24,339L391.54,98.65h155.63v496.98h-96.4V204.42l-102,391.21h-99.9L147.22,204.42v391.21H50.83Z''',
    'N': '''M64.85,595.63V98.65h122.87l255.98,331.88V98.65h117.33v496.98h-126.71L182.18,271.54v324.09H64.85Z''',
    'P': '''M65.92,595.63V98.65h210.16c79.64,0,131.55,2.49,155.74,7.46,37.17,7.46,68.28,23.68,93.35,48.65,25.07,24.98,37.61,57.24,37.61,96.79,0,30.51-7.23,56.16-21.68,76.95-14.46,20.8-32.82,37.12-55.08,48.99-22.27,11.87-44.91,19.72-67.91,23.56-31.27,4.75-76.54,7.12-135.83,7.12h-85.39v187.47H65.92ZM196.89,182.72v141.03h71.67c51.61,0,86.12-2.6,103.53-7.8,17.4-5.2,31.04-13.33,40.92-24.41,9.88-11.07,14.82-23.95,14.82-38.65,0-18.08-6.93-32.99-20.79-44.75-13.87-11.75-31.41-19.1-52.65-22.04-15.64-2.26-47.05-3.39-94.24-3.39h-63.27Z''',
    'Q': '''M446.34,475.4c24.4,15.74,50.93,28.25,79.6,37.52l-36.61,63.25c-15-4.04-29.67-9.58-44-16.64-3.14-1.41-25.19-14.53-66.16-39.34-32.24,12.71-67.96,19.06-107.13,19.06-75.68,0-134.96-20.07-177.83-60.22-42.88-40.14-64.31-96.53-64.31-169.16s21.49-128.76,64.48-169.01c42.99-40.25,101.31-60.37,174.98-60.37s130.86,20.12,173.63,60.37c42.76,40.25,64.15,96.58,64.15,169.01,0,38.33-5.93,72.02-17.8,101.07-8.96,22.2-23.29,43.68-42.99,64.46Zm-79.93-50.54c12.76-13.51,22.33-29.85,28.71-49.02,6.38-19.16,9.57-41.16,9.57-65.97,0-51.24-12.54-89.52-37.61-114.84-25.08-25.32-57.88-37.98-98.4-37.98s-73.38,12.71-98.57,38.13c-25.19,25.42-37.78,63.65-37.78,114.69s12.59,90.63,37.78,116.35c25.19,25.72,57.04,38.58,95.55,38.58,14.33,0,27.88-2.12,40.64-6.35-20.15-11.9-40.64-21.18-61.46-27.84l27.88-51.14c32.69,10.09,63.92,25.22,93.7,45.39Z''',
    'R': '''M56.59,595.63V98.65h235.05c59.1,0,102.05,4.47,128.84,13.39,26.79,8.93,48.23,24.81,64.33,47.63,16.09,22.83,24.15,48.93,24.15,78.31,0,37.29-12.2,68.09-36.6,92.38-24.4,24.3-60.87,39.61-109.41,45.94,24.15,12.66,44.08,26.56,59.8,41.7,15.72,15.14,36.91,42.04,63.57,80.68l67.53,96.96h-133.56l-80.74-108.14c-28.67-38.65-48.29-63-58.86-73.06-10.56-10.05-21.76-16.95-33.58-20.68-11.83-3.73-30.56-5.59-56.21-5.59h-22.64v207.47H56.59Zm111.68-286.8h82.62c53.57,0,87.02-2.03,100.36-6.1,13.33-4.07,23.77-11.07,31.31-21.02,7.54-9.94,11.32-22.37,11.32-37.29,0-16.72-4.97-30.22-14.9-40.51-9.94-10.28-23.96-16.78-42.07-19.49-9.05-1.13-36.22-1.69-81.49-1.69h-87.15v126.11Z''',
    'S': '''M30.88,419.75l120.18-9.18c7.23,31.7,21.91,54.98,44.02,69.85,22.12,14.87,51.95,22.3,89.51,22.3,39.78,0,69.75-6.61,89.92-19.84,20.17-13.22,30.25-28.69,30.25-46.4,0-11.36-4.24-21.04-12.73-29.02-8.49-7.98-23.3-14.92-44.44-20.82-14.47-3.93-47.43-10.93-98.89-20.99-66.21-12.9-112.66-28.74-139.37-47.55-37.55-26.45-56.33-58.7-56.33-96.74,0-24.48,8.83-47.39,26.5-68.7,17.66-21.32,43.12-37.55,76.36-48.7,33.24-11.15,73.37-16.72,120.38-16.72,76.78,0,134.57,13.23,173.38,39.68,38.81,26.45,59.18,61.76,61.13,105.92l-123.51,4.26c-5.29-24.7-16.63-42.47-34.01-53.29-17.39-10.82-43.47-16.23-78.24-16.23s-63.99,5.8-84.29,17.38c-13.08,7.43-19.61,17.38-19.61,29.84,0,11.37,6.12,21.1,18.36,29.19,15.58,10.28,53.41,20.99,113.5,32.14s104.53,22.68,133.32,34.6c28.79,11.92,51.33,28.2,67.6,48.86,16.27,20.66,24.41,46.19,24.41,76.57,0,27.55-9.74,53.34-29.21,77.39-19.48,24.05-47.02,41.92-82.62,53.62-35.61,11.69-79.98,17.54-133.11,17.54-77.34,0-136.73-14.04-178.18-42.14-41.45-28.09-66.21-69.03-74.28-122.81Z''',
    'T': '''M203.32,595.63V182.72H18.68V98.65h494.49v84.07h-184.21v412.91h-125.64Z''',
    'U': '''M62.42,96.92h125.69V361.38c0,41.97,1.55,69.17,4.67,81.6,5.37,19.98,18.19,36.03,38.43,48.13,20.24,12.1,47.91,18.15,83.01,18.15s62.56-5.71,80.68-17.15c18.11-11.43,29.01-25.48,32.7-42.13,3.68-16.65,5.52-44.3,5.52-82.93V96.92h125.69V353.39c0,58.62-3.4,100.04-10.19,124.24-6.79,24.2-19.32,44.63-37.58,61.28-18.26,16.65-42.67,29.92-73.25,39.8-30.57,9.88-70.49,14.82-119.74,14.82-59.45,0-104.53-5.39-135.24-16.15-30.72-10.77-54.99-24.76-72.82-41.97-17.83-17.21-29.58-35.25-35.24-54.12-8.21-27.98-12.31-69.28-12.31-123.9V96.92Z''',
    'V': '''M189.52,595.63L-.36,98.65H115.96L250.4,466.47,380.49,98.65h113.79l-190.25,496.98h-114.51Z''',
    'W': '''M92.34,595.63L1.81,98.65H80.19l57.17,341.38L206.67,98.65h91.05l66.48,347.14,58.2-347.14h77.08l-92.08,496.98h-81.22L250.65,224.08l-75.27,371.55H92.34Z''',
    'Y': '''M193.75,595.63v-209.17L.02,98.65H125.2l124.46,196.62,121.94-196.62h123.02l-194.45,288.49v208.49h-106.42Z'''
}

def read_fas(filepath, sequence_type):
    """
    读取fasta文件，并计算各字母出现频率
    :param filepath: (str) fasta文件位置
    :param sequence_type: (str) 序列类型，可为"dna"或"aa"
    :return: seq_logo_df (pandas.Dataframe) 频率矩阵
    """
    fa_dict = {}
    data_dict = {}
    cnt_list = []
    Frequence_list = []
    if sequence_type == 'aa':
        index = aa_list
    elif sequence_type == 'dna':
        index = dna_list
    else:
        return None

    with open(filepath) as fa:
        for line in fa:
            line = line.replace('\n', '')
            if line.startswith('>'):
                seq_name = line[1:]
                fa_dict[seq_name] = ''
            else:
                fa_dict[seq_name] += line.replace('\n', '')

    seq_num = len(fa_dict)

    for i in fa_dict:
        data_dict[i] = list(fa_dict[i])
    df = pd.DataFrame.from_dict(data_dict)

    for i in range(len(df)):
        cnt_list.append(dict(Counter(df.iloc[i])))

    for i in range(len(cnt_list)):
        dict_F = {}
        i_letters = []
        letter_sum = 0
        for key in cnt_list[i]:
            if key != '-' and key not in index:
                raise 'Please check your fasta file, or you may choose wrong sequence_type. \n' \
                      'sequence_type can only be set "aa" for amino-acid or "dna" for dna sequence.'
            if key == '-':  # 去除出现的"-"
                continue
            letter_sum += cnt_list[i][key]
            i_letters.append(key)
        for letter in i_letters:
            dict_F[letter] = cnt_list[i][letter] / letter_sum
        if '-' in cnt_list[i].keys():
            dict_F['-'] = cnt_list[i]['-'] / seq_num
        Frequence_list.append(dict_F)


    seq_logo_df = pd.DataFrame(np.zeros((len(index), len(df))), index=index)
    for i in range(len(Frequence_list)):
        for key in Frequence_list[i]:
            seq_logo_df.loc[key][i] = Frequence_list[i][key]
    return seq_logo_df


def calculate_entropy(df, sequence_type):
    """
    熵作图情况下需要计算各位点的熵并返回新矩阵以画图
    :param df: (pandas.Dataframe) 频率矩阵
    :param sequence_type: (str) 序列类型，可为"dna"或"aa"
    :return: new_df (pandas.Dataframe) 熵频率矩阵
    """
    if sequence_type == 'aa':
        log_para = np.log2(20)
    elif sequence_type == 'dna':
        log_para = np.log2(4)

    height_list = df.loc['-'].tolist()  # 若序列该位点gap过多，则在画图时降低高度
    new_df = df.drop('-')

    for i in range(len(new_df.columns)):
        temp_df = new_df[new_df[i] > 0][i]
        if len(temp_df) == 0:
            continue
        E = 0
        for j in range(len(temp_df)):
            E -= temp_df[j] * np.log2(temp_df[j])
        new_df[i] *= (log_para - E) * (1 - height_list[i])
    return new_df


def split_list(matrix, draw_type, sequence_type, row_limit):
    """
    分割矩阵得到每一行的画图序列列表
    :param matrix: (pandas.Dataframe) 频率矩阵
    :param draw_type: (str) 画图类型，可为"entropy"或"probability"
    :param sequence_type: (str) 序列类型，可为"dna"或"aa"
    :param row_limit: (int) 每行个数限制，一般在20~50
    :return: plot_list (list) 分割后的列表
    """
    # split_matrix = split_matrix.drop('-')
    if draw_type == 'entropy':
        split_matrix = calculate_entropy(matrix, sequence_type)
    else:
        split_matrix = matrix.drop('-')
    data_list = []
    for i in range(len(split_matrix.columns)):
        tdf = split_matrix[i][split_matrix[i] > 0].sort_values()
        tind = tdf.index
        tval = tdf.values.tolist()
        data_list.append(list(zip(tind, tval)))

    plot_list = []
    for i in range(0, len(split_matrix.columns), row_limit):
        plot_list.append(data_list[i:i + row_limit])
    return plot_list

def create_colors_map(sequence_type, letter_colors):
    '''
    得到颜色映射字典
    :param sequence_type: (str) 序列类型，可为"dna"或"aa"
    :param letter_colors: (dict) 字母的颜色字典
    :return: colors_map (dict) 单字母到颜色的映射字典
    '''
    input_colors = letter_colors
    if sequence_type == 'aa':
        if input_colors is None:
            input_colors = default_aa_colors
        index = aa_list
    elif sequence_type == 'dna':
        if input_colors is None:
            input_colors = default_dna_colors
        index = dna_list

    colors_map = {}
    for key in input_colors.keys():
        for key_char in key:
            if key_char not in index:
                raise 'your input colors has wrong letters of syntax, please check your sequence_type or colors'
            colors_map[key_char] = input_colors[key]
    last_letters = list(set(index) - set(colors_map.keys()))
    for l in last_letters:
        colors_map[l] = 'black'

    return colors_map

def get_patch(letter, color, x, y, dx, dy):
    path = parse_path(letter_vector[letter])
    if letter == 'I':
        path.vertices[:, 0] += path.vertices[:, 0].min() * 3
    else:
        path.vertices[:, 0] -= path.vertices[:, 0].min()
    path.vertices[:, 1] -= path.vertices[:, 1].min()
    path.vertices[:, 0] /= path.vertices[:, 0].max()
    path.vertices[:, 1] /= path.vertices[:, 1].max()
    path.vertices[:, 1] = 1 - path.vertices[:, 1]
    if letter == 'I':
        path.vertices *= [0.6*dx, dy]
    else:
        path.vertices *= [dx, dy]
    path.vertices += [x,y]
    return PathPatch(path, facecolor=colors[color], edgecolor='none')


def plot_seq_logo(data_list, ticks_list, WIDTH, HEIGHT, start, colors_map, row_limit, sequence_type, draw_type, dpi, label_font, index_begin):
    """
    主要的画图函数
    :param data_list: (list) 通过split_list函数分割好的子列表
    :param ticks_list: (list) 自定义横坐标子列表，若无则按照位点为横坐标
    :param WIDTH: (int) 图片宽度(英寸）
    :param HEIGHT: (int) 图片高度(英寸)
    :param start: (int) 子图序列起始位点
    :param colors_map: (dict) 字母颜色字典(单字母-颜色)
    :param row_limit: (int) 每行个数限制，一般在20~50
    :param sequence_type: (str) 序列类型，可为"dna"或"aa"
    :param draw_type: (str) 画图类型，可为"entropy"或“probability"
    :param dpi: (int) 图片分辨率，一般在96~300
    :param label_font: (str) 画图字体
    :param index_begin: (int) 全局序列起始位点
    :return: fig (PIL.Image) 得到子序列子图
    """
    font = FontProperties(label_font)
    Axis_W = row_limit
    if draw_type == 'entropy':
        if sequence_type == 'aa':
            Axis_H = np.log2(20)
        elif sequence_type == 'dna':
            Axis_H = np.log2(4)
    elif draw_type == 'probability':
        Axis_H = 1

    fig, ax = plt.subplots(dpi=dpi)
    fig.set_size_inches(WIDTH, HEIGHT)
    x = 0.5 + start + index_begin

    for i in range(len(data_list)):
        h = 0
        for j in data_list[i]:
            height = j[1]
            letter = j[0]
            patch = get_patch(letter, colors_map[letter],x,h,0.9,height) # 调成1会粘连，设为0.9不会
            ax.add_artist(patch)
            h += height
        x += 1

    if draw_type == 'entropy':
        if sequence_type == 'aa':
            plt.annotate('C', xy=(x, 0), xytext=(x, -0.7), textcoords='data', fontsize=30, weight='bold')
            plt.annotate('N', xy=(x-len(data_list), 0), xytext=(x-len(data_list) - 0.5, -0.7), textcoords='data', fontsize=30, weight='bold')
        else:
            plt.annotate("3'", xy=(x, 0), xytext=(x - 0.5, -0.7), textcoords='data', fontsize=30, weight='bold')
            plt.annotate("5'", xy=(x-len(data_list), 0), xytext=(x-len(data_list) - 0.5, -0.7), textcoords='data', fontsize=30, weight='bold')
    else:
        if sequence_type == 'aa':
            plt.annotate('C', xy=(x, 0), xytext=(x, -0.162), textcoords='data', fontsize=30, weight='bold')
            plt.annotate('N', xy=(x - len(data_list), 0), xytext=(x - len(data_list) - 0.5, -0.162), textcoords='data',
                         fontsize=30, weight='bold')
        else:
            plt.annotate("3'", xy=(x, 0), xytext=(x - 0.5, -0.324), textcoords='data', fontsize=30, weight='bold')
            plt.annotate("5'", xy=(x - len(data_list), 0), xytext=(x - len(data_list) - 0.5, -0.324), textcoords='data',
                         fontsize=30, weight='bold')

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    if ticks_list is not None:
        plt.xticks(range(start + index_begin + 1, start + index_begin + len(data_list) + 1), labels=ticks_list, fontproperties=font, size=30, rotation='vertical', weight='bold')
    else:
        plt.xticks(range(start + index_begin + 1, start + index_begin + len(data_list) + 1), fontproperties=font, size=30, rotation='vertical', weight='bold')
    plt.yticks(fontproperties=font, size=30, weight='bold')
    ax.tick_params(axis='y', which='major', width=5, length=10)
    ax.tick_params(axis='x', which='major', length=0)
    ax.spines['left'].set_linewidth(5)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if draw_type == 'entropy':
        ax.set_ylabel('bits', fontproperties=font, size=40, weight='bold')
    elif draw_type == 'probability':
        ax.set_ylabel('probability', fontproperties=font, size=40, weight='bold')
    ax.set_xlim(start + index_begin + 0.2, start + index_begin + 0.7 + Axis_W)
    ax.set_ylim(0, Axis_H)
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.2, top=0.94)
    # plt.show()
    plt.close()
    return fig


def draw(matrix, row_limit, letters_color,
         sequence_type, draw_type,
         dpi, label_font, ticks,
         seq_begin, seq_end):
    """
    画图函数，将子图拼接为大图
    :param matrix: (pandas.Dataframe) 频率矩阵
    :param row_limit: (int) 每行个数限制，一般在20~50
    :param letters_color (dict) 字母的颜色字典
    :param sequence_type: (str) 序列类型，可为"dna"或"aa"
    :param draw_type: (str) 画图类型，可为"entropy"或“probability"
    :param dpi: (int) 图片分辨率，一般在96~300
    :param label_font: (str) 画图字体
    :param ticks: (list) 自定义横坐标列表
    :param seq_begin: (int) 全局序列起始位点
    :param seq_end: (int) 全局序列终止位点
    :return: big_image (PIL.Image) 结果大图
    """
    if draw_type != 'entropy' and draw_type != 'probability':
        raise 'draw_type can only be set "entropy" or "probability"!'
    index_begin = seq_begin - 1
    if seq_end is not None:
        M = matrix.iloc[:, index_begin:seq_end]
    else:
        M = matrix.iloc[:, index_begin:]
    M.columns = range(len(M.columns))
    plot_list = split_list(M, draw_type, sequence_type, row_limit)

    if row_limit in range(21):
        WIDTH = 20
    elif row_limit in range(21, 51):
        WIDTH = 30
    else:
        WIDTH = 40
    HEIGHT = 5

    colors_map = create_colors_map(sequence_type, letters_color)

    fig_list = []
    if ticks is not None:
        draw_ticks = [ticks[i:i + row_limit] for i in range(0,len(M.columns),row_limit)]
    else:
        draw_ticks = [None for _ in range(0,len(M.columns),row_limit)]

    for i in range(len(plot_list)):
        fig_list.append(plot_seq_logo(plot_list[i], draw_ticks[i], WIDTH, HEIGHT, i * row_limit, colors_map, row_limit, sequence_type, draw_type, dpi, label_font, index_begin))

    y_offset = 0
    big_image = Image.new('RGB', (WIDTH * dpi, (HEIGHT * len(plot_list)) * dpi))
    for f in fig_list:
        f.canvas.draw()
        image = np.array(f.canvas.renderer.buffer_rgba())
        pil_image = Image.fromarray(image)
        big_image.paste(pil_image, (0, y_offset))
        y_offset += int(HEIGHT) * dpi
    return big_image

def WeiSeqLogo(
        fas_filepath=None,
        Matrix=None,
        sequence_type='aa',
        draw_type='entropy',
        row_limit=20,
        letters_colors=None,
        save_filepath='WeiSeqLogoResult.png',
        dpi=300,
        label_font='Arial',
        seq_begin=1,
        seq_end=None,
        ticks=None
):
    """
    集成的sequence logo函数
    :param fas_filepath: (str) fasta文件位置
    :param Matrix: (pandas.Dataframe) 氨基酸频率矩阵
    :param sequence_type: (str) 序列类型，可为"dna"或"aa"
    :param draw_type: (str) 画图类型，可为"entropy"或“probability"
    :param row_limit: (int) 每行个数限制，一般在20~50
    :param letters_colors: (dict) 字母颜色字典
    :param save_filepath: (str) 结果图保存位置
    :param dpi: (int) 图片分辨率，一般在96~300
    :param label_font: (str) 画图字体
    :param seq_begin: (int) 全局序列起始位点
    :param seq_end: (int) 全局序列终止位点
    :param ticks: (list) 自定义横坐标列表
    """
    if fas_filepath:
        matrix = read_fas(fas_filepath, sequence_type)
    else:
        matrix = Matrix
    if matrix is None:
        raise 'Create frequency matrix Error, please check your parameters!'
    result_image = draw(matrix, row_limit, letters_colors, sequence_type, draw_type, dpi, label_font, ticks, seq_begin, seq_end)
    # result_image.show()
    result_image.save(save_filepath)


ID_TO_HHBLITS_AA = {
    0: "A",
    1: "C",  # Also U.
    2: "D",  # Also B.
    3: "E",  # Also Z.
    4: "F",
    5: "G",
    6: "H",
    7: "I",
    8: "K",
    9: "L",
    10: "M",
    11: "N",
    12: "P",
    13: "Q",
    14: "R",
    15: "S",
    16: "T",
    17: "V",
    18: "W",
    19: "Y",
    20: "X",  # Includes J and O.
    21: "-",
}


AA_LITS = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
           "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
           "X", "-"]


def msa_to_matrix(msa):
    """
    msa: np.array[seq_num, seq_len]
    """
    msa = np.array([[ID_TO_HHBLITS_AA[i] for i in row] for row in msa]).astype(str)
    
    matrix = np.zeros((len(AA_LITS), msa.shape[1]))
    for i in range(msa.shape[1]):
        for j in range(msa.shape[0]):
            aa = msa[j, i]
            matrix[AA_LITS.index(aa), i] += 1
    
    # to pandas dataframe and calculate frequency
    matrix = matrix.T
    matrix = matrix/matrix.sum(axis=1)[: ,None]
    matrix = pd.DataFrame(matrix, columns=AA_LITS)
    matrix = matrix.drop(columns=['X'])
    matrix = matrix.T
    
    return matrix


def save_seqlogo(
    name, 
    output_dir, 
    msa, 
    sequence_type='aa', 
    draw_type='entropy', 
    row_limit=50
):
    """
    matrix: np.array
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    matrix = msa_to_matrix(msa)

    colors_letter = {
        'GSTYC': 'green',
        'RKH': 'blue',
        'ED': 'red',
        'NQ': 'purple',
        'APMLVWIF': 'black',
    }

    WeiSeqLogo(
        Matrix=matrix,
        sequence_type=sequence_type,
        draw_type=draw_type,
        row_limit=row_limit,
        letters_colors=colors_letter,
        save_filepath=os.path.join(output_dir, name+'.png'),
    )
