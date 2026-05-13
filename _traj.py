""" This package is for calling HYSPLIT to generate trajectories """

from __future__ import annotations
import os
import re
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from subprocess import call, DEVNULL
from decimal import Decimal, ROUND_HALF_UP
from concurrent.futures import ProcessPoolExecutor


class RunHYSPLIT:

    """ Call the HYSPLIT to generate trajectories """

    def __init__(
            self,
            dir_save: Path,         # Directory for saving trajectories
            dir_meteo: Path,        # Directory for meteorological data
            dict_coords: dict,      # Dictionary containing the coordinates of single or multiple sites, {'1001A': [lat, lon], '1002A': [lat, lon], ...}
            m_agl: int,             # The starting height (Above Ground Level, AGL) of the backward trajectory, unit: m
            datetime: list[str, str],         # The start and end time (UTC time) of batching trajectory, ['2020-01-01 10:00:00', '2020-01-31 23:00:00']
            # dir_working: Path | None = None,      # Working directory for HYSPLIT
            # path_exe: Path | None = None,         # Path of the HYSPLIT executable file
            basename='',            # The prefix of the trajectory files
            runtime=-48,            # The total runtime of each trajectory, backward: negative, forward: positive, unit: hour
            ) -> None:

        """
        Parameters
        ----------
        dir_save: Path
            轨迹数据保存路径
        dir_meteo: Path
            气象数据 (gdas1) 所在路径
        dict_coords: dict
            单站或多站坐标字典，{'1001A': [lat, lon], '1002A': [lat, lon], ...}
        m_agl: int
            轨迹起始高度 (AGL), 即受体站点高度, 单位: m
        datetime: list
            后向轨迹起止时间 (UTC时间), ['2020-01-01 10:00:00', '2020-01-31 23:00:00']
        basename: str
            轨迹文件名前缀
        runtime: int
            每条轨迹总运行时间, 后向: 负值, 前向: 正值, 单位: 小时
            
        Notes
        -----
        2026-05-13
            删除dir_working和path_exe参数
            添加检查gdas1文件是否存在的功能
        """


        # 传参
        self.dir_traj = dir_save
        self.dir_meteo = dir_meteo
        # self.dir_working = dir_working
        self.dir_working = Path(__file__).parent / 'working'
        self.path_hysplit = self.dir_working / 'hyts_std.exe'
        self.dict_coords = dict_coords
        self.basename = basename
        self.m_agl = m_agl
        self.runtime = runtime
        self.datetime = datetime

        # gdas1文件列表
        self.meteofiles = None

        # 轨迹文件保存路径
        self.trajPath = None

        # 数据存储目录
        if not self.dir_traj.exists():
            self.dir_traj.mkdir(parents=True)

        # 初始化字典
        self.dict_month = {1: 'jan',
                           2: 'feb',
                           3: 'mar',
                           4: 'apr',
                           5: 'may',
                           6: 'jun',
                           7: 'jul',
                           8: 'aug',
                           9: 'sep',
                           10: 'oct',
                           11: 'nov',
                           12: 'dec',
                           }

        self.dict_w = {}
        for day in range(1, 32, 1):
            if 1 <= day <= 7:
                self.dict_w[day] = '.w1'
            elif 8 <= day <= 14:
                self.dict_w[day] = '.w2'
            elif 15 <= day <= 21:
                self.dict_w[day] = '.w3'
            elif 22 <= day <= 28:
                self.dict_w[day] = '.w4'
            else:
                self.dict_w[day] = '.w5'

    def get_traj(self):

        # 切换工作目录
        os.chdir(self.dir_working)

        # 生成时间索引
        dt_range = pd.date_range(start=self.datetime[0], end=self.datetime[1], freq='1h')

        for self.dt in dt_range:
            
            # 轨迹文件路径
            self.trajPath = self.dir_traj / self.dt.strftime(f'%Y%m%d_%H_{self.m_agl}.traj')

            if self.trajPath.exists():
                continue
            
            print(f'  Running | {self.trajPath}', end=' ')

            # 查找原始数据
            self._meteofinder()

            # 写入控制文件
            self._write2control()

            # 调用程序提取轨迹
            call(self.path_hysplit, stdout=DEVNULL, stderr=DEVNULL)
            
            print('Done!')

    def _meteofinder(self):
        """ 查找原始数据 """

        if self.runtime > 0:
            dt_range = pd.date_range(start=self.dt - pd.Timedelta(value=24, unit='h'), end=self.dt + pd.Timedelta(value=self.runtime, unit='h'), freq='1h')
        else:
            dt_range = pd.date_range(start=self.dt + pd.Timedelta(value=self.runtime, unit='h'), end=self.dt + pd.Timedelta(value=24, unit='h'), freq='1h')

        list_files = ['gdas1.' + self.dict_month[i.month] + str(i.year)[2:] + self.dict_w[i.day] for i in dt_range]

        self.meteofiles = set(list_files)

        # 检查文件是否都存在
        for file in list_files:
            if not (self.dir_meteo / file).exists():
                raise FileNotFoundError(f'{self.dir_meteo / file} not found!')

    def _write2control(self):
        """ write configuration to CONTROL file """

        # 轨迹开始时间，坐标数量
        controltext = [f'{self.dt.strftime("%y %m %d %H")}\n', f'{len(self.dict_coords)}\n']

        # 坐标
        for i in self.dict_coords.keys():
            line_ = ' '.join([str(round_accurately(j, 3)) for j in self.dict_coords[i]]) + ' ' + str(self.m_agl) + '\n'
            controltext.append(line_)

        # 运行时间
        controltext.append(str(self.runtime) + '\n')

        # 默认边界
        controltext.append('0\n')
        controltext.append('10000.0\n')

        # 气象数据文件的数量
        controltext.append(f'{len(self.meteofiles)}\n')

        # 气象数据文件名
        controltext += [f'{self.dir_meteo}\\\n{i}\n' for i in self.meteofiles]

        # 轨迹保存路径和文件名
        controltext.append(f'{self.trajPath.parent}\\\n{self.trajPath.name}\n')

        # 写入CONTROL文件
        with open('CONTROL', 'w') as control:
            control.writelines(controltext)


def traj2h5(dir_traj: Path | str, path_h5: Path | str) -> None:
    """ 读取目录下所有的*.traj轨迹数据，存入hdf5文件

        dir_traj：轨迹数据存放的路径，指HYSPLIT模型输出的轨迹数据
        path_h5：h5文件存放的路径

        h5文件结构：
            Dataset名为“纬度,经度”，保留三位小数，如“16.840,112.347”，
            每个Dataset存放3d数组，分别为时间、轨迹长度、特征值（lat、lon和m_agl），
            为了节约存储空间，数据全部乘以了1000，并转化为了np.int32格式，

    2022-12-01    v1
    """

    # 筛选路径下所有的*.traj文件
    list_files = dir_traj.glob('*.traj')

    # 进程池初始化
    pool = ProcessPoolExecutor(max_workers=4)
    
    # 使用map读取traj文件
    list_result = pool.map(read_one_traj, list_files)

    # 阻塞，直到所有进程完成
    pool.shutdown(wait=True)

    # 提取数据
    list_result = [i for i in list_result]

    # 字典用于保存每个坐标点的轨迹数据
    dict_coord = {}

    # 遍历所有坐标点
    for k in list_result[0].keys():

        # 计算每条轨迹的总运行小时，无正负
        runtime = list_result[0][k].shape[0]

        # 提取该坐标点在所有traj文件中的数据
        list_k = [d[k] for d in list_result]

        # 将数据合并为numpy数组，并删除其中第一列，也即runtime，保留3列：lat、lon和m_agl
        array_k = np.vstack(list_k)[:, 1:]

        # reshape为3d数组
        array_k = array_k.reshape(-1, runtime, 3)

        # 保存至字典
        dict_coord[k] = array_k

    # 创建hdf5文件对象
    f = h5py.File(path_h5, 'a')

    # 将字典中的数据写入hdf5文件，每个key-value对应一个Dataset
    for key, value in dict_coord.items():
        # f.create_dataset(name=key, data=value, compression='lzf')
        f.create_dataset(name=key, data=value, compression='gzip',  compression_opts=5)

    f.close()


def read_traj_from_h5(path_h5, list_coord, list_length):
    """ 从h5文件中提取坐标的轨迹信息, 读取单一h5文件

        path_h5: 轨迹数据绝对路径;
        list_coord: 需要提取的坐标点[(lat1, lon1), (lat2, lon2), ...];
        list_length: 轨迹长度(时间), [-24, -48, -72, ...];

    2022-10-30    v1.0
    """

    f = h5py.File(name=path_h5, mode='r')

    dict_coord = {}
    for coord in list_coord:
        dataset_name = ','.join([str(round_accurately(num, 3)) for num in coord])
        data_coord = f[dataset_name]
        # print(data_coord)
        data_coord = np.array(f[dataset_name])

        # 轨迹长度索引
        list_traj_index = [abs(i) for i in list_length]

        dict_traj_length = {}
        for i in range(len(list_length)):
            traj_length = list_length[i]
            traj_index = list_traj_index[i]

            traj_data = data_coord[:, traj_index, :]
            # print(traj_data)
            dict_traj_length[traj_length] = traj_data
            # exit(0)

        dict_coord[dataset_name] = dict_traj_length
        # print(abs(list_length))
        # print(traj_index)
        # print(data_coord)

        # exit(0)
    return dict_coord


def read_traj_from_h5_batch(dir_h5, list_coord, datetime, list_traj_length):
    """ 从h5文件中提取坐标的轨迹信息, 多进程版

        dir_h5: 存放轨迹数据的路径;
        list_coord: 需要提取的坐标点[(lat1, lon1), (lat2, lon2), ...];
        datetime: 时间范围, ['2015-01-01 00:00:00', '2020-12-31 00:00:00']; UTC时间
        # m_agl: 轨迹起始高度;
        list_traj_length: 轨迹长度(时间), [-24, -48, -72, ...];

        return: dict, UTC时间

    2022-10-30    v1.0
    """

    # 时间范围 >> 文件名
    date_range = pd.date_range(start=datetime[0], end=datetime[1], freq='H')

    # print('日期范围:', date_range)
    list_datetime = sorted(set([i.strftime('%Y%m') for i in date_range]))

    list_file = [os.path.join(dir_h5, i) + '.h5' for i in list_datetime]
    # print('文件名列表:', list_file)

    # 进程池处理
    pool = ProcessPoolExecutor(max_workers=8)
    list_result = []
    for file in list_file:
        future = pool.submit(read_traj_from_h5, file, list_coord, list_traj_length)
        # dict_file = read_traj_from_h5(path_h5=file, list_coord=list_coord, list_length=list_traj_length)

        list_result.append(future)
        # list_result.append(dict_file)
    pool.shutdown(wait=True)
    list_result = [i.result() for i in list_result]

    # 整理数据
    dict_coord = {}
    for coord in list_result[0].keys():

        dict_length = {}
        for length in list_traj_length:
            list_array = [r[coord][length] for r in list_result]
            array2d = np.vstack(list_array)

            # 恢复数据精确度, 轨迹数据从traj文件转存至h5文件时乘以了1000,这次除以1000恢复
            array2d = array2d * 0.001

            # 转变为DataFrame
            df = pd.DataFrame(data=array2d,
                              index=pd.date_range(start=date_range[0],
                                                  freq='1H',
                                                  periods=array2d.shape[0]),
                              columns=['lat', 'lon', 'm_agl'],
                              )
            df.index.name = 'datetime'

            # 时间范围筛选
            df = df[datetime[0]:datetime[1]]

            dict_length[length] = df

        dict_coord[coord] = dict_length

    return dict_coord


def read_one_traj(path_traj):
    """ 读取一个traj文件，数据以字典返回,
        为了节约存储空间，数据全部乘以了1000，并转化为了np.int32格式，

    2022-10-29    v1
    """

    # 提取时间戳
    # dt_utc = np.uint32(pd.to_datetime(os.path.split(path_traj)[1][:11], format='%Y%m%d_%H').timestamp())
    # print(dt_utc, type(dt_utc), np.uint32(dt_utc))

    with open(path_traj, 'r') as f:
        lines = [re.split('[ ]+', i.strip()) for i in f.readlines()]

    # 读取第一行中的文件数量
    gdas_files_num = int(lines[0][0])
    # print('gdas1文件数量:', gdas_files_num)

    # 读取坐标点个数
    coords_num = int(lines[gdas_files_num + 1][0])
    # print('坐标点个数:', coords_num)

    # 读取轨迹最长长度
    traj_duration = int(float(lines[-1][8]))
    # print('最长轨迹长度:', traj_duration)

    # 读取高度
    # m_agl = float(lines[gdas_files_num + 2][-1])

    # 读取坐标list
    list_coord = [i[4:6] for i in lines[gdas_files_num + 2: gdas_files_num + 2 + coords_num]]
    # array2d_coord = np.array(list_coord, dtype=np.float32)
    # print('坐标:', array2d_coord)

    # 读取轨迹主体数据
    list_data = lines[gdas_files_num + 3 + coords_num:]
    # list_data = [i[0:1] + i[8:12] for i in list_data]
    array2d_all = np.array(list_data, dtype=np.float32)[:, [0, 8, 9, 10, 11]]
    # array2d_all = np.array(list_data)[:, [0, 8, 9, 10, 11]]
    # array2d_all = array2d_all.astype(np.float32)
    # print('主体数据:', array2d_all, array2d_all.shape)

    dict_data = {}
    for j in range(coords_num):
        key_ = ','.join(list_coord[j])

        # 筛选其中的lat, lon, m_agl列数据
        value_ = array2d_all[np.where(array2d_all[:, 0] == j+1)][:, 1:]

        # 将上面三列数据乘以1000, 再转化为整数, 减少存储空间, 提取时再恢复
        value_[:, 1:] = value_[:, 1:] * 1000

        # 浮点数转化为整数
        value_ = value_.astype(np.int32)

        # print('shape of value_:', value_.shape)
        if value_.shape[0] < abs(traj_duration) + 1:

            value_ = insert_blank_line(value_, traj_duration)

        dict_data[key_] = value_

    return dict_data


def insert_blank_line(array2d, line_num):
    """ 在缺少的array中插入空行, 仅用于hysplit数据导出时 
        array2d: 缺少某个轨迹时长的二维numpy数组
        [[0  40.0  110.1  100]
         [-1 41.1  112.2  120]
         [-2 42.3  113.1  120]
         [-4 43.1  114.2  120] 
         ...
        ];
        line_num: 总轨迹长度, 前向轨迹为正, 后向轨迹为负;

    2022-10-25    v1.0
    """

    array2d_ = array2d

    # 生成空行
    nan_row = np.ones(array2d.shape[1])
    nan_row[:] = -999

    if line_num > 0:
        list_lack = [i for i in range(0, line_num + 1, 1) if i not in array2d[:, 0]]
    else:
        list_lack = [i for i in range(0, line_num - 1, -1) if i not in array2d[:, 0]]

    for i in list_lack:
        value_i = nan_row.copy()
        value_i[0] = i
        # print('before', array2d_, value_i)
        array2d_ = np.insert(array2d_, abs(i), values=value_i, axis=0)
        # if line_num > 0:
        #     array2d_ = np.insert(array2d_, i-1, values=value_i, axis=0)
        # else:
        #     array2d_ = np.insert(array2d_, abs(i), values=value_i, axis=0)
        # print('after', array2d_)

    return array2d_


def round_accurately(num, n_digits):
    """ 浮点数精确四舍五入, 为了解决python中round(1.315, 2)结果是1.31, 而不是1.32的问题;
        num: 需要四舍五入的浮点数;
        n_digits: 需要保留的小数位数;

    2022-10-20    v1.0
    """

    d = '0.'.ljust(n_digits + 2, '0')
    return Decimal(str(num)).quantize(Decimal(d), rounding=ROUND_HALF_UP)


def coords2str(coord: tuple[float, float]):
    """ 将经纬度转换为字符串, 保留3位小数
    
        coord: 经纬度元组/列表，(lat, lon)或者(lon, lat)，单位：°

    2024-09-04  v1
    """

    return ','.join([str(round_accurately(i, 3)) for i in coord])
