""" This package is for reading the site information saved in 'NewSiteList.csv' """

from __future__ import annotations
import numpy as np
import pandas as pd
from itertools import chain
from pathlib import Path


class SiteInfo:
    """读取站点信息，并以字典返回

        监测点编码,监测点名称,城市,经度,纬度,对照点,省级行政区,地级行政区
        1001A,万寿西宫,北京,116.3621,39.8784,N,北京市,北京市
        1002A,定陵(对照点),北京,116.2202,40.2915,Y,北京市,北京市
        ...

    2022.08.16  v1.0
    2022.10.08  从ml_io.py文件中转移，之后仅在此文件中更新
    2022.11.08  从ml_geo.py文件中复制, 之后分开更新
    2022.11.08  新增省->市的字典映射
    2023.05.12  从pkg_mean.py中转移，以后在此主要更新

    单进程
    """

    def __init__(self, path_site_list=None):

        # 读取站点数据
        if path_site_list:
            self.path_site_list = path_site_list
        else:
            self.path_site_list = Path(__file__).parent / 'NewSiteList.csv'

        self.df_info = pd.read_csv(self.path_site_list, engine="c", low_memory=False, encoding="gbk")

        # 监测点编码与经纬度对应 {'1001A': [latitude, longitude]}
        list_code_all = self.df_info.loc[:, "监测点编码"].to_numpy()
        list_coordinate = self.df_info.loc[:, ["纬度", "经度"]].to_numpy()
        self.dict_code2coordinate = dict(zip(list_code_all, list_coordinate))

        # 监测点编码与站点名称对应 {'1001A': '万寿西宫'}
        list_code_name = self.df_info.loc[:, "监测点名称"].to_numpy().tolist()
        self.dict_code2name = dict(zip(list_code_all, list_code_name))

        # 监测点编码与省级行政区对应 {'1001A': '北京市', '1047A': '河北省'}，注意：直辖市名称会同时出现在省级行政区和地级行政区
        list_code_province = self.df_info.loc[:, "省级行政区"].to_numpy().tolist()
        self.dict_code2province = dict(zip(list_code_all, list_code_province))

        # 监测点编码与地级行政区对应 {'1001A': '北京市', '1047A': '邯郸市'}
        # list_code_city = self.df_info.loc[:, '地级行政区'].to_numpy().tolist()
        self.list_city_china = self.df_info.loc[:, "地级行政区"].to_numpy().tolist()
        self.dict_code2city = dict(zip(list_code_all, self.list_city_china))

        # 地级行政区与所有站点列表对应 {'北京市': ['1001A', '1002A', '1003A', ...]}
        self.dict_city2code = self.df_info.groupby(self.list_city_china).apply(lambda x: x.loc[:, "监测点编码"].to_numpy()).to_dict()

        # 监测点编码是否为对照点 {'1001A': False, '1002A': True, ... }
        list_code_background = self.df_info.loc[:, "对照点"].to_numpy()
        list_code_background = [True if i == "Y" else False for i in list_code_background]
        self.dict_code2background = dict(zip(list_code_all, list_code_background))

        # 全国所有城市列表去重
        self.list_city_china = sorted(set(self.list_city_china), key=self.list_city_china.index)

        # 全国所有站点
        self.list_code_china = list(chain(*[self.dict_city2code[i] for i in self.list_city_china]))

        # 全国所有省份，省市名对应
        self.list_province_china = self.df_info.loc[:, "省级行政区"].to_numpy()  # 有重复
        self.dict_province2city = self.df_info.groupby(self.list_province_china).apply(lambda x: np.unique(x.loc[:, "地级行政区"].to_numpy()).tolist()).to_dict()  # 省市名对应字典
        self.list_province_china = self.list_province_china[np.sort(np.unique(self.list_province_china, return_index=True)[1])]  # 省名去重

        # 74城市，按照《环境空气质量标准》（GB3095-2012），74个城市为2012年第一批实施新空气质量标准的城市
        self.list_city_74 = [
            '北京市', '天津市', '石家庄市', '唐山市', '秦皇岛市', '邯郸市', '邢台市', '保定市',
            '张家口市', '承德市', '沧州市', '廊坊市', '衡水市', '太原市', '呼和浩特市', '沈阳市',
            '大连市', '长春市', '哈尔滨市', '上海市', '南京市', '无锡市', '徐州市', '常州市', '苏州市',
            '南通市', '连云港市', '淮安市', '盐城市', '扬州市', '镇江市', '泰州市', '宿迁市', '杭州市',
            '宁波市', '温州市', '嘉兴市', '湖州市', '绍兴市', '金华市', '衢州市', '舟山市', '台州市',
            '丽水市', '合肥市', '福州市', '厦门市', '南昌市', '济南市', '青岛市', '郑州市', '武汉市',
            '长沙市', '广州市', '深圳市', '珠海市', '佛山市', '江门市', '肇庆市', '惠州市', '东莞市',
            '中山市', '南宁市', '海口市', '重庆市', '成都市', '贵阳市', '昆明市', '拉萨市', '西安市',
            '兰州市', '西宁市', '银川市', '乌鲁木齐市'
        ]

        # 京津冀BTH
        self.list_city_jjj = ['北京市', '天津市'] + self.dict_province2city['河北省']

        # "2+26"城市所有站点（与环境状况公报一致，共28个城市）
        self.list_city_2p26 = [
            "北京市", "天津市", "石家庄市", "唐山市",
            "廊坊市", "保定市", "沧州市", "衡水市",
            "邢台市", "邯郸市", "太原市", "阳泉市",
            "长治市", "晋城市", "济南市", "淄博市",
            "济宁市", "德州市", "聊城市", "滨州市",
            "菏泽市", "郑州市", "开封市", "安阳市",
            "鹤壁市", "新乡市", "焦作市", "濮阳市",
        ]
        self.list_code_2p26 = np.hstack([self.dict_city2code[i] for i in self.list_city_2p26])

        # 珠三角（与环境状况公报一致，共9个城市）
        self.list_city_prd = [
            "广州市", "深圳市", "珠海市",
            "佛山市", "中山市", "江门市",
            "东莞市", "惠州市", "肇庆市",
        ]

        # 长三角（与环境状况公报一致，共41个城市）
        list_province_yrd = ["上海市", "江苏省", "浙江省", "安徽省"]
        self.list_city_yrd = list(chain(*[self.dict_province2city[i] for i in list_province_yrd]))

        # 四川盆地
        self.list_city_scb = [
            "重庆市", "成都市", "绵阳市", "泸州市",
            "南充市", "自贡市", "德阳市", "广元市",
            "遂宁市", "内江市", "乐山市", "宜宾市",
            "广安市", "达州市", "雅安市", "巴中市",
            "眉山市", "资阳市", "遵义市", "毕节市",
        ]

        # 汾渭平原（与环境状况公报一致，共11个城市）
        self.list_city_fwp = [
            "晋中市", "运城市", "临汾市", "吕梁市",
            "洛阳市", "三门峡市", "西安市", "铜川市",
            "宝鸡市", "咸阳市", "渭南市",
        ]

        # 粤港澳大湾区 the Greater Bay Area, 暂无香港和澳门的观测数据，此处不添加
        self.list_city_gba = ['广州市', '深圳市', '珠海市', '佛山市', '惠州市', '东莞市', '中山市', '江门市', '肇庆市']


if __name__ == '__main__':
    
    """ Test """
    # Path
    path_csv = r'NewSiteList.csv'

    # calling SiteInfo class
    si = SiteInfo(path_csv)

    # print all site code in China
    print(si.list_code_china)

    # print the site code in HaiNan province
    for city in si.dict_province2city['海南省']:
        print(f'city: {city}, code: {si.dict_city2code[city]}')

    # print the coordinates of the 1410A site
    print(si.dict_code2coordinate['1410A'])
