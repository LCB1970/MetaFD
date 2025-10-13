DOMAIN_SETS = {
    "DAB": ['WHU-B/WHU-S2', 'WHU-B/WHU-S1', 'WHU-B/WHU-A', 'MASS-B', 'IAIL/IAIL-A', 'IAIL/IAIL-C',
            'IAIL/IAIL-K', 'IAIL/IAIL-T', 'IAIL/IAIL-V'],
    "DAR": ['DG-R', 'WHU-R/WHU-R1', 'WHU-R/WHU-R2', 'MASS-R', 'ERM-PAIW/EP-A', 'ERM-PAIW/EP-B'],
    "DABR": ['CITY-OSM/CO-PA', 'CITY-OSM/CO-PO', 'CITY-OSM/CO-C', 'CITY-OSM/CO-B', 'CITY-OSM/CO-T',
             'CITY-OSM/CO-Z'],
}
FILE_FORMAT = {
    'WHU-B/WHU-S2': '/*.tif', 'WHU-B/WHU-S1': '/*.tif', 'WHU-B/WHU-A': '/*.tif',
    'MASS-B': '/*.tif', 'IAIL/IAIL-A': '/*.tif', 'IAIL/IAIL-C': '/*.tif',
    'IAIL/IAIL-K': '/*.tif', 'IAIL/IAIL-T': '/*.tif', 'IAIL/IAIL-V': '/*.tif',
    'DG-R': '/*.tif', 'WHU-R/WHU-R1': '/*.png', 'WHU-R/WHU-R2': '/*.png',
    'MASS-R': '/*.tif', 'ERM-PAIW/EP-A': '/*.tif',
    'ERM-PAIW/EP-B': '/*.tif',
    'CITY-OSM/CO-PA': '/*.png', 'CITY-OSM/CO-PO': '/*.png', 'CITY-OSM/CO-C': '/*.png',
    'CITY-OSM/CO-B': '/*.png',
    'CITY-OSM/CO-T': '/*.png', 'CITY-OSM/CO-Z': '/*.png',
}

LABEL_TYPE1 = ['WHU-S2', 'WHU-R1', 'WHU-R2', 'MASS-R', 'IAIL-A', 'IAIL-C', 'IAIL-K', 'IAIL-T',
               'IAIL-V']  # shape = (512,512), max = 255, min = 0
LABEL_TYPE2 = ['WHU-A']  # shape = (512,512), max = 1, min = 0
LABEL_TYPE3 = ['WHU-S1', 'CO-PA', 'CO-B', 'CO-C', 'CO-T', 'CO-Z', 'CO-PO',
               'DG-R', 'MASS-B', 'EP-A', 'EP-B']  # shape = (512,512,3), 需要转换为灰度

COLORMAP = {
    'WHU-S2': [[0, 0, 0], [255, 255, 255]],
    'WHU-S1': [[0, 0, 0], [255, 255, 255]],
    'WHU-A': [[0, 0, 0], [255, 255, 255]],
    'CO-PA': [[255, 255, 255], [255, 0, 0], [0, 0, 255]],
    'CO-PO': [[255, 255, 255], [255, 0, 0], [0, 0, 255]],
    'CO-B': [[255, 255, 255], [255, 0, 0], [0, 0, 255]],
    'CO-C': [[255, 255, 255], [255, 0, 0], [0, 0, 255]],
    'CO-T': [[255, 255, 255], [255, 0, 0], [0, 0, 255]],
    'CO-Z': [[255, 255, 255], [255, 0, 0], [0, 0, 255]],
    'DG-R': [[0, 0, 0], [255, 255, 255]],
    'WHU-R1': [[0, 0, 0], [255, 255, 255]],
    'WHU-R2': [[0, 0, 0], [255, 255, 255]],
    'MASS-B': [[0, 0, 0], [255, 0, 0]],
    'MASS-R': [[0, 0, 0], [255, 255, 255]],
    'EP-A': [[0, 0, 0], [255, 255, 255]],
    'EP-B': [[0, 0, 0], [255, 255, 255]],
    'IAIL-A': [[0, 0, 0], [255, 255, 255]],
    'IAIL-C': [[0, 0, 0], [255, 255, 255]],
    'IAIL-K': [[0, 0, 0], [255, 255, 255]],
    'IAIL-T': [[0, 0, 0], [255, 255, 255]],
    'IAIL-V': [[0, 0, 0], [255, 255, 255]],
}

CLASS_NAME = {
    'WHU-S2': ['background', 'building'],
    'WHU-S1': ['background', 'building'],
    'WHU-A': ['background', 'building'],
    'CO-PA': ['background', 'building', 'road'],
    'CO-B': ['background', 'building', 'road'],
    'CO-C': ['background', 'building', 'road'],
    'CO-T': ['background', 'building', 'road'],
    'CO-PO': ['background', 'building', 'road'],
    'CO-Z': ['background', 'building', 'road'],
    'DG-R': ['background', 'road'],
    'WHU-R1': ['background', 'road'],
    'WHU-R2': ['background', 'road'],
    'MASS-B': ['background', 'building'],
    'MASS-R': ['background', 'road'],
    'EP-A': ['background', 'building'],
    'EP-B': ['background', 'building'],
    'IAIL-A': ['background', 'building'],
    'IAIL-C': ['background', 'building'],
    'IAIL-K': ['background', 'building'],
    'IAIL-T': ['background', 'building'],
    'IAIL-V': ['background', 'building']
}
