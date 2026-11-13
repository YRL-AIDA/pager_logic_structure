from dotenv import load_dotenv
import os
from pathlib import Path
load_dotenv()
PATH_FINTOC_2022 = Path(os.getenv('PATH_FINTOC_2022'))

FINTOC_EN = PATH_FINTOC_2022/'data'/'en'
FINTOC_EN_PDF =FINTOC_EN/'pdf'
FINTOC_EN_ANNOTS = FINTOC_EN/'annots'
FINTOC_EN_PRED = FINTOC_EN/'preds'

JSON_EXTENSION = "fintoc4.json"

if FINTOC_EN_PRED.exists():
    import shutil
    shutil.rmtree(FINTOC_EN_PRED)

FINTOC_EN_PRED.mkdir()
FILE_PATHS = list(FINTOC_EN_PDF.iterdir())
FILE_PATHS.sort()
FILE_PATHS = FILE_PATHS[:10]

import json
from typing import Dict

from pagerlib.file_input import FileInput
from pagerlib.extractors.document_extractor import LogicalStructureExtractor
from pagerlib.extractors.page_extractor import PDFIMGExtractor, Rows2Regions
from pagerlib.dtypes import Section
from pagerlib.extractors.page_extractor.rows2regions.rows2region_glam_20260826 import get_load_model

fi = FileInput()
img_extr = PDFIMGExtractor()
rows2regions = Rows2Regions()
rows2regions.model = get_load_model(path='fine-tuned.pt')
regions2tree = LogicalStructureExtractor()

def pagerlib_pdf2toc(file_path):
    prdf = fi(file_path)
    img_extr.extract(prdf)
    rows2regions.extract(prdf)
    regions2tree.extract(prdf)
    return prdf.data['toc']

def get_header(sec):
    headers = []
    children = sec.children
    for child in children:
        if isinstance(child, Section):
            reg = child.title
            reg.data['level'] = child.level
            headers.append(reg)
            headers.extend(get_header(child))
    return headers


def toc2fintoc_json(toc) -> Dict:
    header_regs = get_header(toc)
    fintoc_json = []
    for id_node, node in enumerate(header_regs):
        reg = {
            "text": node.text,
            "id": id_node,
            "page": node.data['page_num'] + 1, # В PageR специально оставил нумерацию с нуля (поскольку номер определяется не порядком, а пишится на странице и это еще не реализовано)
            "depth": node.data['level']
        }
        fintoc_json.append(reg)
    return fintoc_json    


N = len(FILE_PATHS)
for i, file_path in enumerate(FILE_PATHS):    
    toc = pagerlib_pdf2toc(file_path)
    pred_path = FINTOC_EN_PRED/(file_path.name+'.'+JSON_EXTENSION)
    rez = toc2fintoc_json(toc)
    with open(pred_path, 'w') as f:
        json.dump(rez, f)

    print(f'{i+1}/{N} ({(i+1)/N*100:.2f}%)')


import subprocess
import pandas as pd

subprocess.run(['python', 'metric.py', '--gt_folder', FINTOC_EN_ANNOTS, '--submission_folder', FINTOC_EN_PRED])


df = pd.read_csv('td_report.csv', delimiter='\t')
print( df[-2:-1][['Prec', 'Rec', 'F1']])


df = pd.read_csv('toc_report.csv', delimiter='\t')
tbl = df[-4:-3]
print(tbl[['Inex08-P', 'Inex08-R', 'Inex08-F1', 'Inex08-Title acc', 'Inex08-Level acc']])
