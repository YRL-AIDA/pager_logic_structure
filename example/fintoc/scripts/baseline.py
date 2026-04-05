# -*- coding: utf-8 -*-

"""
SVM on the following layout features:
    - is_bold
    - is_italic
    - is_all_caps
"""

import pandas as pd
import subprocess
import click
import pathlib
import shlex
import os
import json
import pickle
from operator import itemgetter
from collections import Counter

import Levenshtein
import numpy as np
from sklearn.svm import SVC
from dataclasses import dataclass

from fortia.library.document.parser import DocumentParser


STRING_THRESHOLD = 0.85


def toxml(input_path, output_path, ignoreimg=True, zoom=1):
    command = f"pdftohtml -c -s {'-i ' if ignoreimg else ''}-xml -nodrm -zoom {zoom} -hidden -noroundcoord '{input_path}' '{output_path}'"
    encoding = os.device_encoding(1) or "cp437"
    args = shlex.split(command)
    encoding = os.device_encoding(1) or "cp437"
    subprocess.run(args, shell=False, stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE, check=True, encoding=encoding)


def tb_to_vec(tb):
    """Feature extractor: is_bold, is_italic, is_all_caps"""
    out = []
    for flag in ["b", "i", "c"]:
        if flag in tb.style:
            out.append(1)
        else:
            out.append(0)
    return out


class TOCJson:
    def __init__(self, json_file):
        self.parse(json_file)

    def parse(self, json_file):
        with open(json_file, 'r', encoding='utf-8') as infile:
            content = json.load(infile)
        self.entries = []
        for dict_entry in content:
            self.entries.append(Title(dict_entry["text"],
                                      dict_entry["page"],
                                      dict_entry["id"],
                                      dict_entry["depth"]))


@dataclass
class Title:
    text: str
    page_nb: int
    id_: int
    depth: int
    matched: bool = False

    def __repr__(self):
        return f"page={self.page_nb} title={repr(self.text)}"


def find_matching_entry(tb, toc):
    similarities = []
    for entry in toc.entries:
        if not entry.matched and entry.page_nb == tb.page:
            similarities.append(Levenshtein.ratio(tb.clean_text.strip(),
                                                  entry.text))
        else:
            similarities.append(0)
    index, match_score = max(enumerate(similarities), key=itemgetter(1))
    if match_score > STRING_THRESHOLD:
        return index, match_score


def get_gt_labels_by_fuzzy_matching(tbs, toc):
    labels = []
    for tb in tbs:
        res = find_matching_entry(tb, toc)
        if res is not None:
            index, match_score = res
            toc.entries[index].matched = True
            labels.append(1)
            print(f"{tb} matched to gt title {toc.entries[index]} and thus assigned label 1")
        else:
            print(f"{tb} not matched to any gt title and thus assigned label 0")
            labels.append(0)
    return labels


class Helper:

    def __init__(self, pdf_folder):
        self.pdf_folder = pathlib.Path(pdf_folder).resolve()
        self.xml_folder = self.pdf_folder.parent / "xmls"
        self.xml_folder.mkdir(exist_ok=True)
        self.ann_folder = self.pdf_folder.parent / "annots"
        self.ann_folder.mkdir(exist_ok=True)
        self.pred_folder = self.pdf_folder.parent / "preds"
        self.pred_folder.mkdir(exist_ok=True)
        self.pdfs = list(self.pdf_folder.glob("**/*.pdf"))
        self.anns = []
        for pdf in self.pdfs:
            ann = self.ann_folder / pdf.stem
            ann = ann.parent / (ann.name + ".fintoc4.json")
            ann_toc = TOCJson(ann)
            self.anns.append(ann_toc)
        self.doc_parser = DocumentParser()

    def vectorize(self, pdf, ann_toc, assign_td_labels=False):
        xml = self.xml_folder / pdf.stem
        xml = xml.with_suffix(".xml")
        toxml(pdf, xml)
        tbs = self.doc_parser.parse(xml).get_all_textblocks()
        vecs = [tb_to_vec(tb) for tb in tbs]
        if assign_td_labels:
            labels = get_gt_labels_by_fuzzy_matching(tbs, ann_toc)
        else:
            labels = []
        return vecs, tbs, labels

    def vectorize_all(self, assign_td_labels=False):
        all_vecs = []
        all_tbs = []
        all_labels = []
        for i, pdf in enumerate(self.pdfs):
            vecs, tbs, labels = self.vectorize(pdf, self.anns[i],
                                               assign_td_labels)
            all_vecs.extend(vecs)
            all_tbs.extend(tbs)
            all_labels.extend(labels)
        return all_tbs, all_vecs, all_labels

    def get_most_frequent_toc_level(self):
        depths = []
        for ann in self.anns:
            depths.extend([entry.depth for entry in ann.entries])
        counter = Counter(depths)
        return counter.most_common(1)[0][0]

    def write_prediction_file(self, tbs, td_labels, toc_labels, pred_filename):
        content = []
        running_id = 1
        for tb, td_label, toc_label in zip(tbs, td_labels, toc_labels):
            if td_label:
                dic = {
                    "text": tb.clean_text,
                    "page": tb.page,
                    "id": running_id,
                    "depth": toc_label,
                }
                content.append(dic)
                running_id += 1
        out_filename = self.pred_folder / pred_filename
        with open(out_filename, "w", encoding="utf-8") as outfile:
            json.dump(content, outfile, indent=2)
        return out_filename


def dump_vectorized_set(tbs, vecs, labels, folder):
    content = {
        "textblock": [],
        "tb_page_nb": [],
        "is_bold": [],
        "is_italic": [],
        "is_all_caps": [],
        "is_title": [],
    }
    for tb, vec, label in zip(tbs, vecs, labels):
        content["textblock"].append(tb.clean_text)
        content["tb_page_nb"].append(tb.page)
        content["is_bold"].append(vec[0])
        content["is_italic"].append(vec[1])
        content["is_all_caps"].append(vec[2])
        content["is_title"].append(label)
    content = pd.DataFrame(content)
    filename = str(folder / "vectorized.xlsx")
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    content.to_excel(writer, index=False)
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format_ = workbook.add_format({'text_wrap': True})
    worksheet.set_column('A:A', 90, format_)
    worksheet.set_column('B:B', 15, format_)
    worksheet.set_column('C:F', 10, format_)
    writer.save()
    print(f"Vectorized dataset is written to {str(filename)}")


@click.command()
@click.option('--train_pdf_folder', type=str)
@click.option('--test_pdf_folder', type=str)
def main(train_pdf_folder, test_pdf_folder):
    train = Helper(train_pdf_folder)
    print("\nVectorizing training set...")
    train_tbs, train_vecs, train_labels = train.vectorize_all(True)
    dump_vectorized_set(train_tbs, train_vecs, train_labels, train.ann_folder)
    print("Vectorizing training set...DONE")
    print("\nOptimizing a SVM classifier...")
    classifier = SVC(gamma='scale')
    classifier.fit(np.array(train_vecs), np.array(train_labels))
    print("Optimizing a SVM classifier...DONE")
    print("\nSaving the SVM classifier...")
    with open(train.ann_folder / "svm.pk", "wb") as pkf:
        pickle.dump(classifier, pkf)

    print("Saving the SVM classifier...DONE")
    print("\nComputing most frequent hierarchy label in the training set...")
    hierarchy_label = train.get_most_frequent_toc_level()
    print(f"Most frequent hierarchy label is {hierarchy_label}")
    print("Computing most frequent hierarchy label in the training set...DONE")
    print("\nInfering on private dataset...")
    test = Helper(test_pdf_folder)
    for pdf in test.pdfs:
        print(f"\nProcessing {pdf.stem}")
        vecs, tbs, _ = test.vectorize(pdf, ann_toc=None)
        predicted_td_labels = classifier.predict(np.array(vecs))
        predicted_toc_labels = [hierarchy_label]*len(tbs)
        pred_filename = pdf.stem + ".fintoc4.json"
        out_filename = test.write_prediction_file(tbs, predicted_td_labels,
                                                  predicted_toc_labels,
                                                  pred_filename)
        print(f"Prediction file for {pdf.stem} is written to {out_filename}")
    print("Infering on private dataset...DONE")


if __name__ == '__main__':
    main()


