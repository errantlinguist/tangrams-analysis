#!/usr/bin/env python3

import argparse
import csv
import sys
import tkinter as tk
from numbers import Number
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageTk

INFILE_CSV_DIALECT = csv.excel_tab
INFILE_ENCODING = 'utf-8'

_INFILE_DTYPES = {"DYAD": "category", "ENTITY": "category", "WORD": "category", "IS_OOV": bool, "SHAPE": "category",
				  "TARGET": bool}


def draw_image():
	pass


def read_dataframe_file(inpath: str) -> pd.DataFrame:
	return pd.read_csv(inpath, encoding=INFILE_ENCODING, dialect=INFILE_CSV_DIALECT, sep=INFILE_CSV_DIALECT.delimiter,
					   float_precision="round_trip", memory_map=True, dtype=_INFILE_DTYPES)


def read_dataframe_files(inpaths: Iterable[str]) -> pd.DataFrame:
	dfs = (read_dataframe_file(inpath) for inpath in inpaths)
	# noinspection PyTypeChecker
	return pd.concat(dfs)


def __create_argparser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(
		description="Cross-validation of reference resolution for tangram sessions.")
	result.add_argument("inpaths", metavar="INPATH", nargs='+',
						help="The directories to process.")
	return result


def __replace_red(array: np.array, v: Number):
	array[:, :, 0] = v


def __replace_green(array: np.array, v: Number):
	array[:, :, 1] = v


def __replace_blue(array: np.array, v: Number):
	array[:, :, 2] = v


def create_filtered_image(img: Image.Image, rgb: Tuple[Number, Number, Number]) -> Image.Image:
	rgb_img = img.convert("RGBA")
	data = np.array(rgb_img.getdata())
	__replace_red(data, rgb[0])
	__replace_green(data, rgb[1])
	__replace_blue(data, rgb[2])
	return Image.fromarray(data, mode='RGBA')


def __main(args):
	inpaths = args.inpaths
	print("Will read {} path(s).".format(len(inpaths)), file=sys.stderr)
	# dataframe = read_dataframe_files(inpaths)
	# print("Read {} dataframe row(s).".format(dataframe.shape[0]), file=sys.stderr)
	root = tk.Tk()
	img = Image.open("D:\\Users\\tcshore\\Documents\\Projects\\Tangrams\\Data\\Tangram images\\1200-squares\\arrow.png")
	colored_img = create_filtered_image(img, (255, 0, 200))
	tkimage = ImageTk.PhotoImage(colored_img)
	tk.Label(root, image=tkimage).pack()
	root.mainloop()


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
