#! /usr/bin/env python

import fraposa as fp
import argparse, sys

parser=argparse.ArgumentParser()
parser.add_argument('ref_filepref', help='Prefix of binary PLINK file for the reference data.')
parser.add_argument('stu_filepref', help='Prefix of binary PLINK file for the study data.')
args=parser.parse_args()

fp.plot_pcs(args.ref_filepref, args.stu_filepref)
