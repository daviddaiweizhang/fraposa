import fraposa as fp
import argparse, sys

parser=argparse.ArgumentParser()
parser.add_argument('ref_filepref', help='Prefix of binary PLINK file for the reference data.')
parser.add_argument('stu_filepref', help='Prefix of binary PLINK file for the study data.')
parser.add_argument('--nneighbors', help='The number of neighbors for each study sample. Default is 20.')
parser.add_argument('--weights', help='The method for calculating the weights in the nearest neighbor method. uniform: each neighbor receives the same weight; distance: the weight of each neighbor is inversly proportional to its distance from the study sample. Default is uniform.')
args=parser.parse_args()

weights = 'uniform'
n_neighbors = 20
if args.weights:
    weights = args.weights
if args.nneighbors:
    n_neighbors = int(args.nneighbors)

fp.pred_popu_stu(args.ref_filepref, args.stu_filepref, n_neighbors, weights)
