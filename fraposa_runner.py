import fraposa as fp
import argparse, sys

parser=argparse.ArgumentParser()
parser.add_argument('ref_filepref', help='Prefix of the binary PLINK file for the reference samples.')
parser.add_argument('stu_filepref', help='Prefix of the binary PLINK file for the study samples.')
parser.add_argument('--method', help='The method for PCA prediction. oadp: most accurate. ap: fastest. adp: accurate but slow. sp: fast but inaccurate. Default is odap.')
parser.add_argument('--dim_ref', help='Number of PCs you need.')
parser.add_argument('--dim_stu', help='Number of PCs predicted for the study samples before doing the Procrustes transformation. Only needed for the oadp and adp methods. Default is 2*dim_ref.')
parser.add_argument('--dim_online', help='Number of PCs to calculate in online SVD. Only needed for the oadp method. Default is 2*dim_stu')
parser.add_argument('--dim_spikes', help='Number of PCs to adjust for shrinkage. Only needed for the ap method. If this argument is not set, dim_spikes_max will be used.')
parser.add_argument('--dim_spikes_max', help='The maximal number of PCs to adjust for shrinkage. Only needed for the ap method. This argument will be ignored if dim_spikes is set. Default is 4*dim_ref.')
parser.add_argument('--out', help='Prefix of output file(s). Default is stu_filepref')
args=parser.parse_args()

ref_filepref = args.ref_filepref
stu_filepref = args.stu_filepref
out_filepref = stu_filepref
method = 'oadp'
dim_ref = 4
dim_stu = None
dim_online = None
dim_spikes = None
dim_spikes_max = None

if args.out:
    out_filepref = args.out
if args.method:
    method = args.method
if args.dim_ref:
    dim_ref = int(args.dim_ref)
if args.dim_stu:
    dim_stu = int(args.dim_stu)
if args.dim_online:
    dim_online = int(args.dim_online)
if args.dim_spikes:
    dim_spikes = int(args.dim_spikes)
if args.dim_spikes_max:
    dim_spikes_max = int(args.dim_spikes_max)

fp.pca(ref_filepref, stu_filepref, out_filepref, method,
       dim_ref, dim_stu, dim_online, dim_spikes, dim_spikes_max)
