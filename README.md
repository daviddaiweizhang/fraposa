FRAPOSA
(Fast and Robutst Ancestry Prediction by using Online singular value decomposition and Shrinkage Adjustment)
predicts the ancestry of study samples
by using principle component analysis (PCA) with a reference panel.
The software accompanies
[this paper](https://doi.org/10.1093/bioinformatics/btaa152).

# Example

Run
```
./run_example.sh
```
to see an example of using FRAPOSA.

The script will download the reference and study data,
predict the study samples' PC scores and ancestry memberships,
and plot the results.

# Software requirements

- Python 3
    - numpy
    - pandas
    - scikit-learn
    - pyplink
    - matplotlib
    - rpy2
- R
    - hdpca
- PLINK 1.9

# Inputs and Outputs

## Input files

- Binary PLINK files for the reference set: `refpref_raw.{bed,bim,fam}`
- Binary PLINK files for the study set: `stupref_raw.{bed,bim,fam}`

      - If no study set is given, FRAPOSA will only run PCA on the reference set and output the reference PC scores.
      
- Reference population membership: refpref_raw.popu

      - Without this file, the study PC scores will still be computed, but you will not be able to predict the population memberships for the study samples.
      - Format
          - Column 1 and 2: Family and individual IDs (same as in `refpref_raw.fam`)
          - Column 3: Population membership label 

## Output files

- Reference PC scores: `refpref.pcs`
- Study PC scores: `stupref.pcs`
- PC plot: `stupref.png`


# Preprocessing

## Extract common variants

The reference and study samples must have the same set of variants
(i.e. the two `.bim` files must be identical).
To extract the common variants between two datasets,
you can use PLINK manually
or use the included `commvar.sh` script:
```
./commvar.sh refpref_raw stupref_raw refpref stupref
```
This command will find the common variants in 
`refpref_raw.{bed,bim,fam}` and `stupref_raw.{bed,bim,fam}`
and then output the intersected datasets
in `refpref.{bed,bim,fam}` and `stupref.{bed,bim,fam}`.

## Split study samples

FRAPOSA loads all the study samples into memory.
If the study set is too large,
its samples can be split into smaller batches.
Then FRAPOSA can be run on each batch
sequentially or embarrassingly parallelly.
Just as for extracting the common variants,
you can split the study samples manually using PLINK
or run the included script `splitindiv.sh`: 
```
./splitindiv.sh stupref n i stupref_batch_i
```
which divides the samples in `stupref.{bed,bim,fam}`
evenly into $n$ batches
and saves the samples in the $i^\text{th}$ batch
into `stupref_batch_i.{bed,bim,fam}`
For example,
if `stupref.{bed,bim,fam}` has 100,000 samples,
then
```
./splitindiv.sh stupref 100 12 stupref_batch_12
```
produces `stupref_batch_12.{bed,bim,fam}`
that contains sample 12,001 to 13,000.
To generate all the 100 batches,
you can use
```
for i in `seq 1 100`; do
  ./splitindiv.sh stupref 100 $i stupref_batch_$i
done;
```

# Running FRAPOSA

To use FRAPOSA with the default settings, run
```
./fraposa_runner.py --stu_filepref stupref refpref 
```
This will produce `refpref.pcs`,
which contains the IDs and reference PC scores,
and `stupref.pcs`,
which contains the IDs and the study PC scores.
Some intermediate files are also produced
to reduce the computatio time for future usage.


## Change analysis method


FRAPOSA includes 4 methods for ancestry prediction,

1. **OADP** (default and recommended):
This method is fast and provides robust PC score prediction
by using the online SVD algorithm.

2. **AP** (also recommended):
This method is even faster and its results are close to OADP's.
However, sometimes you may want to manually set the number of PCs to be adjusted for shrinkage
(i.e. by setting `--dim_spikes`)
if you believe that a PC has been shrinked but has not been adjusted automatically.

3. **SP** (fast but inaccurate):
This method is similar to AP
and is the standard method of PC prediction.
It computes the PC loadings of the reference set
and projects the study PC scores onto them.
Its speed is the same as AP but does not adjust for the shrinkage bias,
which makes it inaccurate when the number of variants greatly exceeds the sample size. 

4. **ADP** (accurate but slow):
This method is similar to OADP but has a much higher computation complexity.
While OADP only computes the top few PCs,
ADP computes all the PCs
(i.e. running a full eigendecomposition for every study sample).
The results are very close to OADP's.


To change the analysis method, set the `--method` option. For example,
```
./fraposa_runner.py --stu_filepref stupref --method ap refpref 
```

## Change the other parameters

Several PCA-related parameters can be changed.
For example, to set the number of reference PCs to 20, run
```
./fraposa_runner.py --stu_filepref stupref --dim_ref 20 refpref 
```
To learn all the options for FRAPOSA, run
```
./fraposa_runner.py --help
```

## Important: Remove the intermediate files

If you have run FRAPOSA previously by using

1. the same reference set, and
2. different parameter settings
(e.g. by changin `--dim_ref` or `--dim_stu`),

then you need to delete all the intermediate `.dat` files with the same prefix as this reference set.

FRAPOSA saves the intermediate files
related to PCA on the reference set.
Specifically,
the mean and standard deviation of each variant (`refpref_mnsd.dat`),
singular values (`refpref_s.dat`),
reference PC loadings (`refpref_U.dat`),
scaled (`refpref_V.dat`) and unscaled (`refpref_Vs.dat`) reference PC scores
are saved
and will be automatically loaded
if the same reference set is used again.
This avoids running PCA on the same reference set for multiple times,
especially in the case when the study samples are split into batches
and are analyzed with the same reference set.
However,
FRAPOSA only checks whether the reference set file prefix is the same
when deciding whether to load the intermediate files.
It does *not* detect whether the parameters have been changed.


# Postprocessing

## Predict ancestry memberships

After predicting the study samples' PC scores,
their ancestry memberships can also be predicted,
if the reference ancestry information `refpref.popu` is provided.
Running
```
./predstupopu.py refpref stupref
```
will produce `stupref.popu`,
which contains

- IDs (columns 1 and 2)
- the population that the study sample most likely belongs to (column 3)
- how likely the study sample belongs to the population in column 3 (column 4)
- The distance between the study sample and the nearest reference sample (column 5)
- How likely the study sample belongs to each of the refrence populations
(columns 6 to 6+k-1, where k is the number of reference populations)
- Names of the refrence populations. This is the same in every row. (columns 6+k to 6+2k-1)

Population prediction is done by using the k-nearest neighbor algorithm
to classify the study PC scores with respect to the reference PC scores.
You can set the number of neighbors or the method for calculating the weights by using
```
./predstupopu.py --nneighbors 20 --weights uniform refpref stupref 
```

## Plot the PC scores
A simple script for plotting the PC scores is included:
```
./plotpcs.py refpref stupref
```
The PC plot will be saved to `refpref.png`.

# Data

A reference data set (`umich.edu/~daiweiz/fraposa/data/thousandGenomes.{bed,bim,fam}`) is included for your convenience.
We took the 2,492 unrelated samples from the 1000 Genomes project and selected the 637,177 SNPs that are included in the Human Genome Diversity Project.
The population memberships of the samples are also included.
