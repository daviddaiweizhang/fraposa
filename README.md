
# Preprocessing

## Intersect the variants of the reference and the study samples

Before using FRAPOSA,
the reference samples and the study samples must have the same set of variants
(i.e. the two .bim files are identical).
This can be achieved by using the script `commvar.sh` as follows:
```
bash commvar.sh ref_bedprefix stu_bedprefix ref_bedprefix_comm stu_bedprefix_comm
```

## Spliting the study data set into smaller ones by samples (optional)

If the study data set is too large to load into memory,
you can split it into multiple data sets with fewer samples in each.
This can be done by using the `splitindiv.sh` script.
For example, if `stu_bedprefix_comm` has 1,000,000 samples,
then to extract samples 12,000 - 13,300, you can run
```
bash splitindiv.sh stu_bedprefix_comm 1000 12
```
To completely breakdown the study data, run
```
for i in `seq 1 1000`; do
  bash splitindiv.sh stu_bedprefix_comm 1000 $i
done;
```

# FRAPOSA
To use FRAPOSA with default settings, run (in Python 3)
```
python fraposa_runner.py ref_bedprefix_comm stu_bedprefix_comm
```

To learn about the options for FRAPOSA, run (in Python 3)
```
python fraposa_runner.py --help
```
