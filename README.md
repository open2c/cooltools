# cooltools

[![Build Status](https://travis-ci.org/mirnylab/cooltools.svg?branch=master)](https://travis-ci.org/mirnylab/cooltools)
[![Documentation Status](https://readthedocs.org/projects/cooltools/badge/?version=latest)](https://cooltools.readthedocs.io/en/latest/?badge=latest)

The tools for your .cool's

## Calling compartments

To call compartments, use `cooltools` to calculate the first three eigenvectors
across the matrix. 

```
    cooltools call-compartments ${fullfile}::/resolutions/100000 --out-prefix $filename;
```

Sample output:

```
chrom   start   end     E1      E2      E3
1       0       100000
...
1       3000000 3100000 3.009943538549052       2.762190276669322       0.13578728460826148
...
```
