First, edit `sequences.csv` and `parameters.csv` files, and then run by
```
$ python ../optimise.py Lattice ../../notebooks/scapTet.pdb  ./sequences.csv ./parameters.csv 2000 20 8
```
You can enter more than one group of sequences. When more than one group of sequences are given, an optimisation will be executed for each group and the resulting model/parameters are saved to disk with the `id` of each group as a suffix.