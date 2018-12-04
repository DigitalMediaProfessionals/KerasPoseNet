Original source: https://github.com/ildoonet/tf-pose-estimation/tree/master/tf_pose/pafprocess
git SHA1 ID: b119759e8a41828c633bd39b5c883bf5a56a214f
Apache 2.0 License

# post-processing for Part-Affinity Fields Map implemented in C++ & Swig

Need to install swig.

```bash
$ sudo apt install swig
```

You need to build pafprocess module which is written in c++. It will be used for post processing.

```bash
$ swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace
```
