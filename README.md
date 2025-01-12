# Timeloop (with quantization)

## About

Timeloop is an infrastructure that aims to provide modeling, mapping and code-generation for dense- and sparse- tensor algebra workloads on a range of accelerator architectures. It is built from two modular components:

* A fast analytical model that can emulate a range of architecture designs and provide performance and energy projections
* A mapper that that searches for an optimal mapping in the space of mappings of a tensor-algebra problem on a given architecture

## Fork additions

Additions to account for specifying different workload tensors bitwidths during the mapping evaluation using bit-packing technique to reduce memory footprint.

To account for the use of different bitwidths, you need to specify them inside the workload's YAML specification.

Simply specify the following under the instance key according to your desire (NOTE: the bitwidth must be able to fit in HW's word size):
```yaml
bitwidths:
      Inputs: X
      Outputs: X
      Weights: X
```

For example:
```yaml
problem:
  instance:
    C: 3
    Hdilation: 1
    Hstride: 2
    M: 32
    N: 1
    P: 112
    Q: 112
    R: 3
    S: 3
    Wdilation: 1
    Wstride: 2
    bitwidths:
      Inputs: 8
      Outputs: 8
      Weights: 2
  shape:
    ...
```


## Documentation

Timeloop documentation is hosted at https://timeloop.csail.mit.edu/timeloop. The guides there cover installation, usage and examples.
For a deeper understanding of Timeloop's internals please read the original [ISPASS 2019 paper](https://parashar.org/ispass19.pdf).

Timeloop version 2.0 (a.k.a. Sparseloop) provides stochastic modeling of compressed-sparse tensor algebra. This work is described in our [MICRO 2022 paper](https://www.computer.org/csdl/proceedings-article/micro/2022/627200b377/1HMSE23T13a).

Timeloop version 3.0 (a.k.a. Ruby) adds support for imperfectly-factorized mappings (described in our [ISPASS 2022 paper](https://ieeexplore.ieee.org/document/9804679)), in addition to support for spatial skews and flattened mappings.

## Tutorial

New users are strongly encouraged to complete the original Timeloop [tutorial](https://accelergy.mit.edu/tutorial.html). Serially walking through the [exercises](https://github.com/Accelergy-Project/timeloop-accelergy-exercises/) from the tutorial serves as an essential hands-on introduction to the tool.

## Dependencies

Timeloop depends on the isl and barvinok libraries. In particular, barvinok version 0.41.6 (along with the pre-packaged isl library) has been tested to
build successfully with this version of Timeloop. Instructions for installing barvinok can be found in the [this link](https://barvinok.sourceforge.io/).
