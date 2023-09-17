# Timeloop (with quantization)

## About

Timeloop is an infrastructure that aims to provide modeling, mapping and code-generation for Explicitly-Decoupled Data Orchestration (EDDO) architectures, with a focus on for dense- and sparse- tensor algebra workloads. It is built from two modular components:

* A fast analytical model that can emulate a range of EDDO architecture designs and provide performance and energy projections
* A mapper that that searches for an optimal mapping in the space of mappings of a tensor-algebra problem on a given architecture

## Fork additions

Additions to account for specifying different workload tensors bitwidths during the mapping evaluation using bit-packing technique to reduce memory footprint.

To account for the use of different bitwidths, you need to specify it inside the workload's YAML specification.

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

## Tutorial

New users are strongly encouraged to complete the original Timeloop [tutorial](https://accelergy.mit.edu/tutorial.html). Serially walking through the [exercises](https://github.com/Accelergy-Project/timeloop-accelergy-exercises/) from the tutorial serves as an essential hands-on introduction to the tool.
