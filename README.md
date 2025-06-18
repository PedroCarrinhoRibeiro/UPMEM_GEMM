# UPMEM GEMM



UPMEM and C parallel implementation of the General Matrix multiplication (GEMM) for UPMEM DIMMs. The multiplication is conducted using Multi DPUs and Mutithreading.
The implementation follows a host-device fashion and the MRAM banks of the UPMEM DIMMs are used.
If you plan on using this code or parts of this code please cite: https://ieeexplore.ieee.org/abstract/document/10768222
I highly advise that you read it before using the code.
Author: Pedro Jose Carrinho Ribeiro

```
@inproceedings{carrinho2024processing,
  title={Processing Multi-Layer Perceptrons In-Memory},
  author={Carrinho, Pedro and Ferraz, Oscar and Ferreira, Jo{\~a}o Dinis and Falevoz, Yann and Silva, Vitor and Falcao, Gabriel},
  booktitle={2024 IEEE Workshop on Signal Processing Systems (SiPS)},
  pages={7--12},
  year={2024},
  organization={IEEE}
}
```

## How to run

Just compile using make and then run. You can customize matrix size on common.h and matrix values on host.c.

## Requirements

Either run on a UPMEM PIM-enabled machine or use UPMEM SDK: https://sdk.upmem.com/

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
│
├── Makefile           <- Makefile to create the executables
│
├── README.md          <- The top-level README for developers using this project.
│
├── host.c             <- Host code for creating the matrices, sending data and fetching the result to and from the DPUs
│
├── kdot_kernls.c      <- UPMEM kernels that run on the DPU
│
├── KDOT.c             <- Code that is executed on the DPU
│
└── common                      <- Folder with the headers
    │
    ├── common.h                <- Defining parameters such as, the number of DPUs to allocate and the matrix dimensions
    │
    └── timer.h                 <- Code for time counting of the operation
```

--------

## Acknowledgements

Please cite my article: https://ieeexplore.ieee.org/abstract/document/10768222

```
@inproceedings{carrinho2024processing,
  title={Processing Multi-Layer Perceptrons In-Memory},
  author={Carrinho, Pedro and Ferraz, Oscar and Ferreira, Jo{\~a}o Dinis and Falevoz, Yann and Silva, Vitor and Falcao, Gabriel},
  booktitle={2024 IEEE Workshop on Signal Processing Systems (SiPS)},
  pages={7--12},
  year={2024},
  organization={IEEE}
}
```