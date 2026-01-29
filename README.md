# RNS-ROCm - Residue Number System Library For Matrix Multiplications on ROCm/HIP based GPU

A high-performance ROCm/HIP C++ library for exact modular arithmetic using the **Residue Number System (RNS)**, focused on batched small matrix multiplications and CRT reconstruction.

## What is RNS?

The Residue Number System represents large integers as a tuple of residues modulo a set of coprime moduli. This enables **exact** arithmetic on very large integers using only 32-bit operations per prime, with results reconstructed via the Chinese Remainder Theorem (CRT). RNS is ideal for GPU computation because each prime lane is independent—enabling massive parallelism.

## Features

- **Batched modular GEMM** for small matrices (m = 4, 6, 8, 10) with specialized unrolled kernels
- **Barrett reduction** for fast modular multiplication without division
- **CRT reconstruction** using Garner's algorithm for reconstructing integers from residues
- **CPU reference implementations** for correctness testing
- **Flexible prime generation** with verified primality
- **SoA data layout** optimized for GPU memory access: `[K][BATCH*m*m]`

## Data Layout

Arrays are stored in Structure-of-Arrays (SoA) format:
```
data[k * (B * E) + b * E + (i * m + j)]
```
Where:
- `K` = number of primes
- `B` = batch size (number of matrices per prime)
- `E` = m × m (elements per matrix)
- `k` = prime index
- `b` = batch index
- `i, j` = matrix row/column

## Building

### Prerequisites

- CMake 3.21+
- C++17 compiler
- ROCm 5.0+ (for GPU support)
- (Optional) Boost for extended bigint support

### CPU-only build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DRNS_ENABLE_GPU=OFF
cmake --build . -j
```

### GPU build (ROCm)

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

### Running tests

```bash
cd build
ctest --output-on-failure
```

Or run individual tests:
```bash
./test_rns_ops
./test_crt
./test_crt_matrix
./test_gemm_mod  # requires GPU
```

### Running demos

```bash
./demo_gemm          # GPU GEMM benchmark
./demo_crt_matrix    # CRT encoding/decoding demo
```

## API Overview

### Device Context

```cpp
#include "rns/rns_api.h"

auto primes = rns::get_standard_primes_31bit(K);
auto ctx = rns::create_context(primes);

// ... use ctx ...

rns::destroy_context(ctx);
```

### Batched GEMM

```cpp
// Allocate device buffers
uint32_t* dA = rns::device_alloc_u32(K * B * m * m);
uint32_t* dB = rns::device_alloc_u32(K * B * m * m);
uint32_t* dC = rns::device_alloc_u32(K * B * m * m);

// Upload data
rns::h2d_u32(dA, hostA, K * B * m * m);
rns::h2d_u32(dB, hostB, K * B * m * m);

// Matrix multiply: C = A @ B (mod p) for all primes and batches
rns::gemm_mod(ctx, dC, dA, dB, B, m);
rns::device_sync();

// Download results
rns::d2h_u32(hostC, dC, K * B * m * m);
```

### CRT Reconstruction

```cpp
#include "rns/crt.h"

auto primes = rns::get_standard_primes_31bit(16);
auto plan = rns::CrtPlan::create(primes);

// Encode a BigInt to residues
rns::BigInt x("123456789012345678901234567890");
std::vector<uint32_t> residues(plan.K);
rns::rns_encode(x, plan, residues.data());

// Decode back
rns::BigInt y = rns::crt_reconstruct(residues.data(), plan);
assert(y == x % plan.M_total);
```

## Integer Range / Capacity

With K primes of ~31 bits each, the total capacity is approximately:
- **K=8**: ~248 bits
- **K=16**: ~496 bits  
- **K=32**: ~992 bits
- **K=64**: ~1984 bits

Ensure your intermediate results stay within capacity to avoid wraparound.

## Correctness Guarantees

- All operations are **exact** (no floating-point approximation)
- Results are correct modulo M = ∏ p_k
- CRT reconstruction is unique for values in [0, M)
- All primes are verified for primality and distinctness

## Architecture

```
rocm-rns-matmul/
├── include/rns/
│   ├── rns_types.h     # Core types (Modulus32, DeviceContext)
│   ├── rns_ops.h       # Device modular arithmetic
│   ├── rns_kernels.h   # GPU kernel declarations
│   ├── rns_api.h       # Public API
│   ├── crt.h           # CRT reconstruction
│   ├── crt_types.h     # BigInt, CrtPlan
│   ├── primes.h        # Prime generation
│   └── utils.h         # Utilities
├── src/
│   ├── hip/            # HIP kernels
│   ├── rns_api.cpp
│   ├── crt.cpp
│   ├── primes.cpp
│   └── utils.cpp
├── tests/              # Unit tests
└── examples/           # Demo programs
```

## Future Roadmap

1. **Kernel fusion**: Fuse P ← P @ M_t for walk loops
2. **Expression evaluator**: GPU bytecode evaluation for matrix construction
3. **TopK selection**: GPU-side candidate ranking
4. **GPU CRT**: Partial CRT for fast float approximation
5. **More matrix sizes**: Add m=5, 7, 12 specializations
6. **Multi-stream pipeline**: Overlap compute and transfers
7. **Python bindings**: PyBind11 wrapper

## License

MIT License - see LICENSE file.
