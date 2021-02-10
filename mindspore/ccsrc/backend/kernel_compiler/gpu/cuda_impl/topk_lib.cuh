/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

constexpr int kWarpSize = 32;

constexpr __host__ __device__ int Log2(int n, int p = 0) { return (n <= 1) ? p : Log2(n / 2, p + 1); }
constexpr __host__ __device__ bool IsPow2(int v) { return (v && !(v & (v - 1))); }
constexpr __host__ __device__ int NextPow2(int v) { return (IsPow2(v) ? 2 * v : (1 << static_cast<int>(Log2(v) + 1))); }

__device__ __forceinline__ int GetLaneId() {
  int laneId;
  asm("mov.u32 %0, %%laneid;" : "=r"(laneId));
  return laneId;
}

template <typename T, typename S>
struct CmpKV {
  __device__ static inline bool gt(T k1, S v1, T k2, S v2) { return k1 > k2 || (k1 == k2 && v1 < v2); }
  __device__ static inline bool lt(T k1, S v1, T k2, S v2) { return k1 < k2 || (k1 == k2 && v1 > v2); }
};

template <typename T>
struct Cmp {
  __device__ static inline bool lt(T a, T b) { return a < b; }
  __device__ static inline bool gt(T a, T b) { return a > b; }
};

template <typename T>
inline __device__ T shfl_xor(const T val, int laneMask, int width = kWarpSize) {
  return __shfl_xor_sync(0xffffffff, val, laneMask, width);
}

template <typename T, typename S, bool is_descend>
inline __device__ void L2CompareAndSwap(T *a, S *b, int i_1, int i_2) {
  bool swap =
      is_descend ? CmpKV<T, S>::gt(a[i_1], b[i_1], a[i_2], b[i_2]) : CmpKV<T, S>::lt(a[i_1], b[i_1], a[i_2], b[i_2]);

  if (!swap) return;

  T a_tmp = a[i_1];
  a[i_1] = a[i_2];
  a[i_2] = a_tmp;

  T b_tmp = b[i_1];
  b[i_1] = b[i_2];
  b[i_2] = b_tmp;
}

template <typename T>
inline __device__ void ConditionalAssign(bool is_assign, T *x, const T &y) {
  (*x) = is_assign ? y : (*x);
}

// Merge pairs of lists smaller than threads-per-block
// NumThreads is 128
// N is 2, 1 etc
// L is 32, 64 etc
template <int NumThreads, typename T, typename S, int N, int L, bool AllThreads, bool is_descend, bool FullMerge>
inline __device__ void BlockSortSmallK(T *list_k, S *list_v) {
  int mergeId = threadIdx.x / L;
  int tid = threadIdx.x % L;

  list_k += 2 * L * mergeId;
  list_v += 2 * L * mergeId;

  int pos = L - 1 - tid;
  int stride = 2 * tid + 1;

  if (AllThreads || (static_cast<int>(threadIdx.x) < N * L)) {
    L2CompareAndSwap<T, S, is_descend>(list_k, list_v, pos, pos + stride);
  }

  __syncthreads();

  _Pragma("unroll") for (int stride = L / 2; stride > 0; stride /= 2) {
    int pos = 2 * tid - (tid & (stride - 1));

    if (AllThreads || (static_cast<int>(threadIdx.x) < N * L)) {
      L2CompareAndSwap<T, S, is_descend>(list_k, list_v, pos, pos + stride);
    }

    __syncthreads();
  }
}

// Merge pairs of lists larger than threads-per-block
template <int NumThreads, typename T, typename S, int L, bool is_descend, bool FullMerge>
inline __device__ void BlockSortBigK(T *list_k, S *list_v) {
  constexpr int kLoopPerThread = L / NumThreads;

  _Pragma("unroll") for (int loop = 0; loop < kLoopPerThread; ++loop) {
    int tid = loop * NumThreads + threadIdx.x;
    int pos = L - 1 - tid;
    int stride = 2 * tid + 1;

    L2CompareAndSwap<T, S, is_descend>(list_k, list_v, pos, pos + stride);
  }

  __syncthreads();

  constexpr int kSecondLoopPerThread = FullMerge ? kLoopPerThread : kLoopPerThread / 2;

  _Pragma("unroll") for (int stride = L / 2; stride > 0; stride /= 2) {
    _Pragma("unroll") for (int loop = 0; loop < kSecondLoopPerThread; ++loop) {
      int tid = loop * NumThreads + threadIdx.x;
      int pos = 2 * tid - (tid & (stride - 1));
      L2CompareAndSwap<T, S, is_descend>(list_k, list_v, pos, pos + stride);
    }
    __syncthreads();
  }
}

/// Merging lists smaller than threads-per-block
template <int NumThreads, typename T, typename S, int N, int L, bool is_descend, bool FullMerge = true>
inline __device__ void SortBlockStep(T *list_k, S *list_v) {
  if (L <= NumThreads) {
    int kNumParallelMerges = NumThreads / L;
    int kNumIterations = N / kNumParallelMerges;

    if (N < kNumParallelMerges) {
      BlockSortSmallK<NumThreads, T, S, N, L, false, is_descend, FullMerge>(list_k, list_v);
    } else {
      _Pragma("unroll") for (int i = 0; i < kNumIterations; ++i) {
        int start = i * kNumParallelMerges * 2 * L;
        BlockSortSmallK<NumThreads, T, S, N, L, true, is_descend, FullMerge>(list_k + start, list_v + start);
      }
    }
  } else {
    _Pragma("unroll") for (int i = 0; i < N; ++i) {
      int start = i * 2 * L;
      BlockSortBigK<NumThreads, T, S, L, is_descend, FullMerge>(list_k + start, list_v + start);
    }
  }
}

// Block-wide merge
template <int NumWarps, int NumThreads, typename T, typename S, int warp_queue, bool is_descend>
inline __device__ void SortBlockWide(T *shared_K, S *shared_V) {
  if (NumWarps == 2) {
    SortBlockStep<NumThreads, T, S, NumThreads / (kWarpSize * 2), warp_queue, !is_descend, false>(shared_K, shared_V);
  } else if (NumWarps == 4) {
    SortBlockStep<NumThreads, T, S, NumThreads / (kWarpSize * 2), warp_queue, !is_descend>(shared_K, shared_V);
    SortBlockStep<NumThreads, T, S, NumThreads / (kWarpSize * 4), warp_queue * 2, !is_descend, false>(shared_K,
                                                                                                      shared_V);
  } else if (NumWarps == 8) {
    SortBlockStep<NumThreads, T, S, NumThreads / (kWarpSize * 2), warp_queue, !is_descend>(shared_K, shared_V);
    SortBlockStep<NumThreads, T, S, NumThreads / (kWarpSize * 4), warp_queue * 2, !is_descend>(shared_K, shared_V);
    SortBlockStep<NumThreads, T, S, NumThreads / (kWarpSize * 8), warp_queue * 4, !is_descend, false>(shared_K,
                                                                                                      shared_V);
  } else if (NumWarps == 16) {
    SortBlockStep<NumThreads, T, S, NumThreads / (kWarpSize * 2), warp_queue, !is_descend>(shared_K, shared_V);
    SortBlockStep<NumThreads, T, S, NumThreads / (kWarpSize * 4), warp_queue * 2, !is_descend>(shared_K, shared_V);
    SortBlockStep<NumThreads, T, S, NumThreads / (kWarpSize * 8), warp_queue * 4, !is_descend>(shared_K, shared_V);
    SortBlockStep<NumThreads, T, S, NumThreads / (kWarpSize * 16), warp_queue * 8, !is_descend>(shared_K, shared_V);
  } else if (NumWarps == 32) {
    SortBlockStep<NumThreads, T, S, NumThreads / (kWarpSize * 2), warp_queue, !is_descend>(shared_K, shared_V);
    SortBlockStep<NumThreads, T, S, NumThreads / (kWarpSize * 4), warp_queue * 2, !is_descend>(shared_K, shared_V);
    SortBlockStep<NumThreads, T, S, NumThreads / (kWarpSize * 8), warp_queue * 4, !is_descend>(shared_K, shared_V);
    SortBlockStep<NumThreads, T, S, NumThreads / (kWarpSize * 16), warp_queue * 8, !is_descend>(shared_K, shared_V);
    SortBlockStep<NumThreads, T, S, NumThreads / (kWarpSize * 32), warp_queue * 16, !is_descend>(shared_K, shared_V);
  }
}

template <typename T, typename S, int L, bool is_descend, bool IsBitonic>
inline __device__ void BitonicSortWarpLE16(T *k, S *v) {
  int laneId = GetLaneId();

  if (!IsBitonic) {
    // Reverse the first comparison stage. head-tail swap.
    T other_K = shfl_xor((*k), 2 * L - 1);
    S other_V = shfl_xor((*v), 2 * L - 1);

    bool small = !(laneId & L);
    bool small_compare = small ? CmpKV<T, S>::gt((*k), (*v), other_K, other_V) :
                                 CmpKV<T, S>::lt((*k), (*v), other_K, other_V);
    bool small_compare_descend = is_descend ? small_compare : !small_compare;
    ConditionalAssign(small_compare_descend, k, other_K);
    ConditionalAssign(small_compare_descend, v, other_V);
  }

  _Pragma("unroll") for (int stride = IsBitonic ? L : L / 2; stride > 0; stride /= 2) {
    T other_K = shfl_xor((*k), stride);
    S other_V = shfl_xor((*v), stride);

    bool small = !(laneId & stride);
    bool small_compare = small ? CmpKV<T, S>::gt((*k), (*v), other_K, other_V) :
                                 CmpKV<T, S>::lt((*k), (*v), other_K, other_V);
    bool small_compare_descend = is_descend ? small_compare : !small_compare;
    ConditionalAssign(small_compare_descend, k, other_K);
    ConditionalAssign(small_compare_descend, v, other_V);
  }
}

template <typename T, typename S, int N, bool is_descend, bool Low, bool Pow2>
struct MergeWarpStepBitonic {};

// All merges call this
template <typename T, typename S, bool is_descend, bool Low>
struct MergeWarpStepBitonic<T, S, 1, is_descend, Low, true> {
  static inline __device__ void merge(T k[1], S v[1]) { BitonicSortWarpLE16<T, S, 16, is_descend, true>(&k[0], &v[0]); }
};

template <typename T, typename S, int N, bool is_descend, bool Low>
struct MergeWarpStepBitonic<T, S, N, is_descend, Low, true> {
  static inline __device__ void merge(T k[N], S v[N]) {
    _Pragma("unroll") for (int i = 0; i < N / 2; ++i) { L2CompareAndSwap<T, S, is_descend>(k, v, i, i + N / 2); }

    {
      T newK[N / 2];
      S newV[N / 2];

      _Pragma("unroll") for (int i = 0; i < N / 2; ++i) {
        newK[i] = k[i];
        newV[i] = v[i];
      }

      MergeWarpStepBitonic<T, S, N / 2, is_descend, true, true>::merge(newK, newV);

      _Pragma("unroll") for (int i = 0; i < N / 2; ++i) {
        k[i] = newK[i];
        v[i] = newV[i];
      }
    }

    {
      T newK[N / 2];
      S newV[N / 2];

      _Pragma("unroll") for (int i = 0; i < N / 2; ++i) {
        newK[i] = k[i + N / 2];
        newV[i] = v[i + N / 2];
      }

      MergeWarpStepBitonic<T, S, N / 2, is_descend, false, true>::merge(newK, newV);

      _Pragma("unroll") for (int i = 0; i < N / 2; ++i) {
        k[i + N / 2] = newK[i];
        v[i + N / 2] = newV[i];
      }
    }
  }
};

// Low recursion
template <typename T, typename S, int N, bool is_descend>
struct MergeWarpStepBitonic<T, S, N, is_descend, true, false> {
  static inline __device__ void merge(T k[N], S v[N]) {
    constexpr int kNextHighestPowerOf2 = NextPow2(N);

    _Pragma("unroll") for (int i = 0; i < N - kNextHighestPowerOf2 / 2; ++i) {
      L2CompareAndSwap<T, S, is_descend>(k, v, i, i + kNextHighestPowerOf2 / 2);
    }

    constexpr int kLowSize = N - kNextHighestPowerOf2 / 2;
    constexpr int kHighSize = kNextHighestPowerOf2 / 2;
    {
      T newK[kLowSize];
      S newV[kLowSize];

      _Pragma("unroll") for (int i = 0; i < kLowSize; ++i) {
        newK[i] = k[i];
        newV[i] = v[i];
      }

      constexpr bool kLowIsPowerOf2 = IsPow2(N - kNextHighestPowerOf2 / 2);
      MergeWarpStepBitonic<T, S, kLowSize, is_descend, true, kLowIsPowerOf2>::merge(newK, newV);

      _Pragma("unroll") for (int i = 0; i < kLowSize; ++i) {
        k[i] = newK[i];
        v[i] = newV[i];
      }
    }

    {
      T newK[kHighSize];
      S newV[kHighSize];

      _Pragma("unroll") for (int i = 0; i < kHighSize; ++i) {
        newK[i] = k[i + kLowSize];
        newV[i] = v[i + kLowSize];
      }

      constexpr bool kHighIsPowerOf2 = IsPow2(kNextHighestPowerOf2 / 2);
      MergeWarpStepBitonic<T, S, kHighSize, is_descend, false, kHighIsPowerOf2>::merge(newK, newV);

      _Pragma("unroll") for (int i = 0; i < kHighSize; ++i) {
        k[i + kLowSize] = newK[i];
        v[i + kLowSize] = newV[i];
      }
    }
  }
};

// High recursion
template <typename T, typename S, int N, bool is_descend>
struct MergeWarpStepBitonic<T, S, N, is_descend, false, false> {
  static inline __device__ void merge(T k[N], S v[N]) {
    constexpr int kNextHighestPowerOf2 = NextPow2(N);

    _Pragma("unroll") for (int i = 0; i < N - kNextHighestPowerOf2 / 2; ++i) {
      L2CompareAndSwap<T, S, is_descend>(k, v, i, i + kNextHighestPowerOf2 / 2);
    }

    constexpr int kLowSize = kNextHighestPowerOf2 / 2;
    constexpr int kHighSize = N - kNextHighestPowerOf2 / 2;
    {
      T newK[kLowSize];
      S newV[kLowSize];

      _Pragma("unroll") for (int i = 0; i < kLowSize; ++i) {
        newK[i] = k[i];
        newV[i] = v[i];
      }

      constexpr bool kLowIsPowerOf2 = IsPow2(kNextHighestPowerOf2 / 2);
      MergeWarpStepBitonic<T, S, kLowSize, is_descend, true, kLowIsPowerOf2>::merge(newK, newV);

      _Pragma("unroll") for (int i = 0; i < kLowSize; ++i) {
        k[i] = newK[i];
        v[i] = newV[i];
      }
    }

    {
      T newK[kHighSize];
      S newV[kHighSize];

      _Pragma("unroll") for (int i = 0; i < kHighSize; ++i) {
        newK[i] = k[i + kLowSize];
        newV[i] = v[i + kLowSize];
      }

      constexpr bool kHighIsPowerOf2 = IsPow2(N - kNextHighestPowerOf2 / 2);
      MergeWarpStepBitonic<T, S, kHighSize, is_descend, false, kHighIsPowerOf2>::merge(newK, newV);

      _Pragma("unroll") for (int i = 0; i < kHighSize; ++i) {
        k[i + kLowSize] = newK[i];
        v[i + kLowSize] = newV[i];
      }
    }
  }
};

/// Merges two sets of registers across the warp of any size;
template <typename T, typename S, int N1, int N2, bool is_descend, bool FullMerge = true>
inline __device__ void MergeWarpByRegister(T k1[N1], S v1[N1], T k2[N2], S v2[N2]) {
  constexpr int kSmallestN = N1 < N2 ? N1 : N2;

  _Pragma("unroll") for (int i = 0; i < kSmallestN; ++i) {
    T &ka = k1[N1 - 1 - i];
    S &va = v1[N1 - 1 - i];

    T &kb = k2[i];
    S &vb = v2[i];

    T other_Ka;
    S other_Va;

    if (FullMerge) {
      other_Ka = shfl_xor(ka, kWarpSize - 1);
      other_Va = shfl_xor(va, kWarpSize - 1);
    }

    T other_Kb = shfl_xor(kb, kWarpSize - 1);
    S other_Vb = shfl_xor(vb, kWarpSize - 1);

    bool swapa = is_descend ? CmpKV<T, S>::gt(ka, va, other_Kb, other_Vb) : CmpKV<T, S>::lt(ka, va, other_Kb, other_Vb);
    ConditionalAssign(swapa, &ka, other_Kb);
    ConditionalAssign(swapa, &va, other_Vb);

    if (FullMerge) {
      bool swapb = is_descend ? CmpKV<T, S>::lt(kb, vb, other_Ka, other_Va) :
                                CmpKV<T, S>::gt(kb, vb, other_Ka, other_Va);
      ConditionalAssign(swapb, &kb, other_Ka);
      ConditionalAssign(swapb, &vb, other_Va);
    }
  }

  MergeWarpStepBitonic<T, S, N1, is_descend, true, IsPow2(N1)>::merge(k1, v1);
  if (FullMerge) {
    MergeWarpStepBitonic<T, S, N2, is_descend, false, IsPow2(N2)>::merge(k2, v2);
  }
}

// Recursive template that uses the above bitonic merge
template <typename T, typename S, int N, bool is_descend>
struct SortWarpStepBitonic {
  static inline __device__ void Sort(T k[N], S v[N]) {
    constexpr int kSizeA = N / 2;
    constexpr int kSizeB = N - kSizeA;

    T aK[kSizeA];
    S aV[kSizeA];

    _Pragma("unroll") for (int i = 0; i < kSizeA; ++i) {
      aK[i] = k[i];
      aV[i] = v[i];
    }

    // Recursive sort
    SortWarpStepBitonic<T, S, kSizeA, is_descend>::Sort(aK, aV);

    T bK[kSizeB];
    S bV[kSizeB];

    _Pragma("unroll") for (int i = 0; i < kSizeB; ++i) {
      bK[i] = k[i + kSizeA];
      bV[i] = v[i + kSizeA];
    }

    SortWarpStepBitonic<T, S, kSizeB, is_descend>::Sort(bK, bV);

    // Merge halves
    MergeWarpByRegister<T, S, kSizeA, kSizeB, is_descend>(aK, aV, bK, bV);

    _Pragma("unroll") for (int i = 0; i < kSizeA; ++i) {
      k[i] = aK[i];
      v[i] = aV[i];
    }

    _Pragma("unroll") for (int i = 0; i < kSizeB; ++i) {
      k[i + kSizeA] = bK[i];
      v[i + kSizeA] = bV[i];
    }
  }
};

template <typename T, typename S, bool is_descend>
struct SortWarpStepBitonic<T, S, 1, is_descend> {
  static inline __device__ void Sort(T k[1], S v[1]) {
    // up to warp-size/2
    BitonicSortWarpLE16<T, S, 1, is_descend, false>(&k[0], &v[0]);
    BitonicSortWarpLE16<T, S, 2, is_descend, false>(&k[0], &v[0]);
    BitonicSortWarpLE16<T, S, 4, is_descend, false>(&k[0], &v[0]);
    BitonicSortWarpLE16<T, S, 8, is_descend, false>(&k[0], &v[0]);
    BitonicSortWarpLE16<T, S, 16, is_descend, false>(&k[0], &v[0]);
  }
};

template <typename T, typename S, int N, bool is_descend>
inline __device__ void SortWarpByRegister(T k[N], S v[N]) {
  SortWarpStepBitonic<T, S, N, is_descend>::Sort(k, v);
}

template <typename T, typename S, int warp_queue, int thread_queue, bool is_descend>
inline __device__ void MergeWarpQueue(T *threadK, S *threadV, T *warp_K, S *warp_V) {
  int laneId = GetLaneId();
  SortWarpByRegister<T, S, thread_queue, !is_descend>(threadK, threadV);

  constexpr int kWarpQueueRegisters = warp_queue / kWarpSize;
  T warp_KRegisters[kWarpQueueRegisters];
  S warp_VRegisters[kWarpQueueRegisters];
  _Pragma("unroll") for (int i = 0; i < kWarpQueueRegisters; ++i) {
    warp_KRegisters[i] = warp_K[i * kWarpSize + laneId];
    warp_VRegisters[i] = warp_V[i * kWarpSize + laneId];
  }
  __syncwarp();
  MergeWarpByRegister<T, S, kWarpQueueRegisters, thread_queue, !is_descend, false>(warp_KRegisters, warp_VRegisters,
                                                                                   threadK, threadV);
  _Pragma("unroll") for (int i = 0; i < kWarpQueueRegisters; ++i) {
    warp_K[i * kWarpSize + laneId] = warp_KRegisters[i];
    warp_V[i * kWarpSize + laneId] = warp_VRegisters[i];
  }
  __syncwarp();
}
