/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPL_OP_CUSTOM_GEMM
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPL_OP_CUSTOM_GEMM

#include "extendrt/delegate/ascend_native/ascend_native_impl/utils.h"
#include "tikcfw/kernel_operator.h"
#ifdef __CCE_AICORE__
#include "lib/matmul_intf.h"
using AscendC::AIC;
using AscendC::AIV;
using AscendC::BLOCK_CUBE;
using AscendC::Ceil;
using AscendC::GetBlockIdx;
using AscendC::GetBlockNum;
using AscendC::GetSubBlockIdx;
using AscendC::LocalTensor;
using AscendC::ONE_BLK_SIZE;
using AscendC::SetSysWorkspace;
using AscendC::TOTAL_L0C_SIZE;
using AscendC::TOTAL_L1_SIZE;
using AscendC::TPipe;
using AscendC::TPosition;
using mindspore::ascend_native::MMExtra;
template <typename T, int pipeSize>
class KernelFuseNone {
 public:
  __aicore__ inline KernelFuseNone() {}
  __aicore__ inline void Init(TPipe *pipe, int elem_num) { elem_num_ = elem_num; }
  __aicore__ inline void Process(TPipe *pipe, const AscendC::LocalTensor<T> &inTensor) {}

 private:
  int elem_num_;
};

template <typename T, int pipeSize>
class KernelFuseGelu {
 public:
  __aicore__ inline KernelFuseGelu() {}
  __aicore__ inline void Init(TPipe *pipe, int elem_num) {
    pipe->InitBuffer(q_, pipeSize, ALIGN32(elem_num * sizeof(T)));
    elem_num_ = elem_num;
  }
  __aicore__ inline void Process(TPipe *pipe, const AscendC::LocalTensor<T> &local_tensor) {
    auto tmp = q_.template AllocTensor<T>();
    DataCopy(tmp, local_tensor, ALIGN32(elem_num_ * sizeof(T)));
    pipe_barrier(PIPE_ALL);  // tmp holds matrix output
    FasterGelu(local_tensor, tmp, elem_num_);
    q_.FreeTensor(tmp);
  }

 private:
  AscendC::TQue<TPosition::VECOUT, pipeSize> q_;
  int elem_num_;
};

template <class aT, class bT, class cT, class biasT, class fuseOp, int pipeSize>
class KernelFuseMatMul {
 public:
  __aicore__ inline KernelFuseMatMul() {}
  __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR tilingGM,
                              GM_ADDR workspaceGM, GM_ADDR extra) {
    set_atomic_none();
    // copy tiling to local memory
    auto tempTilingGM = (__gm__ TCubeTiling *)tilingGM;
    CopyTiling(&tiling_, tempTilingGM, &que_);

    ldc_ = tiling_.N;
    bmm_num_ = 1;
    m_ = tiling_.M;
    n_ = tiling_.N;
    k_a_ = tiling_.Ka;
    k_b_ = tiling_.Kb;
    bmm_num_ = 1;
    if (extra) {
      auto tempMmExtraGM = (__gm__ MMExtra *)extra;
      CopyMmExtra(tempMmExtraGM);
      // setup LDA and LDB
      ldc_ = (mm_extra_.ldc_ != -1) ? mm_extra_.ldc_ : tiling_.N;
      if constexpr (aT::isTrans == true) {
        tiling_.M = (mm_extra_.lda_ != -1) ? mm_extra_.lda_ : tiling_.M;
      } else {
        tiling_.Ka = (mm_extra_.lda_ != -1) ? mm_extra_.lda_ : tiling_.Ka;
        k_a_ = tiling_.Ka;
      }
      if constexpr (bT::isTrans == true) {
        tiling_.Kb = (mm_extra_.ldb_ != -1) ? mm_extra_.ldb_ : tiling_.Kb;
        k_b_ = tiling_.Kb;
      } else {
        tiling_.N = (mm_extra_.ldb_ != -1) ? mm_extra_.ldb_ : tiling_.N;
      }
      bmm_num_ = mm_extra_.bmm_num_ > 1 ? mm_extra_.bmm_num_ : 1;
    }
    // setup global tensors
    uint32_t offsetA = 0, offsetB = 0, offsetC = 0, offsetBias = 0;
    uint32_t blockIdx = GetBlockIdx();
#ifdef CLNTSRV
    blockIdx /= 2;
#endif
    blck_idx_ = blockIdx;
    block_num_ = GetBlockNum();
    CalcGMOffset(blck_idx_, tiling_.usedCoreNum, &tiling_, &offsetA, &offsetB, &offsetC, &offsetBias);
    a_global_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(aGM), tiling_.M * tiling_.Ka * bmm_num_);
    b_global_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(bGM), tiling_.Kb * tiling_.N * bmm_num_);
    c_global_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(cGM), m_ * tiling_.N * bmm_num_);
    bias_global_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(biasGM), tiling_.N * bmm_num_);
    a_global_ = a_global_[offsetA];
    b_global_ = b_global_[offsetB];
    c_global_ = c_global_[offsetC];
    if (tiling_.isBias) {
      bias_global_ = bias_global_[offsetBias];
    }
    que_.InitBuffer(out_q_, pipeSize, tiling_.baseM * tiling_.baseN * sizeof(T));
    fuse_op_.Init(&que_, tiling_.baseM * tiling_.baseN);
    InitDefaultTiling();
    SetSysWorkspace(workspaceGM);
  }

  __aicore__ inline void InitDefaultTiling() {
    tiling_.shareMode = 0;
    tiling_.shareL1Size = TOTAL_L1_SIZE;
    tiling_.shareL0CSize = TOTAL_L0C_SIZE;
    tiling_.shareUbSize = 0;
  }

  __aicore__ inline void Process() {
    uint64_t a_stride = tiling_.M * k_a_;
    uint64_t b_stride = k_b_ * tiling_.N;
    uint64_t c_stride = m_ * ldc_;
    uint64_t bias_stride = tiling_.N;
    uint64_t start_idx = 0;
    if (bmm_num_ > 1) {
      start_idx = blck_idx_;
    }

#ifdef CLNTSRV
    REGIST_MATMUL_OBJ(&que_, GetSysWorkSpacePtr(), matmul_obj_);
    matmul_obj_.Init(&tiling_);
#else
    matmul_obj_.Init(&tiling_, &que_);
#endif
    for (uint64_t mmIdx = start_idx; mmIdx < bmm_num_; mmIdx += block_num_) {
      matmul_obj_.SetWorkspace(GetSysWorkSpacePtr(), 16 * 1024 * 1024);
      matmul_obj_.SetTensorA(a_global_[mmIdx * a_stride], (aT::isTrans == true));
      matmul_obj_.SetTensorB(b_global_[mmIdx * b_stride], (bT::isTrans == true));
      if (tiling_.isBias) {
        matmul_obj_.SetBias(bias_global_[mmIdx * bias_stride]);
      }
      int computeRound = 0;
      if constexpr (cT::pos == TPosition::LCM) {
        while (matmul_obj_.template Iterate<true>()) {
          set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID7);
          wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID7);
          AscendC::LocalTensor<T> outTensor = out_q_.template AllocTensor<T>();
          MamtulCompute(outTensor);
          FuseCompute(outTensor);
          CopyOut(computeRound, mmIdx);
          computeRound++;
        }
      } else {
        matmul_obj_.IterateAll(c_global_[mmIdx * c_stride]);
      }
      pipe_barrier(PIPE_ALL);
      set_atomic_none();
    }
    matmul_obj_.End();
  }

 private:
  using T = typename aT::T;
  uint32_t blck_idx_;
  uint32_t block_num_;
  uint32_t bmm_num_ = 1;
  fuseOp fuse_op_;
  TPipe que_;
  TCubeTiling tiling_;
  MMExtra mm_extra_;
  uint32_t ldc_;
  uint32_t m_;
  uint32_t n_;
  uint32_t k_a_;
  uint32_t k_b_;

  AscendC::GlobalTensor<T> a_global_;
  AscendC::GlobalTensor<T> b_global_;
  AscendC::GlobalTensor<T> c_global_;
  AscendC::GlobalTensor<T> bias_global_;
  AscendC::TQue<TPosition::VECOUT, pipeSize> out_q_;

#ifdef CLNTSRV
  matmul::Matmul<aT, bT, cT, biasT> matmul_obj_;
#else
  matmul::MatmulImpl<aT, bT, cT, biasT> matmul_obj_;
#endif

  __aicore__ inline void MamtulCompute(const AscendC::LocalTensor<T> &outTensor) {
    matmul_obj_.template GetTensorC<true>(outTensor, false, true);
  }

  __aicore__ inline void FuseCompute(const AscendC::LocalTensor<T> &outTensor) {
    if constexpr (g_coreType == AIV) {
      fuse_op_.Process(&que_, outTensor);
      out_q_.EnQue(outTensor);
    }
  }

  __aicore__ inline void CalcGMOffset(int blockIdx, int usedCoreNum, TCubeTiling *param, uint32_t *offsetA,
                                      uint32_t *offsetB, uint32_t *offsetC, uint32_t *offsetBias) {
    if (bmm_num_ > 1) {
      blockIdx = 0;
      usedCoreNum = 1;
    }
    auto temp0 = Ceil(param->M, param->singleCoreM);
    auto temp1 = Ceil(param->N, param->singleCoreN);
    auto temp2 = Ceil(param->Ka, param->singleCoreK);  // 不切K， 应该=1

    auto divideKcoreNum = usedCoreNum / temp2;

    auto mCoreIndx = (blockIdx % divideKcoreNum) % temp0;
    auto nCoreIndx = (blockIdx % divideKcoreNum) / temp0;
    auto subKindx = blockIdx / divideKcoreNum;  // 缺省为0

    if constexpr (aT::format == CubeFormat::ND) {
      if constexpr (aT::isTrans == true) {
        *offsetA = mCoreIndx * param->singleCoreM + subKindx * param->M * param->singleCoreK;
      } else {
        *offsetA = mCoreIndx * param->Ka * param->singleCoreM + subKindx * param->singleCoreK;
      }
    } else if constexpr (aT::format == CubeFormat::NZ) {
      *offsetA = subKindx * param->singleCoreK * param->M + mCoreIndx * param->singleCoreM * BLOCK_CUBE;
    } else {
      ASSERT(false && "Data format of A matrix should be ND or NZ.");
    }

    if constexpr (bT::format == CubeFormat::ND) {
      if constexpr (bT::isTrans == true) {
        *offsetB = subKindx * param->singleCoreK + nCoreIndx * param->Kb * param->singleCoreN;
      } else {
        *offsetB = subKindx * param->singleCoreK * param->N + nCoreIndx * param->singleCoreN;
      }
    } else if constexpr (bT::format == CubeFormat::NZ) {
      *offsetB = param->Kb * nCoreIndx * param->singleCoreN + subKindx * param->singleCoreK * BLOCK_CUBE;
    } else {
      ASSERT(false && "Data format of B matrix should be ND or NZ.");
    }

    if constexpr (cT::format == CubeFormat::ND) {
      *offsetC = mCoreIndx * param->N * param->singleCoreM + nCoreIndx * param->singleCoreN;
    } else if constexpr (cT::format == CubeFormat::NZ) {
      *offsetC = param->M * nCoreIndx * param->singleCoreN + mCoreIndx * param->singleCoreM * BLOCK_CUBE;
    } else {
      ASSERT(false && "Data format of C matrix should be ND or NZ.");
    }

    if constexpr (biasT::format == CubeFormat::ND) {
      *offsetBias = nCoreIndx * param->singleCoreN;
    } else {
      ASSERT(false && "Data format of BIAS should be ND.");
    }

    // 尾块M
    int gmUseM = param->M - mCoreIndx * param->singleCoreM;
    param->singleCoreM = gmUseM < param->singleCoreM ? gmUseM : param->singleCoreM;

    // 尾块N
    int gmUseN = param->N - nCoreIndx * param->singleCoreN;
    param->singleCoreN = gmUseN < param->singleCoreN ? gmUseN : param->singleCoreN;

    // 尾块K
    int gmUseK = param->Ka - subKindx * param->singleCoreK;
    param->singleCoreK = gmUseK < param->singleCoreK ? gmUseK : param->singleCoreK;
  }

  __aicore__ inline void CopyTiling(TCubeTiling *tiling, __gm__ TCubeTiling *tilingGM, TPipe *tpipe) {
    uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);

    for (int i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++, ptr++) {
      *ptr = *(tiling32 + i);
    }
    return;
  }

  __aicore__ inline void CopyMmExtra(__gm__ MMExtra *eGM) {
    uint32_t *ptr = reinterpret_cast<uint32_t *>(&mm_extra_);
    auto extra32 = reinterpret_cast<__gm__ uint32_t *>(eGM);

    for (int i = 0; i < sizeof(MMExtra) / sizeof(uint32_t); i++, ptr++) {
      *ptr = *(extra32 + i);
    }
    return;
  }

  __aicore__ inline void CopyOut(uint32_t count, uint32_t c_idx) {
    if constexpr (g_coreType == AIC) {
      return;
    }
    auto outTensor = out_q_.template DeQue<T>();
    uint32_t c_stride = m_ * ldc_;
    const uint32_t roundM = UP_DIV(tiling_.singleCoreM, tiling_.baseM);
    const uint32_t roundN = UP_DIV(tiling_.singleCoreN, tiling_.baseN);
    uint32_t mIdx;
    uint32_t nIdx;
    if (tiling_.iterateOrder == 0) {
      mIdx = (count % roundM) * tiling_.baseM * ldc_;
      nIdx = (count / roundM) * tiling_.baseN;
    } else {
      mIdx = (count / roundN) * tiling_.baseM * ldc_;
      nIdx = (count % roundN) * tiling_.baseN;
    }

    int startOffset = mIdx + nIdx + c_idx * c_stride;
    int mLoops = (tiling_.singleCoreM + tiling_.baseM - 1) / tiling_.baseM;
    int nLoops = (tiling_.singleCoreN + tiling_.baseN - 1) / tiling_.baseN;
    int mTail = tiling_.singleCoreM - (mLoops - 1) * tiling_.baseM;
    int nTail = tiling_.singleCoreN - (nLoops - 1) * tiling_.baseN;
    int curM = ((mIdx / tiling_.baseM) == (mLoops - 1)) ? mTail : tiling_.baseM;
    int curN = ((nIdx / tiling_.baseN) == (nLoops - 1)) ? nTail : tiling_.baseN;

    bool copyPerLine = false;  // !(curN % 16 == 0 && tiling_.N % 16 == 0);
    if (!copyPerLine) {
      const int elemPerBlock = ONE_BLK_SIZE / sizeof(T);
      int blockLen = UP_DIV(curN, elemPerBlock);
      int blockCount = curM;
      int dstStride = (ldc_ - curN) * sizeof(T) / ONE_BLK_SIZE;
      int srcStride = 0;
      DataCopy(c_global_[startOffset], outTensor,
               {static_cast<uint16_t>(blockCount), static_cast<uint16_t>(blockLen), static_cast<uint16_t>(srcStride),
                static_cast<uint16_t>(dstStride)});
    }
    out_q_.FreeTensor(outTensor);
  }
};

template <typename T, bool ta, bool tb, class fuseOp, int pipeSize>
#ifdef CLNTSRV
[mix] void kernel_mm_fuse_operator(GM_ADDR a_mat, GM_ADDR b_mat, GM_ADDR c_mat, GM_ADDR bias, GM_ADDR tiling,
                                   GM_ADDR ws, GM_ADDR mm_extra = nullptr) {
#else
__aicore__ void kernel_mm_fuse_operator(GM_ADDR a_mat, GM_ADDR b_mat, GM_ADDR c_mat, GM_ADDR bias, GM_ADDR tiling,
                                        GM_ADDR ws, GM_ADDR mm_extra = nullptr) {
#endif
  typedef matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T, ta> aType;
  typedef matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T, tb> bType;
#ifdef CLNTSRV
  typedef matmul::MatmulType<AscendC::TPosition::LCM, CubeFormat::ND, T> cType;
#else
  typedef matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T> cType;
#endif
  typedef matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T> biasType;

  KernelFuseMatMul<aType, bType, cType, biasType, fuseOp, pipeSize> op;
  op.Init(a_mat, b_mat, c_mat, bias, tiling, ws, mm_extra);
  op.Process();
}
#endif
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPL_OP_CUSTOM_GEMM
