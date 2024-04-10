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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPL_BINARY_OP_CORE_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPL_BINARY_OP_CORE_H_

#include "tikcfw/kernel_operator.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/param.h"

namespace mindspore::ascend_native {
template <typename T>
class kernelAdd {
 public:
  __aicore__ inline void Process(const LocalTensor<T> &dstLocal, const LocalTensor<T> &src0Local,
                                 const LocalTensor<T> &src1Local, const int32_t &calCount) {
    Add<T>(dstLocal, src0Local, src1Local, calCount);
  }
};

template <int pipeSize, int blockSize, typename T, class binOp>
class KernelBinaryOp {
 public:
  __aicore__ inline KernelBinaryOp() = default;
  __aicore__ inline void Init(GM_ADDR in1, GM_ADDR in2, GM_ADDR out, uint32_t len, uint32_t len_per_core) {
    len_per_core_ = len_per_core;
    len_ = len;

    src1_global_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(in1), len_);
    src2_global_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(in2), len_);
    dst_global_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out), len_);
    pipe_.InitBuffer(inQueueX1_, pipeSize, ALIGN32(blockSize * sizeof(T)));
    pipe_.InitBuffer(inQueueX2_, pipeSize, ALIGN32(blockSize * sizeof(T)));
    pipe_.InitBuffer(outQueue_, pipeSize, ALIGN32(blockSize * sizeof(T)));
  }
  __aicore__ inline void Process() {
    int blckId = GetBlockIdx();
    int start = blckId * len_per_core_;
    int end = (blckId + 1) * len_per_core_;

    if (start > len_) return;
    if (end > len_) end = len_;
    actual_len_per_core_ = end - start;

    int block_loop = actual_len_per_core_ / blockSize;
    int tail = actual_len_per_core_ % blockSize;

    for (int i = 0; i < block_loop; i++) {
      int offset = start + i * blockSize;
      CopyIn(offset, blockSize);
      Compute(blockSize);
      CopyOut(offset, blockSize);
    }

    if (tail) {
      int offset = start + block_loop * blockSize;
      CopyIn(offset, tail);
      Compute(tail);
      CopyOut(offset, tail);
    }
  }

 private:
  __aicore__ inline void CopyIn(uint32_t offset, uint32_t size) {
    LocalTensor<T> src1Local = inQueueX1_.template AllocTensor<T>();
    LocalTensor<T> src2Local = inQueueX2_.template AllocTensor<T>();

    uint32_t cpyElem = ALIGN32(size * sizeof(T)) / sizeof(T);
    DataCopy(src1Local, src1_global_[offset], cpyElem);
    DataCopy(src2Local, src2_global_[offset], cpyElem);
    inQueueX1_.EnQue(src1Local);
    inQueueX2_.EnQue(src2Local);
  }
  __aicore__ inline void Compute(uint32_t size) {
    LocalTensor<T> dstLocal = outQueue_.template AllocTensor<T>();
    LocalTensor<T> src1Local = inQueueX1_.template DeQue<T>();
    LocalTensor<T> src2Local = inQueueX2_.template DeQue<T>();
    op.Process(dstLocal, src1Local, src2Local, size);
    outQueue_.template EnQue<T>(dstLocal);
    inQueueX1_.FreeTensor(src1Local);
    inQueueX2_.FreeTensor(src2Local);
  }
  __aicore__ inline void CopyOut(uint32_t offset, uint32_t size) {
    LocalTensor<T> dstLocal = outQueue_.template DeQue<T>();
    uint32_t cpyElem = ALIGN32(size * sizeof(T)) / sizeof(T);
    DataCopy(dst_global_[offset], dstLocal, cpyElem);
    outQueue_.FreeTensor(dstLocal);
  }

 private:
  binOp op;
  GlobalTensor<T> src1_global_;
  GlobalTensor<T> src2_global_;
  GlobalTensor<T> dst_global_;
  TPipe pipe_;
  TQue<QuePosition::VECIN, pipeSize> inQueueX1_;
  TQue<QuePosition::VECIN, pipeSize> inQueueX2_;
  TQue<QuePosition::VECOUT, pipeSize> outQueue_;
  uint32_t len_ = 0;
  uint32_t len_per_core_ = 0;
  uint32_t actual_len_per_core_ = 0;
};
}  // namespace mindspore::ascend_native
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPL_BINARY_OP_H_
