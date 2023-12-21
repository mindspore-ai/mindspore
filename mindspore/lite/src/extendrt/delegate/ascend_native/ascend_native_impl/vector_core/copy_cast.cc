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

#include <string>
#include "extendrt/delegate/ascend_native/ascend_native_impl/vector_core/copy_cast.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/utils.h"
#include "tikcfw/kernel_operator.h"
#include "tikcfw/impl/kernel_utils.h"
using AscendC::GetBlockIdx;
using AscendC::GetBlockNum;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::QuePosition;
using AscendC::RoundMode;
using AscendC::TPipe;
using AscendC::TQue;
namespace mindspore::ascend_native {
template <int pipeSize, int blockSize, typename srcType, typename dstType>
class KernelCopyCast {
 public:
  __aicore__ inline KernelCopyCast() {}
  __aicore__ inline void Init(GM_ADDR src_gm, GM_ADDR dst_gm, uint32_t token_num, uint32_t token_per_core) {
    token_per_core_ = token_per_core;
    token_num_ = token_num;

    src_global_.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(src_gm), token_num_);
    dst_global_.SetGlobalBuffer(reinterpret_cast<__gm__ dstType *>(dst_gm), token_num_);
    pipe_.InitBuffer(inQueueX_, pipeSize, ALIGN32(blockSize * sizeof(srcType)));
    pipe_.InitBuffer(outQueue_, pipeSize, ALIGN32(blockSize * sizeof(dstType)));
  }

  __aicore__ inline void Process() {
    int blckId = GetBlockIdx();
    int start = blckId * token_per_core_;
    int end = (blckId + 1) * token_per_core_;

    if (start > token_num_) return;
    if (end > token_num_) end = token_num_;
    actual_token_per_core_ = end - start;

    int block_loop = actual_token_per_core_ / blockSize;
    int tail = actual_token_per_core_ % blockSize;
    // per block cast
    for (int i = 0; i < block_loop; i++) {
      int offset = start + i * blockSize;
      copyIn(offset, blockSize);
      compute(blockSize);
      copyOut(offset, blockSize);
    }
    // tail cast
    if (tail) {
      int offset = start + block_loop * blockSize;
      copyIn(offset, tail);
      compute(tail);
      copyOut(offset, tail);
    }
  }

 private:
  __aicore__ inline void copyIn(uint32_t offset, uint32_t size) {
    LocalTensor<srcType> srcLocal = inQueueX_.template AllocTensor<srcType>();
    uint32_t cpyElem = ALIGN32(size * sizeof(srcType)) / sizeof(srcType);
    DataCopy(srcLocal, src_global_[offset], cpyElem);
    inQueueX_.EnQue(srcLocal);
  }

  __aicore__ inline void compute(uint32_t size) {
    LocalTensor<dstType> dstLocal = outQueue_.template AllocTensor<dstType>();
    LocalTensor<srcType> srcLocal = inQueueX_.template DeQue<srcType>();
    Cast(dstLocal, srcLocal, RoundMode::CAST_NONE, size);
    outQueue_.template EnQue<dstType>(dstLocal);
    inQueueX_.FreeTensor(srcLocal);
  }

  __aicore__ inline void copyOut(uint32_t offset, uint32_t size) {
    LocalTensor<dstType> dstLocal = outQueue_.template DeQue<dstType>();
    uint32_t cpyElem = ALIGN32(size * sizeof(dstType)) / sizeof(dstType);
    DataCopy(dst_global_[offset], dstLocal, cpyElem);
    outQueue_.FreeTensor(dstLocal);
  }

 private:
  GlobalTensor<srcType> src_global_;
  GlobalTensor<dstType> dst_global_;
  TPipe pipe_;
  TQue<QuePosition::VECIN, pipeSize> inQueueX_;
  TQue<QuePosition::VECOUT, pipeSize> outQueue_;
  uint32_t token_num_ = 0;
  uint32_t token_per_core_ = 0;
  uint32_t actual_token_per_core_ = 0;
};

template <int pipeSize, int blockSize, typename srcType, typename dstType>
__aicore__ void kernel_copy_cast_operator(GM_ADDR src_gm, GM_ADDR dst_gm, uint32_t token_number,
                                          uint32_t tokenPerCore) {
  KernelCopyCast<pipeSize, blockSize, srcType, dstType> op;
  op.Init(src_gm, dst_gm, token_number, tokenPerCore);
  op.Process();
}

__global__ __aicore__ void cast_custom(GM_ADDR src_gm, GM_ADDR dst_gm, uint32_t token_number, bool f_to_h) {
  int blockNum = GetBlockNum();
  int tokenPerCore = ALIGN32((ALIGN(token_number, BLOCK_CAST) / blockNum));
  if (f_to_h)
    kernel_copy_cast_operator<PIPE_CAST, BLOCK_CAST, float, half>(src_gm, dst_gm, token_number, tokenPerCore);
  else
    kernel_copy_cast_operator<PIPE_CAST, BLOCK_CAST, half, float>(src_gm, dst_gm, token_number, tokenPerCore);
}

// src - host memory
// dst - device memory (allocated if null)
template <typename t_src, typename t_dst, size_t ub_size>
void CopyCastHTD(void *src, void **dst_ptr, size_t elem_num, void *q, void *ctx) {
  void *dst = nullptr;
  if (*(dst_ptr) == nullptr) {
    *(dst_ptr) = ascend_native::MallocDevice(elem_num * sizeof(t_dst), ctx);
  }
  dst = *(dst_ptr);
  if constexpr (!std::is_same<t_src, t_dst>::value) {
    void *g_src = nullptr;
    g_src = ascend_native::MallocDevice(elem_num * sizeof(t_src), ctx);
    ascend_native::CopyHTD(g_src, src, elem_num * sizeof(t_src), ctx);
    int core_num = ALIGN(elem_num, BLOCK_CAST) / BLOCK_CAST;
    if (core_num > VEC_CORE_NUM) core_num = VEC_CORE_NUM;
    bool float_to_half = false;
    if (std::is_same<t_src, float>::value && std::is_same<t_dst, half>::value) float_to_half = true;
    cast_custom<<<core_num, nullptr, q>>>(reinterpret_cast<GM_ADDR>(g_src), reinterpret_cast<GM_ADDR>(dst), elem_num,
                                          float_to_half);
    SyncDevice(q, ctx);
    FreeDevice(g_src, ctx);
  } else {
    ascend_native::CopyHTD(dst, src, elem_num * sizeof(t_src), ctx);
    ascend_native::SyncDevice(q, ctx);
  }
}

// src - device memory
// dst - host memory
template <typename t_src, typename t_dst, size_t ub_size>
void CopyCastDTH(void *src, void *dst, size_t elem_num, void *q, void *ctx) {
  if constexpr (!std::is_same<t_src, t_dst>::value) {
    void *g_dst = nullptr;
    g_dst = ascend_native::MallocDevice(elem_num * sizeof(t_dst), ctx);
    int core_num = ALIGN(elem_num, BLOCK_CAST) / BLOCK_CAST;
    if (core_num > VEC_CORE_NUM) core_num = VEC_CORE_NUM;
    bool float_to_half = false;
    if (std::is_same<t_src, float>::value && std::is_same<t_dst, half>::value) float_to_half = true;
    cast_custom<<<core_num, nullptr, q>>>(reinterpret_cast<GM_ADDR>(src), reinterpret_cast<GM_ADDR>(g_dst), elem_num,
                                          float_to_half);
    SyncDevice(q, ctx);
    ascend_native::CopyDTH(dst, g_dst, elem_num * sizeof(t_src), ctx);
    FreeDevice(g_dst, ctx);
  } else {
    ascend_native::CopyDTH(dst, src, elem_num * sizeof(t_src), ctx);
  }
}
void CopyHostFp32ToDeviceFp16(void *src, void **dst_ptr, size_t elem_num, void *stream, void *ctx) {
  CopyCastHTD<float, half, BLOCK_CAST>(src, dst_ptr, elem_num, stream, ctx);
}

void CopyHostFp32ToDeviceFp32(void *src, void **dst_ptr, size_t elem_num, void *stream, void *ctx) {
  CopyCastHTD<float, float, BLOCK_CAST>(src, dst_ptr, elem_num, stream, ctx);
}

void CopyHostFp16ToDeviceFp16(void *src, void **dst_ptr, size_t elem_num, void *stream, void *ctx) {
  CopyCastHTD<half, half, BLOCK_CAST>(src, dst_ptr, elem_num, stream, ctx);
}

void CopyHostFp16ToDeviceFp32(void *src, void **dst_ptr, size_t elem_num, void *stream, void *ctx) {
  CopyCastHTD<half, float, BLOCK_CAST>(src, dst_ptr, elem_num, stream, ctx);
}

void CopyDeviceFp16ToHostFp32(void *src, void *dst, size_t elem_num, void *stream, void *ctx) {
  CopyCastDTH<half, float, BLOCK_CAST>(src, dst, elem_num, stream, ctx);
}

void CopyDeviceFp32ToHostFp32(void *src, void *dst, size_t elem_num, void *stream, void *ctx) {
  CopyCastDTH<float, float, BLOCK_CAST>(src, dst, elem_num, stream, ctx);
}

void CopyDeviceFp16ToHostFp16(void *src, void *dst, size_t elem_num, void *stream, void *ctx) {
  CopyCastDTH<half, half, BLOCK_CAST>(src, dst, elem_num, stream, ctx);
}

void CopyDeviceFp32ToHostFp16(void *src, void *dst, size_t elem_num, void *stream, void *ctx) {
  CopyCastDTH<float, half, BLOCK_CAST>(src, dst, elem_num, stream, ctx);
}
}  // namespace mindspore::ascend_native
