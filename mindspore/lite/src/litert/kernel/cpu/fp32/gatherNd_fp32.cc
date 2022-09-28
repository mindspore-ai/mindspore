/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/cpu/fp32/gatherNd_fp32.h"
#include <limits>
#include <vector>
#include "schema/model_generated.h"
#include "include/errorcode.h"
#include "src/litert/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_GatherNd;

namespace mindspore::kernel {
GatherNdCPUKernel::~GatherNdCPUKernel() {
  if (in_offset_ != nullptr) {
    free(in_offset_);
    in_offset_ = nullptr;
  }
}

int GatherNdCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int GatherNdCPUKernel::ReSize() {
  if (in_offset_ != nullptr) {
    free(in_offset_);
    in_offset_ = nullptr;
  }
  auto indices_tensor = in_tensors_.at(1);
  auto indices_shape = indices_tensor->shape();
  int indices_rank = indices_shape.size();
  count_ = 1;
  for (int i = 0; i < indices_rank - 1; ++i) {
    MS_CHECK_FALSE_MSG(INT_MUL_OVERFLOW(count_, indices_shape[i]), RET_ERROR, "mul overflow.");
    count_ *= indices_shape[i];
  }
  if (count_ >= std::numeric_limits<int>::max() / static_cast<int>(sizeof(int))) {
    MS_LOG(ERROR) << "count_ is invalid, count_: " << count_;
    return RET_ERROR;
  }
  in_offset_ = reinterpret_cast<int *>(malloc(count_ * sizeof(int)));
  if (in_offset_ == nullptr) {
    MS_LOG(ERROR) << "GatherNd Malloc in_offset_ error!";
    return RET_ERROR;
  }
  thread_sz_count_ = MSMIN(thread_count_, count_);
  if (thread_sz_count_ != 0) {
    thread_sz_stride_ = UP_DIV(count_, thread_sz_count_);
  }
  return RET_OK;
}

int GatherNdCPUKernel::InitOffset() {
  MS_ASSERT(in_offset_ != nullptr);
  auto indices_tensor = in_tensors_.at(1);
  auto indices_shape = indices_tensor->shape();
  auto in_shape = in_tensors_.front()->shape();
  int indices_rank = indices_shape.size();
  int in_rank = in_shape.size();
  if (indices_rank < 1) {
    MS_LOG(ERROR) << name() << " indices shape size must be greater than or equal to 1!";
    return RET_ERROR;
  }
  int idx_lastshape = indices_shape[indices_rank - 1];
  if (idx_lastshape > in_rank) {
    MS_LOG(ERROR) << name() << " indices shape error!";
    return RET_ERROR;
  }

  area_ = 1;
  for (int i = idx_lastshape; i < in_rank; ++i) {
    area_ *= in_shape[i];
  }
  std::vector<int> in_stride(in_rank);
  in_stride[in_rank - 1] = 1;
  for (int i = in_rank - 2; i >= 0; --i) {
    in_stride[i] = in_shape[i + 1] * in_stride[i + 1];
  }

  int idx_stride = idx_lastshape;
  (void)memset(in_offset_, 0, count_ * sizeof(int));

  if (indices_tensor->data_type() == kNumberTypeInt || indices_tensor->data_type() == kNumberTypeInt32) {
    auto indices_ptr = reinterpret_cast<int *>(indices_tensor->data());
    CHECK_NULL_RETURN(indices_ptr);
    for (int j = 0; j < count_; ++j) {
      for (int k = 0; k < idx_lastshape; ++k) {
        MS_CHECK_LT(indices_ptr[j * idx_stride + k], in_shape[k], RET_ERROR);
        MS_CHECK_GE(indices_ptr[j * idx_stride + k], 0, RET_ERROR);
        in_offset_[j] += indices_ptr[j * idx_stride + k] * in_stride.at(k);
      }
    }
  } else if (indices_tensor->data_type() == kNumberTypeInt64) {
    auto indices_ptr = reinterpret_cast<int64_t *>(indices_tensor->data());
    CHECK_NULL_RETURN(indices_ptr);
    for (int j = 0; j < count_; ++j) {
      for (int k = 0; k < idx_lastshape; ++k) {
        MS_CHECK_LT(indices_ptr[j * idx_stride + k], in_shape[k], RET_ERROR);
        MS_CHECK_GE(indices_ptr[j * idx_stride + k], 0, RET_ERROR);
        in_offset_[j] += indices_ptr[j * idx_stride + k] * in_stride.at(k);
      }
    }
  } else {
    MS_LOG(ERROR) << "Unsupported data type for indices tensor.";
    return RET_ERROR;
  }
  return RET_OK;
}

int GatherNdCPUKernel::DoGatherNd(int task_id) const {
  MS_CHECK_INT_MUL_NOT_OVERFLOW(task_id, thread_sz_stride_, RET_ERROR);
  int count = MSMIN(thread_sz_stride_, count_ - task_id * thread_sz_stride_);
  if (count <= 0) {
    return RET_OK;
  }
  int offset = task_id * thread_sz_stride_;
  int dtype_len = lite::DataTypeSize(in_tensors_.front()->data_type());
  MS_CHECK_INT_MUL_NOT_OVERFLOW(offset, area_, RET_ERROR);
  auto ret = GatherNd(in_ptr_, static_cast<int8_t *>(out_ptr_) + offset * area_ * dtype_len, in_offset_ + offset, area_,
                      count, dtype_len);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "GatherNdRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int GatherNdRun(void *cdata, int task_id, float, float) {
  auto g_kernel = reinterpret_cast<const GatherNdCPUKernel *>(cdata);
  auto ret = g_kernel->DoGatherNd(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "GatherNdRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int GatherNdCPUKernel::Run() {
  in_ptr_ = in_tensors_.front()->data();
  out_ptr_ = out_tensors_.front()->data();
  CHECK_NULL_RETURN(in_ptr_);
  CHECK_NULL_RETURN(out_ptr_);
  if (InitOffset() != RET_OK) {
    MS_LOG(ERROR) << "InitOffset failed.";
    return RET_ERROR;
  }
  auto ret = ParallelLaunch(this->ms_context_, GatherNdRun, this, thread_sz_count_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "gatherNd error error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_GatherNd, LiteKernelCreator<GatherNdCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_GatherNd, LiteKernelCreator<GatherNdCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_GatherNd, LiteKernelCreator<GatherNdCPUKernel>)
#ifdef ENABLE_FP16
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_GatherNd, LiteKernelCreator<GatherNdCPUKernel>)
#endif
}  // namespace mindspore::kernel
