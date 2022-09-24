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

#include "src/litert/kernel/cpu/int8/gatherNd_int8.h"
#include <cstring>
#include <limits>
#include <vector>
#include "schema/model_generated.h"
#include "include/errorcode.h"
#include "src/litert/kernel_registry.h"
#include "nnacl/int8/gatherNd_int8.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_GatherNd;

namespace mindspore::kernel {
GatherNdInt8CPUKernel::~GatherNdInt8CPUKernel() {
  if (in_offset_ != nullptr) {
    free(in_offset_);
    in_offset_ = nullptr;
  }
}

int GatherNdInt8CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_[0]);
  CHECK_NULL_RETURN(in_tensors_[1]);
  CHECK_NULL_RETURN(out_tensors_[0]);
  if (in_tensors_[0]->data_type() != mindspore::kNumberTypeInt8 ||
      out_tensors_[0]->data_type() != mindspore::kNumberTypeInt8) {
    MS_LOG(ERROR) << "Datatype error, input0 data_type is " << in_tensors_[0]->data_type() << ", output data_type is "
                  << out_tensors_[0]->data_type();
    return RET_ERROR;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int GatherNdInt8CPUKernel::ReSize() {
  if (in_offset_ != nullptr) {
    free(in_offset_);
    in_offset_ = nullptr;
  }
  auto in_quant_args = in_tensors_.at(0)->quant_params();
  auto out_quant_args = out_tensors_.at(0)->quant_params();
  MS_CHECK_TRUE_MSG(!in_quant_args.empty(), RET_ERROR, "Input quant param cannot be empty.");
  MS_CHECK_TRUE_MSG(!out_quant_args.empty(), RET_ERROR, "Output quant param cannot be empty.");

  param_.alpha_ = in_quant_args.front().scale / out_quant_args.front().scale;
  param_.zp_in_ = in_quant_args.front().zeroPoint;
  param_.zp_out_ = out_quant_args.front().zeroPoint;

  auto indices_tensor = in_tensors_.at(1);
  auto indices_shape = indices_tensor->shape();
  int indices_rank = static_cast<size_t>(indices_shape.size());
  count_ = 1;
  for (int i = 0; i < indices_rank - 1; ++i) {
    count_ *= indices_shape[i];
  }
  if (count_ >= std::numeric_limits<int>::max() / static_cast<int>(sizeof(int))) {
    MS_LOG(ERROR) << "count_ is invalid, count_: " << count_;
    return RET_ERROR;
  }
  in_offset_ = reinterpret_cast<int *>(malloc(static_cast<size_t>(count_) * sizeof(int)));
  if (in_offset_ == nullptr) {
    MS_LOG(ERROR) << "GatherNdInt8 Malloc in_offset_ error!";
    return RET_ERROR;
  }
  (void)memset(in_offset_, 0, static_cast<size_t>(count_) * sizeof(int));
  thread_sz_count_ = MSMIN(thread_count_, count_);
  if (thread_sz_count_ == 0) {
    MS_LOG(ERROR) << "div zero";
    return RET_ERROR;
  }
  thread_sz_stride_ = UP_DIV(count_, thread_sz_count_);
  return RET_OK;
}

int GatherNdInt8CPUKernel::InitOffset() {
  auto ind_quant_args = in_tensors_.at(1)->quant_params();
  auto indices_tensor = in_tensors_.at(1);
  auto indices_shape = indices_tensor->shape();
  int indices_rank = static_cast<size_t>(indices_shape.size());
  auto in_shape = in_tensors_.front()->shape();
  int in_rank = static_cast<size_t>(in_shape.size());
  if (indices_rank < 1) {
    MS_LOG(ERROR) << "index out of bounds";
    return RET_ERROR;
  }
  int idx_lastshape = indices_shape.at(indices_rank - 1);
  if (idx_lastshape > in_rank) {
    MS_LOG(ERROR) << name() << " indices shape error!";
    return RET_ERROR;
  }
  auto indices_ptr = reinterpret_cast<int8_t *>(indices_tensor->data());
  CHECK_NULL_RETURN(indices_ptr);
  area_ = 1;
  for (int i = idx_lastshape; i < in_rank; ++i) {
    area_ *= in_shape.at(i);
  }
  std::vector<int> in_stride(in_rank);
  in_stride[in_rank - 1] = 1;
  for (int i = in_rank - 2; i >= 0; --i) {
    in_stride[i] = in_shape[i + 1] * in_stride[i + 1];
  }

  int idx_stride = idx_lastshape;
  for (int j = 0; j < count_; ++j) {
    for (int k = 0; k < idx_lastshape; ++k) {
      int tmp = static_cast<int>(
        round((indices_ptr[j * idx_stride + k] - ind_quant_args.front().zeroPoint) * ind_quant_args.front().scale));
      if (tmp < in_shape[k]) {
        in_offset_[j] += tmp * in_stride[k];
      } else {
        MS_LOG(ERROR) << name() << " indices value invalid!";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

int GatherNdInt8CPUKernel::DoGatherNd(int task_id) {
  int count = MSMIN(thread_sz_stride_, count_ - task_id * thread_sz_stride_);
  if (count <= 0) {
    return RET_OK;
  }
  int offset = task_id * thread_sz_stride_;
  auto ret = GatherNdInt8(in_ptr_, out_ptr_ + offset * area_, in_offset_ + offset, area_, count, param_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "GatherNdRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int GatherNdInt8Run(void *cdata, int task_id, float, float) {
  auto g_kernel = reinterpret_cast<GatherNdInt8CPUKernel *>(cdata);
  auto ret = g_kernel->DoGatherNd(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "GatherNdRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int GatherNdInt8CPUKernel::Run() {
  in_ptr_ = reinterpret_cast<int8_t *>(in_tensors_.front()->data());
  out_ptr_ = reinterpret_cast<int8_t *>(out_tensors_.front()->data());
  CHECK_NULL_RETURN(in_ptr_);
  CHECK_NULL_RETURN(out_ptr_);
  auto ret = InitOffset();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "InitOffset failed.";
    return ret;
  }
  ret = ParallelLaunch(this->ms_context_, GatherNdInt8Run, this, thread_sz_count_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "gatherNd error error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_GatherNd, LiteKernelCreator<GatherNdInt8CPUKernel>)
}  // namespace mindspore::kernel
