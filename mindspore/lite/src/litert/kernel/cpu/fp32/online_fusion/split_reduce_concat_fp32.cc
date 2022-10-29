/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/cpu/fp32/online_fusion/split_reduce_concat_fp32.h"
#include "nnacl/fp32/online_fusion/split_reduce_concat_fp32.h"
#include <algorithm>
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "src/litert/infer_manager.h"
#include "src/litert/kernel/cpu/base/split_base.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int SplitReduceConcatFusionCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  return RET_OK;
}

int SplitReduceConcatFusionCPUKernel::ReSize() {
  auto in_tensor = in_tensors_.front();
  CHECK_NULL_RETURN(in_tensor);
  auto status = SplitBaseCPUKernel::CheckAndInitSplitParam(*in_tensor, param_);
  if (RET_OK != status) {
    MS_LOG(ERROR) << "CheckAndInitSplitParam failed";
    return status;
  }
  axis_ = param_->split_dim_;
  split_slices_ = param_->split_sizes_;

  outer_size_ = 1;
  auto shape = in_tensors_.at(0)->shape();
  for (size_t i = 0; i < axis_; i++) {
    outer_size_ *= shape[i];
  }

  inner_size_ = 1;
  for (size_t i = axis_ + 1; i < shape.size(); i++) {
    inner_size_ *= shape[i];
  }

  mid_size_ = shape[axis_];
  mid_len_ = out_tensors_.at(0)->shape()[axis_];
  return RET_OK;
}

int SplitReduceConcatFusionCPUKernel::DoSplitReduceConcatFusion(int task_id) {
  auto input_addr = reinterpret_cast<float *>(in_tensors_.at(0)->MutableData());
  auto output_addr = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());

  auto outer_size_tile = UP_DIV(outer_size_, thread_num_);
  auto in_data = input_addr + task_id * outer_size_tile * inner_size_ * mid_size_;
  auto out_data = output_addr + task_id * outer_size_tile * inner_size_ * mid_len_;
  auto outer_size_tile_tmp = MSMIN(outer_size_tile, outer_size_ - task_id * outer_size_tile);
  Fp32SplitReduceSumConcatFusion(in_data, out_data, inner_size_, mid_size_, split_slices_, mid_len_,
                                 outer_size_tile_tmp);
  return RET_OK;
}

int SplitReduceConcatFusionRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  CHECK_NULL_RETURN(cdata);
  auto fusion_kernel = reinterpret_cast<SplitReduceConcatFusionCPUKernel *>(cdata);
  MS_ASSERT(fusion_kernel != nullptr);
  auto error_code = fusion_kernel->DoSplitReduceConcatFusion(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ActivationRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SplitReduceConcatFusionCPUKernel::Run() {
  int error_code = ParallelLaunch(this->ms_context_, SplitReduceConcatFusionRun, this, thread_num_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Activation function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimType_Inner_SplitReduceConcatFusion,
           LiteKernelCreator<SplitReduceConcatFusionCPUKernel>)
}  // namespace mindspore::kernel
