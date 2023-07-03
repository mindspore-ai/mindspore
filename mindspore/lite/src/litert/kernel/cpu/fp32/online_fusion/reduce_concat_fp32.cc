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

#include "src/litert/kernel/cpu/fp32/online_fusion/reduce_concat_fp32.h"
#include "nnacl/fp32/online_fusion/reduce_concat_fp32.h"
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
int ReduceConcatFusionCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  return RET_OK;
}

int ReduceConcatFusionCPUKernel::ReSize() {
  inner_tile_ = in_tensors_.at(0)->shape()[C2NUM];
  return RET_OK;
}

int ReduceConcatFusionCPUKernel::DoReduceConcatFusion(int task_id) {
  auto input_nums = in_tensors_.size();
  auto batch = in_tensors_.at(0)->shape()[0];
  auto batch_tile_size = out_tensors_.at(0)->shape()[2];

  Fp32ReduceSumConcatFusion(output_data_, input_datas_.data(), reduce_axis_size_.data(), input_nums, batch,
                            batch_tile_size, inner_tile_, thread_num_, task_id);
  return RET_OK;
}

int ReduceConcatFusionRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  CHECK_NULL_RETURN(cdata);
  auto fusion_kernel = reinterpret_cast<ReduceConcatFusionCPUKernel *>(cdata);
  MS_ASSERT(fusion_kernel != nullptr);
  auto error_code = fusion_kernel->DoReduceConcatFusion(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ActivationRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ReduceConcatFusionCPUKernel::Run() {
  input_datas_.clear();
  reduce_axis_size_.clear();
  for (size_t i = 0; i < in_tensors_.size(); i++) {
    input_datas_.emplace_back(reinterpret_cast<float *>(in_tensors_.at(i)->MutableData()));
    reduce_axis_size_.emplace_back(in_tensors_.at(i)->shape()[1]);
  }
  output_data_ = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());

  int error_code = ParallelLaunch(this->ms_context_, ReduceConcatFusionRun, this, thread_num_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Activation function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimType_Inner_ReduceConcatFusion, LiteKernelCreator<ReduceConcatFusionCPUKernel>)
}  // namespace mindspore::kernel
