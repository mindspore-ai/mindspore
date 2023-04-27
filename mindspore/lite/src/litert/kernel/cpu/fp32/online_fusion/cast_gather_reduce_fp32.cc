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

#include "src/litert/kernel/cpu/fp32/online_fusion/cast_gather_reduce_fp32.h"
#include "nnacl/fp32/online_fusion/cast_gather_reduce_fp32.h"
#include <algorithm>
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "src/litert/infer_manager.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int CastGatherReduceFusionCPUKernel::Prepare() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int CastGatherReduceFusionCPUKernel::ReSize() { return RET_OK; }

int CastGatherReduceFusionCPUKernel::DoCastGatherReduceFusion(int task_id) {
  auto input_data_shape = in_tensors_.at(0)->shape();
  auto input_data_inner_size_ = input_data_shape[1];
  auto input_data = reinterpret_cast<float *>(in_tensors_.at(0)->MutableData());

  auto input_indices_shape = in_tensors_.at(1)->shape();
  inner_size_ = input_indices_shape.back();
  outer_size_ = input_indices_shape.front();

  auto output_data = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());

  auto outer_size_tile = UP_DIV(outer_size_, thread_num_);
  auto outer_start = task_id * outer_size_tile;
  auto outer_end = MSMIN(outer_size_, outer_size_tile + outer_start);

  if (in_tensors_.at(1)->data_type() == TypeId::kNumberTypeInt64) {
    Fp32CastGatherReduceInt64Fusion(output_data, reinterpret_cast<int64_t *>(in_tensors_.at(1)->MutableData()),
                                    input_data, inner_size_, input_data_inner_size_, outer_start, outer_end);
  } else if (in_tensors_.at(1)->data_type() == TypeId::kNumberTypeInt32 ||
             in_tensors_.at(1)->data_type() == TypeId::kNumberTypeInt) {
    Fp32CastGatherReduceInt32Fusion(output_data, reinterpret_cast<int32_t *>(in_tensors_.at(1)->MutableData()),
                                    input_data, inner_size_, input_data_inner_size_, outer_start, outer_end);
  } else {
    MS_LOG(ERROR) << "Dont support data type is : " << in_tensors_.at(1)->data_type();
    return RET_ERROR;
  }
  return RET_OK;
}

int CastGatherReduceFusionRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  CHECK_NULL_RETURN(cdata);
  auto fusion_kernel = reinterpret_cast<CastGatherReduceFusionCPUKernel *>(cdata);
  MS_ASSERT(fusion_kernel != nullptr);
  auto error_code = fusion_kernel->DoCastGatherReduceFusion(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "CastGatherReduceFusionRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int CastGatherReduceFusionCPUKernel::Run() {
  int error_code = ParallelLaunch(this->ms_context_, CastGatherReduceFusionRun, this, thread_num_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Activation function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimType_Inner_CastGatherReduceFusion,
           LiteKernelCreator<CastGatherReduceFusionCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimType_Inner_CastGatherReduceFusion,
           LiteKernelCreator<CastGatherReduceFusionCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt64, PrimType_Inner_CastGatherReduceFusion,
           LiteKernelCreator<CastGatherReduceFusionCPUKernel>)
}  // namespace mindspore::kernel
