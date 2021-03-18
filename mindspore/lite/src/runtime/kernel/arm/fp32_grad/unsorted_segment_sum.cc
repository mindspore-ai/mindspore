/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/arm/fp32_grad/unsorted_segment_sum.h"
#include <vector>
#include <algorithm>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "nnacl/fp32_grad/unsorted_segment_sum.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_UnsortedSegmentSum;

namespace mindspore::kernel {

int UnsortedSegmentSumCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  auto input_shape = in_tensors_.at(0)->shape();
  auto segment_ids_shape = in_tensors_.at(1)->shape();
  auto output_shape = out_tensors_.at(0)->shape();
  unit_num_ = 1;
  input_dim1_ = 1;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    unit_num_ *= input_shape[i];
    if (i >= segment_ids_shape.size()) {
      input_dim1_ *= input_shape[i];
    }
  }
  output_dim0_ = output_shape[0];
  output_dim1_ = 1;
  for (size_t j = 1; j < output_shape.size(); j++) {
    output_dim1_ *= output_shape[j];
  }
  return RET_OK;
}

int UnsortedSegmentSumCPUKernel::ReSize() { return RET_OK; }

int UnsortedSegmentSumRun(void *cdata, int task_id) {
  MS_ASSERT(cdata != nullptr);
  auto kernel = reinterpret_cast<UnsortedSegmentSumCPUKernel *>(cdata);
  auto error_code = kernel->Execute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "UnsortedSegmentSum Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int UnsortedSegmentSumCPUKernel::Run() {
  int error_code = ParallelLaunch(this->context_->thread_pool_, UnsortedSegmentSumRun, this, 1);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Strided slice error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int UnsortedSegmentSumCPUKernel::Execute(int task_id) {
  int ret;
  auto input_tensor = in_tensors_.at(0);
  auto indices_tensor = in_tensors_.at(1);
  auto output_tensor = out_tensors_.at(0);
  float *input = reinterpret_cast<float *>(input_tensor->data_c());
  int *indices = reinterpret_cast<int *>(indices_tensor->data_c());
  float *output = reinterpret_cast<float *>(output_tensor->MutableData());
  std::fill(output, output + output_tensor->ElementsNum(), 0.f);
  ret = UnsortedSegmentSum(input, unit_num_, input_dim1_, indices, output, output_dim0_, output_dim1_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "StridedSliceGrad error error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_UnsortedSegmentSum, LiteKernelCreator<UnsortedSegmentSumCPUKernel>)
}  // namespace mindspore::kernel
