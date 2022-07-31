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

#include "src/litert/kernel/cpu/fp16_grad/unsorted_segment_sum_fp16.h"
#include <vector>
#include <algorithm>
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "nnacl/fp16_grad/unsorted_segment_sum.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_UnsortedSegmentSum;

namespace mindspore::kernel {
int UnsortedSegmentSumCPUKernelFp16::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), 2);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_.at(0));
  CHECK_NULL_RETURN(in_tensors_.at(1));
  CHECK_NULL_RETURN(out_tensors_.at(0));
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

int UnsortedSegmentSumCPUKernelFp16::ReSize() { return RET_OK; }

int UnsortedSegmentSumFp16Run(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  CHECK_NULL_RETURN(cdata);
  auto kernel = reinterpret_cast<UnsortedSegmentSumCPUKernelFp16 *>(cdata);
  CHECK_NULL_RETURN(kernel);
  auto error_code = kernel->DoExecute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "UnsortedSegmentSum Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int UnsortedSegmentSumCPUKernelFp16::Run() {
  int error_code = ParallelLaunch(this->ms_context_, UnsortedSegmentSumFp16Run, this, 1);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Strided slice error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int UnsortedSegmentSumCPUKernelFp16::DoExecute(int task_id) {
  auto input_tensor = in_tensors_.at(0);
  auto indices_tensor = in_tensors_.at(1);
  auto output_tensor = out_tensors_.at(0);
  float16_t *input = reinterpret_cast<float16_t *>(input_tensor->data());
  CHECK_NULL_RETURN(input);
  int *indices = reinterpret_cast<int *>(indices_tensor->data());
  CHECK_NULL_RETURN(indices);
  float16_t *output = reinterpret_cast<float16_t *>(output_tensor->data());
  CHECK_NULL_RETURN(output);
  std::fill(output, output + output_tensor->ElementsNum(), 0.f);
  int ret = UnsortedSegmentSumFp16(input, unit_num_, input_dim1_, indices, output, output_dim0_, output_dim1_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "UnsortedSegmentSumFp16 error error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_UnsortedSegmentSum,
           LiteKernelCreator<UnsortedSegmentSumCPUKernelFp16>)
}  // namespace mindspore::kernel
