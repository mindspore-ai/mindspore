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

#include "src/runtime/kernel/arm/fp32/reverse_sequence_fp32.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ReverseSequence;

namespace mindspore::kernel {
int ReverseSequenceCPUKernel::Init() {
  MS_CHECK_TRUE_RET(in_tensors_.size() == kInputSize1, RET_ERROR);
  MS_CHECK_TRUE_RET(out_tensors_.size() == 1, RET_ERROR);
  CHECK_NULL_RETURN(in_tensors_[0]);
  CHECK_NULL_RETURN(in_tensors_[1]);
  CHECK_NULL_RETURN(out_tensors_[0]);
  CHECK_NULL_RETURN(op_parameter_);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

void ReverseSequenceCPUKernel::ConvertAxisToPositive(const std::vector<int> shape, int *axis) const {
  if (axis != nullptr && *axis < 0) {
    *axis += static_cast<int>(shape.size());
  }
}

int ReverseSequenceCPUKernel::CalcCountPreAxis(const std::vector<int> shape, int axis) const {
  int count = 1;
  for (int i = 0; i < axis; ++i) {
    count *= shape.at(i);
  }
  return count;
}
int ReverseSequenceCPUKernel::CalcCountAfterAxis(const std::vector<int> shape, int axis) const {
  int count = 1;
  for (size_t i = axis + 1; i < shape.size(); ++i) {
    count *= shape.at(i);
  }
  return count;
}

int ReverseSequenceCPUKernel::ReSize() {
  auto input0 = in_tensors_.at(0);
  auto output = out_tensors_.at(0);

  auto para = reinterpret_cast<ReverseSequenceParameter *>(op_parameter_);

  ConvertAxisToPositive(input0->shape(), &(para->batch_axis_));
  ConvertAxisToPositive(input0->shape(), &(para->seq_axis_));

  para->ndim_ = input0->shape().size();
  for (int i = 0; i < para->ndim_; i++) {
    para->input_shape0_[i] = input0->DimensionSize(i);
    para->output_shape_[i] = output->DimensionSize(i);
  }

  int less_axis = MSMIN(para->batch_axis_, para->seq_axis_);
  int greater_axis = MSMAX(para->batch_axis_, para->seq_axis_);

  para->outer_count_ = CalcCountPreAxis(input0->shape(), less_axis);
  para->outer_stride_ = input0->DimensionSize(less_axis) * CalcCountAfterAxis(input0->shape(), less_axis);

  para->inner_count_ = 1;
  for (int i = less_axis + 1; i < greater_axis; ++i) {
    para->inner_count_ *= input0->DimensionSize(i);
  }

  para->inner_stride_ = input0->DimensionSize(greater_axis) * CalcCountAfterAxis(input0->shape(), greater_axis);

  para->copy_byte_size_ = sizeof(float) * CalcCountAfterAxis(input0->shape(), greater_axis);
  para->total_data_size_ = input0->Size();
  return RET_OK;
}

int ReverseSequenceCPUKernel::Run() {
  float *input0 = reinterpret_cast<float *>(in_tensors_.at(0)->MutableData());
  void *input1 = in_tensors_.at(1)->MutableData();
  float *output = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());
  ReverseSequenceParameter *param = reinterpret_cast<ReverseSequenceParameter *>(op_parameter_);
  CHECK_NULL_RETURN(param);
  param->is_seq_length_int32_ = in_tensors_.at(1)->data_type() == kNumberTypeInt32;
  CHECK_NULL_RETURN(input0);
  CHECK_NULL_RETURN(input1);
  CHECK_NULL_RETURN(output);
  ReverseSequence(input0, input1, output, param);
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ReverseSequence, LiteKernelCreator<ReverseSequenceCPUKernel>)
}  // namespace mindspore::kernel
