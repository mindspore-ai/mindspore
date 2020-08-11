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

#include "src/runtime/kernel/arm/fp32/reverse_sequence.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ReverseSequence;

namespace mindspore::kernel {
int ReverseSequenceCPUKernel::Init() {
  if (context_->infer_shape_interrupt_ && !context_->running_) {
    set_need_reinit();
    return RET_OK;
  }
  auto input0 = in_tensors_.at(0);
  auto input1 = in_tensors_.at(1);
  auto output = out_tensors_.at(0);
  MS_ASSERT(input0 != nullptr);
  MS_ASSERT(input1 != nullptr);
  MS_ASSERT(output != nullptr);

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

void ReverseSequenceCPUKernel::ConvertAxisToPositive(const std::vector<int> shape, int *axis) {
  if (axis != nullptr && *axis < 0) {
    *axis += shape.size();
  }
}

int ReverseSequenceCPUKernel::CalcCountPreAxis(const std::vector<int> shape, int axis) {
  int count = 1;
  for (int i = 0; i < axis; ++i) {
    count *= shape[i];
  }
  return count;
}
int ReverseSequenceCPUKernel::CalcCountAfterAxis(const std::vector<int> shape, int axis) {
  int count = 1;
  for (int i = axis + 1; i < shape.size(); ++i) {
    count *= shape[i];
  }
  return count;
}

int ReverseSequenceCPUKernel::ReSize() { return RET_OK; }

int ReverseSequenceCPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare failed.";
    return RET_ERROR;
  }
  float *input0 = reinterpret_cast<float *>(in_tensors_.at(0)->Data());
  int *input1 = reinterpret_cast<int *>(in_tensors_.at(1)->Data());
  float *output = reinterpret_cast<float *>(out_tensors_.at(0)->Data());
  ReverseSequence(input0, input1, output, reinterpret_cast<ReverseSequenceParameter *>(op_parameter_));
  return RET_OK;
}

kernel::LiteKernel *CpuReverseSequenceFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                        const std::vector<lite::tensor::Tensor *> &outputs,
                                                        OpParameter *parameter, const lite::Context *ctx,
                                                        const KernelKey &desc, const lite::Primitive *primitive) {
  MS_ASSERT(parameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_ReverseSequence);
  auto *kernel = new (std::nothrow) ReverseSequenceCPUKernel(parameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Create kernel failed, name: " << parameter->name_;
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << parameter->name_
                  << ", type: " << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(parameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ReverseSequence, CpuReverseSequenceFp32KernelCreator)
}  // namespace mindspore::kernel
