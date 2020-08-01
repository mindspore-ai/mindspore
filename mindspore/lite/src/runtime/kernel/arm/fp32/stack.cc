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
#include "src/runtime/kernel/arm/fp32/stack.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/opclib/fp32/stack.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Stack;

namespace mindspore::kernel {
int StackCPUKernel::Init() {
  StackParameter *param = reinterpret_cast<StackParameter *>(opParameter);
  auto input0_shape = inputs_[0]->shape();
  axis_ = param->axis_ < 0 ? param->axis_ + input0_shape.size() : param->axis_;
  schema::Format input0_format = inputs_[0]->GetFormat();
  bool need_convert_format = false;
  for (size_t i = 1; i < inputs_.size(); ++i) {
    if (inputs_[i]->GetFormat() != input0_format) {
      need_convert_format = true;
    }
  }
  if (!need_convert_format) {
    outputs_[0]->SetFormat(input0_format);
    return RET_OK;
  }

  for (size_t i = 0; i < inputs_.size(); ++i) {
    if (inputs_[i]->GetFormat() != schema::Format_NHWC) {
      convert_functions_[i] = LayoutTransform(inputs_[i]->data_type(), inputs_[i]->GetFormat(), schema::Format_NHWC);
      if (convert_functions_[i] == nullptr) {
        MS_LOG(ERROR) << "Can not convert format " << inputs_[i]->GetFormat() << " to " << schema::Format_NHWC;
        return RET_ERROR;
      }
      size_t packed_input_size =
        inputs_[i]->Channel() * inputs_[i]->Batch() * inputs_[i]->Height() * inputs_[i]->Width();
      packed_inputs_[i] = reinterpret_cast<float *>(malloc(packed_input_size * sizeof(float)));
      if (packed_inputs_[i] == nullptr) {
        MS_LOG(ERROR) << "malloc memory fail!";
        return RET_ERROR;
      }
      memset(packed_inputs_[i], 0, packed_input_size * sizeof(float));
    } else {
      convert_functions_[i] = nullptr;
      packed_inputs_[i] = nullptr;
    }
  }
  outputs_[0]->SetFormat(schema::Format_NHWC);
  return RET_OK;
}

int StackCPUKernel::Run() {
  size_t inputs_num = inputs_.size();
  auto input0_shape = inputs_[0]->shape();
  auto *output_data = reinterpret_cast<float *>(outputs_[0]->Data());
  float *inputs[inputs_num];
  for (size_t i = 0; i < inputs_num; ++i) {
    inputs[i] = reinterpret_cast<float *>(inputs_[i]->Data());
    if (convert_functions_[i] != nullptr) {
      convert_functions_[i](inputs[i], packed_inputs_[i], inputs_[i]->Batch(),
                            inputs_[i]->Height() * inputs_[i]->Width(), inputs_[i]->Channel());
    } else {
      packed_inputs_[i] = inputs[i];
    }
  }
  DoStack(packed_inputs_.data(), inputs_num, input0_shape.data(), input0_shape.size(), axis_, output_data);
  return RET_OK;
}

kernel::LiteKernel *CpuStackFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                              const std::vector<lite::tensor::Tensor *> &outputs,
                                              OpParameter *op_parameter, const lite::Context *ctx,
                                              const kernel::KernelKey &desc) {
  if (op_parameter == nullptr) {
    MS_LOG(ERROR) << "Input op_parameter is nullptr!";
    return nullptr;
  }
  MS_ASSERT(desc.type == schema::PrimitiveType_Stack);
  auto *kernel = new (std::nothrow) StackCPUKernel(op_parameter, inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new StackCPUKernel fail!";
    return nullptr;
  }

  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    MS_LOG(ERROR) << "Init kernel failed, name: " << op_parameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(op_parameter->type_));
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Stack, CpuStackFp32KernelCreator)
}  // namespace mindspore::kernel
