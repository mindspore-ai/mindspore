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
#include "src/runtime/kernel/arm/fp32/broadcast_to.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_BroadcastTo;

namespace mindspore::kernel {
int BroadcastToCPUKernel::ReSize() {
  auto input_shape = in_tensors_[0]->shape();
  for (size_t i = 0; i < input_shape.size(); ++i) {
    shape_info_.input_shape_[i] = input_shape[i];
  }

  shape_info_.input_shape_size_ = static_cast<int>(input_shape.size());
  auto output_shape = out_tensors_[0]->shape();
  for (size_t i = 0; i < output_shape.size(); ++i) {
    shape_info_.output_shape_[i] = output_shape[i];
  }
  shape_info_.output_shape_size_ = static_cast<int>(output_shape.size());
  return RET_OK;
}

int BroadcastToCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }

  return ReSize();
}

int BroadcastToCPUKernel::Run() {
  auto prepare_ret = Prepare();
  if (prepare_ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << prepare_ret;
    return prepare_ret;
  }
  auto input_data = reinterpret_cast<float *>(in_tensors_.at(0)->Data());
  auto output_data = reinterpret_cast<float *>(out_tensors_.at(0)->Data());

  return BroadcastTo(input_data, &shape_info_, output_data);
}

kernel::LiteKernel *CpuBroadcastToFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                    const std::vector<lite::tensor::Tensor *> &outputs,
                                                    OpParameter *op_parameter, const lite::Context *ctx,
                                                    const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  if (op_parameter == nullptr) {
    MS_LOG(ERROR) << "Input op_parameter is nullptr!";
    return nullptr;
  }
  MS_ASSERT(desc.type == schema::PrimitiveType_BroadcastTo);
  auto *kernel = new (std::nothrow) BroadcastToCPUKernel(op_parameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new BroadcastToCPUKernel fail!";
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_BroadcastTo, CpuBroadcastToFp32KernelCreator)
}  // namespace mindspore::kernel
