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

#include "src/runtime/kernel/arm/fp32/softmax.h"
#include <string.h>
#include <vector>
#include "src/runtime/kernel/arm/opclib/fp32/softmax.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_SoftMax;

namespace mindspore::kernel {
int SoftmaxCPUKernel::Init() {
  auto input_tensor = inputs_.front();
  auto in_shape = input_tensor->shape();
  auto in_dims = in_shape.size();
  int ele_size = 1;
  (reinterpret_cast<SoftmaxParameter *>(opParameter))->n_dim_ = in_dims;
  for (size_t i = 0; i < in_dims; i++) {
    (reinterpret_cast<SoftmaxParameter *>(opParameter))->input_shape_[i] = in_shape[i];
    ele_size *= in_shape[i];
  }
  (reinterpret_cast<SoftmaxParameter *>(opParameter))->element_size_ = ele_size;

  // malloc tmp buffer
  auto axis = reinterpret_cast<SoftmaxParameter *>(opParameter)->axis_;
  sum_data = reinterpret_cast<float *>(malloc(in_shape[axis] * sizeof(float)));
  memset(sum_data, 0, in_shape[axis] * sizeof(float));
  return RET_OK;
}

int SoftmaxCPUKernel::ReSize() { return RET_OK; }

int SoftmaxCPUKernel::Run() {
  auto input_ptr = reinterpret_cast<float *>(inputs_.at(kInputIndex)->Data());
  auto output_ptr = reinterpret_cast<float *>(outputs_.at(kOutputIndex)->Data());
  Softmax(input_ptr, output_ptr, sum_data, reinterpret_cast<SoftmaxParameter *>(opParameter));
  return RET_OK;
}

kernel::LiteKernel *CpuSoftmaxFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                const std::vector<lite::tensor::Tensor *> &outputs,
                                                OpParameter *opParameter, const lite::Context *ctx,
                                                const kernel::KernelKey &desc) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_SoftMax);
  auto *kernel = new (std::nothrow) SoftmaxCPUKernel(opParameter, inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new SoftmaxCPUKernel fail!";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SoftMax, CpuSoftmaxFp32KernelCreator)
}  // namespace mindspore::kernel

