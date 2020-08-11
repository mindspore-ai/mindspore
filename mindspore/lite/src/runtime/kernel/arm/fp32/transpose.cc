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

#include "src/runtime/kernel/arm/fp32/transpose.h"
#include <vector>
#include "src/runtime/kernel/arm/nnacl/transpose.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Transpose;

namespace mindspore::kernel {
namespace {
constexpr int kTransposeInputNum = 1;
constexpr int kTransposeOutputNum = 1;
}  // namespace
int TransposeCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int TransposeCPUKernel::ReSize() {
  auto &inTensor = in_tensors_.front();
  auto &outTensor = out_tensors_.front();
  auto param = reinterpret_cast<TransposeParameter *>(op_parameter_);
  auto in_shape = inTensor->shape();
  auto out_shape = outTensor->shape();
  param->strides_[param->num_axes_ - 1] = 1;
  param->out_strides_[param->num_axes_ - 1] = 1;
  param->data_size_ = inTensor->Size();
  for (int i = param->num_axes_ - 2; i >= 0; i--) {
    param->strides_[i] = in_shape[i + 1] * param->strides_[i + 1];
    param->out_strides_[i] = out_shape[i + 1] * param->out_strides_[i + 1];
  }
  return RET_OK;
}

int TransposeCPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << ret;
    return ret;
  }
  MS_ASSERT(in_tensors_.size() == TransposeInputNum);
  MS_ASSERT(out_tensors_.size() == TransposeOutputNum);
  auto &inTensor = in_tensors_.front();
  auto &outTensor = out_tensors_.front();
  if (inTensor == nullptr || outTensor == nullptr) {
    MS_LOG(ERROR) << "null pointer dreferencing.";
    return RET_ERROR;
  }
  auto *in_data = static_cast<float *>(inTensor->Data());
  auto *out_data = static_cast<float *>(outTensor->Data());
  auto in_shape = inTensor->shape();
  auto out_shape = outTensor->shape();
  auto *input_shape = &in_shape.front();
  auto *output_shape = &out_shape.front();

  ret =
    DoTranspose(in_data, out_data, input_shape, output_shape, reinterpret_cast<TransposeParameter *>(op_parameter_));
  return ret;
}

kernel::LiteKernel *CpuTransposeFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                  const std::vector<lite::tensor::Tensor *> &outputs,
                                                  OpParameter *opParameter, const lite::Context *ctx,
                                                  const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  MS_ASSERT(desc.type == schema::PrimitiveType_Transpose);
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "desc type is not Transpose";
    return nullptr;
  }
  auto *kernel = new (std::nothrow) TransposeCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "New kernel fails.";
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Transpose, CpuTransposeFp32KernelCreator)
}  // namespace mindspore::kernel
