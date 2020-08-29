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
#include "src/runtime/kernel/arm/fp32/space_to_batch.h"
#include <vector>
#include "schema/ops_generated.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "nnacl/fp32/space_to_batch.h"
#include "nnacl/errorcode.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_FORMAT_ERR;
using mindspore::lite::RET_OK;
using mindspore::lite::RET_OP_EXECUTE_FAILURE;
using mindspore::schema::PrimitiveType_SpaceToBatch;
using mindspore::schema::PrimitiveType_SpaceToBatchND;

namespace mindspore::kernel {
namespace {
size_t EnumElement(int *shape, int n_dims) {
  size_t total = 1;
  for (int i = 0; i < n_dims; i++) {
    total *= shape[i];
  }
  return total;
}
}

int SpaceToBatchCPUKernel::Init() {
  SpaceToBatchParameter *param = reinterpret_cast<SpaceToBatchParameter *>(this->op_parameter_);
  for (int i = 0; i < SPACE_TO_BATCH_PADDINGS_SIZE; ++i) {
    if (param->paddings_[i] != 0) {
      param->need_paddings_ = true;
      break;
    }
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

void SpaceToBatchCPUKernel::FreeTmpBuffer() {
  if (pedding_h_data_ != nullptr) {
    context_->allocator->Free(pedding_h_data_);
    pedding_h_data_ = nullptr;
  }
  if (pedding_w_data_ != nullptr) {
    context_->allocator->Free(pedding_w_data_);
    pedding_w_data_ = nullptr;
  }
  if (pedding_input_ != nullptr) {
    context_->allocator->Free(pedding_input_);
    pedding_input_ = nullptr;
  }
}

int SpaceToBatchCPUKernel::ReSize() {
  if (in_tensors_[0]->GetFormat() != schema::Format_NHWC) {
    MS_LOG(ERROR) << "space_to_batch only support NHWC now!";
    return RET_FORMAT_ERR;
  }
  FreeTmpBuffer();
  SpaceToBatchParameter *param = reinterpret_cast<SpaceToBatchParameter *>(this->op_parameter_);
  if (!param->need_paddings_) {
    return RET_OK;
  }
  auto input = in_tensors_[0];
  auto in_shape = input->shape();
  padded_in_shape_ = in_shape;
  padded_in_shape_[1] = in_shape[1] + param->paddings_[0] + param->paddings_[1];
  padded_in_shape_[2] = in_shape[2] + param->paddings_[2] + param->paddings_[3];
  auto num_elements_padded = EnumElement(padded_in_shape_.data(), in_shape.size());
  auto output_shape = out_tensors_[0]->shape();
  auto pedding_h_size = output_shape[2] * output_shape[3] * sizeof(float);
  pedding_h_data_ = reinterpret_cast<float *>(context_->allocator->Malloc(pedding_h_size));
  if (pedding_h_data_ == nullptr) {
    MS_LOG(ERROR) << "malloc pedding h data fail!";
    return RET_ERROR;
  }
  auto pedding_w_size = output_shape[3] * sizeof(float);
  pedding_w_data_ = reinterpret_cast<float *>(context_->allocator->Malloc(pedding_w_size));
  if (pedding_w_data_ == nullptr) {
    MS_LOG(ERROR) << "malloc pedding w data fail!";
    FreeTmpBuffer();
    return RET_ERROR;
  }
  pedding_input_ =
      reinterpret_cast<float *>(context_->allocator->Malloc(num_elements_padded * sizeof(float)));
  if (pedding_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc pedding buffer fail!";
    return RET_ERROR;
  }
  memset(pedding_h_data_, 0, pedding_h_size);
  memset(pedding_w_data_, 0, pedding_w_size);
  return RET_OK;
}

int SpaceToBatchCPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << ret;
    return ret;
  }
  auto input = in_tensors_[0];
  auto output = out_tensors_[0];
  const float *input_ptr_ = reinterpret_cast<const float *>(input->Data());
  float *output_ptr_ = reinterpret_cast<float *>(output->Data());
  SpaceToBatchParameter *param = reinterpret_cast<SpaceToBatchParameter *>(this->op_parameter_);
  auto in_shape = input->shape();
  auto out_shape = output->shape();
  if (param->need_paddings_) {
    DoSpaceToBatchPaddingNHWC(input_ptr_, pedding_input_, in_shape.data(), param->paddings_,
                              padded_in_shape_.data(), pedding_h_data_, pedding_w_data_);
    DoSpaceToBatchNHWC(pedding_input_, output_ptr_, param, padded_in_shape_.data(), out_shape.data());
    return RET_OK;
  } else {
    DoSpaceToBatchNHWC(input_ptr_, output_ptr_, param, in_shape.data(), out_shape.data());
    return RET_OK;
  }
}  // namespace mindspore::kernel

kernel::LiteKernel *CpuSpaceToBatchFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                     const std::vector<lite::tensor::Tensor *> &outputs,
                                                     OpParameter *param, const lite::Context *ctx,
                                                     const kernel::KernelKey &desc,
                                                     const mindspore::lite::PrimitiveC *primitive) {
  if (param == nullptr) {
    MS_LOG(ERROR) << "Input param is nullptr!";
    return nullptr;
  }
  auto *kernel = new (std::nothrow) SpaceToBatchCPUKernel(param, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new SpaceToBatchCPUKernel fail!";
    return nullptr;
  }

  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    MS_LOG(ERROR) << "Init kernel failed, name: " << param->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(param->type_));
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SpaceToBatch, CpuSpaceToBatchFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SpaceToBatchND, CpuSpaceToBatchFp32KernelCreator)
}  // namespace mindspore::kernel
