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
#include "src/runtime/kernel/arm/nnacl/fp32/space_to_batch.h"
#include "src/runtime/kernel/arm/nnacl/errorcode.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_FORMAT_ERR;
using mindspore::lite::RET_OK;
using mindspore::lite::RET_OP_EXECUTE_FAILURE;
using mindspore::schema::PrimitiveType_SpaceToBatch;

namespace mindspore::kernel {

int SpaceToBatchCPUKernel::Init() {
  SpaceToBatchParameter *param = reinterpret_cast<SpaceToBatchParameter *>(this->op_parameter_);
  for (int i = 0; i < SPACE_TO_BATCH_PADDINGS_SIZE; ++i) {
    if (param->paddings_[i] != 0) {
      param->need_paddings_ = true;
      break;
    }
  }
  param->n_dims_ = DIMENSION_4D;
  param->n_space_dims_ = SPACE_TO_BATCH_BLOCK_SIZES_SIZE;
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int SpaceToBatchCPUKernel::ReSize() {
  if (in_tensors_[0]->GetFormat() != schema::Format_NHWC) {
    MS_LOG(ERROR) << "space_to_batch only support NHWC now!";
    return RET_FORMAT_ERR;
  }
  SpaceToBatchParameter *param = reinterpret_cast<SpaceToBatchParameter *>(this->op_parameter_);
  param->num_elements_ = EnumElement(param->in_shape_, param->n_dims_);
  param->num_elements_padded_ = EnumElement(param->padded_in_shape_, param->n_dims_);
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
  input_ptr_ = reinterpret_cast<const float *>(input->Data());
  output_ptr_ = reinterpret_cast<float *>(output->Data());
  SpaceToBatchParameter *param = reinterpret_cast<SpaceToBatchParameter *>(this->op_parameter_);

  float *tmp_space[3] = {nullptr, nullptr, nullptr};
  if (param->need_paddings_) {
    for (int i = 0; i < 3; ++i) {
      tmp_space[i] =
        reinterpret_cast<float *>(context_->allocator->Malloc(param->num_elements_padded_ * sizeof(float)));
      (void)memset(tmp_space[i], 0, param->num_elements_padded_ * sizeof(float));
      if (tmp_space[i] == nullptr) {
        MS_LOG(ERROR) << "malloc tmp buffer fail!";
        return RET_ERROR;
      }
    }
    ret = SpaceToBatch(input_ptr_, output_ptr_, *param, tmp_space);
    for (int i = 0; i < 3; ++i) {
      context_->allocator->Free(tmp_space);
    }
  } else {
    ret = SpaceToBatch(input_ptr_, output_ptr_, *param, tmp_space);
  }
  if (ret != NNACL_OK) {
    MS_LOG(ERROR) << "Do space to batch fails!";
    return RET_OP_EXECUTE_FAILURE;
  }

  return RET_OK;
}

kernel::LiteKernel *CpuSpaceToBatchFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                     const std::vector<lite::tensor::Tensor *> &outputs,
                                                     OpParameter *opParameter, const lite::Context *ctx,
                                                     const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Input opParameter is nullptr!";
    return nullptr;
  }
  auto *kernel = new (std::nothrow) SpaceToBatchCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new SpaceToBatchCPUKernel fail!";
    return nullptr;
  }

  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SpaceToBatch, CpuSpaceToBatchFp32KernelCreator)
}  // namespace mindspore::kernel
