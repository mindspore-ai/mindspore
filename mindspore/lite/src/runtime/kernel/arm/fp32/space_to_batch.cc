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
#include "src/runtime/kernel/arm/opclib/fp32/space_to_batch.h"
#include "src/runtime/kernel/arm/opclib/errorcode.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_FORMAT_ERR;
using mindspore::lite::RET_OK;
using mindspore::lite::RET_OP_EXECUTE_FAILURE;
using mindspore::schema::PrimitiveType_SpaceToBatch;

namespace mindspore::kernel {

int SpaceToBatchCPUKernel::Init() {
  if (inputs_[0]->GetFormat() != schema::Format_NHWC) {
    MS_LOG(ERROR) << "space_to_batch only support NHWC now!";
    return RET_FORMAT_ERR;
  }
  SpaceToBatchParameter *param = reinterpret_cast<SpaceToBatchParameter *>(this->opParameter);
  for (int i = 0; i < SPACE_TO_BATCH_PADDINGS_SIZE; ++i) {
    if (param->paddings_[i] != 0) {
      param->need_paddings_ = true;
      break;
    }
  }
  param->n_dims_ = DIMENSION_4D;
  param->n_space_dims_ = SPACE_TO_BATCH_BLOCK_SIZES_SIZE;
  param->num_elements_ = EnumElement(param->in_shape_, param->n_dims_);
  param->num_elements_padded_ = EnumElement(param->padded_in_shape_, param->n_dims_);
  return RET_OK;
}

int SpaceToBatchCPUKernel::Run() {
  auto input = inputs_[0];
  auto output = outputs_[0];
  input_ptr_ = reinterpret_cast<const float *>(input->Data());
  output_ptr_ = reinterpret_cast<float *>(output->Data());
  SpaceToBatchParameter *param = reinterpret_cast<SpaceToBatchParameter *>(this->opParameter);

  int ret;
  float *tmp_space[3] = {nullptr, nullptr, nullptr};
  if (param->need_paddings_) {
    tmp_space[0] = reinterpret_cast<float *>(malloc(param->num_elements_padded_ * sizeof(float)));
    (void)memset(tmp_space[0], 0, param->num_elements_padded_);
    tmp_space[1] = reinterpret_cast<float *>(malloc(param->num_elements_padded_ * sizeof(float)));
    (void)memset(tmp_space[1], 0, param->num_elements_padded_);
    tmp_space[2] = reinterpret_cast<float *>(malloc(param->num_elements_padded_ * sizeof(float)));
    (void)memset(tmp_space[2], 0, param->num_elements_padded_);

    ret = SpaceToBatch(input_ptr_, output_ptr_, *param, tmp_space);
  } else {
    ret = SpaceToBatch(input_ptr_, output_ptr_, *param, tmp_space);
  }
  if (ret != OPCLIB_OK) {
    MS_LOG(ERROR) << "Do space to batch fails!";
    return RET_OP_EXECUTE_FAILURE;
  }

  return RET_OK;
}

kernel::LiteKernel *CpuSpaceToBatchFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                     const std::vector<lite::tensor::Tensor *> &outputs,
                                                     OpParameter *opParameter, const lite::Context *ctx,
                                                     const kernel::KernelKey &desc) {
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Input opParameter is nullptr!";
    return nullptr;
  }
  auto *kernel = new (std::nothrow) SpaceToBatchCPUKernel(opParameter, inputs, outputs);
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
