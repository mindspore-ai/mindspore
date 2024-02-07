/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"
#include "src/litert/kernel/cpu/base/custom_masked_fill.h"
#include "src/common/tensor_util.h"
#include "nnacl/op_base.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {

int CustomMaskedFillCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C3NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), C1NUM);

  // only support input value as a single float value
  MS_CHECK_TRUE_MSG(in_tensors_[FIRST_INPUT]->data_type() == mindspore::TypeId::kNumberTypeFloat32 ||
                      in_tensors_[FIRST_INPUT]->data_type() == mindspore::TypeId::kNumberTypeFloat,
                    RET_ERROR, "input dtype must be float32");
  if (in_tensors_[THIRD_INPUT]->ElementsNum() != 1) {
    MS_LOG(ERROR) << "only support fill value as a single float";
    return RET_ERROR;
  }
  MS_CHECK_TRUE_MSG(in_tensors_[SECOND_INPUT]->data_type() == mindspore::TypeId::kNumberTypeBool, RET_ERROR,
                    "mask dtype must be bool");
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int CustomMaskedFillCPUKernel::ReSize() { return RET_OK; }

int CustomMaskedFillCPUKernel::Run() {
  auto input = in_tensors_[FIRST_INPUT];
  auto mask = in_tensors_[SECOND_INPUT];
  auto value = in_tensors_[THIRD_INPUT];
  auto output = out_tensors_[FIRST_INPUT];
  CHECK_NULL_RETURN(input);
  CHECK_NULL_RETURN(mask);
  CHECK_NULL_RETURN(value);
  CHECK_NULL_RETURN(output);

  if (input->shape() != mask->shape()) {
    MS_LOG(ERROR) << "Not support broadcast mask to input";
    return RET_ERROR;
  }

  auto value_data = reinterpret_cast<float *>(value->data());
  auto fill_value = value_data[0];

  auto data_num = input->ElementsNum();
  auto input_data = reinterpret_cast<float *>(input->data());
  auto mask_data = reinterpret_cast<bool *>(mask->data());
  auto output_data = reinterpret_cast<float *>(output->data());
  for (int64_t i = 0; i < data_num; i++) {
    if (mask_data[i]) {
      output_data[i] = fill_value;
    } else {
      output_data[i] = input_data[i];
    }
  }

  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimType_Inner_CustomMaskedFill, LiteKernelCreator<CustomMaskedFillCPUKernel>)
}  // namespace mindspore::kernel
