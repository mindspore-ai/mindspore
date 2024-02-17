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
#include "src/litert/kernel/cpu/base/custom_is_inf.h"
#include "src/common/tensor_util.h"
#include "nnacl/op_base.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {

int CustomIsInfCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C1NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), C1NUM);
  return RET_OK;
}

int CustomIsInfCPUKernel::ReSize() { return RET_OK; }

void CustomIsInfCPUKernel::LaunchKernelFloat(const float *input, bool *output) {
  auto elem_num = in_tensors_[FIRST_INPUT]->ElementsNum();

  for (int i = 0; i < elem_num; i++) {
    output[i] = std::isinf(input[i]);
  }
}

int CustomIsInfCPUKernel::Run() {
  auto input = in_tensors_[FIRST_INPUT];
  auto output = out_tensors_[FIRST_INPUT];
  CHECK_NULL_RETURN(input);
  CHECK_NULL_RETURN(output);

  if (input->data_type() == kNumberTypeFloat32 || input->data_type() == kNumberTypeFloat) {
    LaunchKernelFloat(reinterpret_cast<const float *>(input->data()), reinterpret_cast<bool *>(output->data()));
  } else {
    MS_LOG(ERROR) << "unsupported input data type " << input->data_type();
    return RET_ERROR;
  }

  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimType_Inner_CustomIsInf, LiteKernelCreator<CustomIsInfCPUKernel>)
}  // namespace mindspore::kernel
