/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/cpu/base/format_transpose.h"
#include "nnacl/base/format_transpose.h"
#include "src/litert/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_FormatTranspose;

namespace mindspore::kernel {
int FormatTransposeCPUKernel::Run() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  auto input = in_tensors_.at(0);
  auto output = out_tensors_.at(0);
  CHECK_NULL_RETURN(input);
  CHECK_NULL_RETURN(output);
  auto input_ptr = in_tensors_[0]->data();
  auto output_ptr = out_tensors_[0]->data();
  CHECK_NULL_RETURN(input_ptr);
  CHECK_NULL_RETURN(output_ptr);

  int batch = input->Batch();
  int height = input->Height();
  int width = input->Width();
  int channel = input->Channel();

  auto data_type = input->data_type();
  CHECK_NULL_RETURN(param_);
  return TransData(input_ptr, output_ptr, param_->src_format_, param_->dst_format_, (TypeIdC)data_type, batch, channel,
                   height * width);
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_FormatTranspose, LiteKernelCreator<FormatTransposeCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_FormatTranspose, LiteKernelCreator<FormatTransposeCPUKernel>)
}  // namespace mindspore::kernel
