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

#include "src/runtime/kernel/arm/fp32/unique_fp32.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Unique;

namespace mindspore::kernel {
int UniqueCPUKernel::Init() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), C2NUM);
  return RET_OK;
}

int UniqueCPUKernel::ReSize() { return RET_OK; }

int UniqueCPUKernel::Run() {
  auto input = reinterpret_cast<float *>(in_tensors_.at(0)->MutableData());
  CHECK_NULL_RETURN(input);
  auto output0 = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());
  CHECK_NULL_RETURN(output0);
  auto output1 = reinterpret_cast<int *>(out_tensors_.at(1)->MutableData());
  CHECK_NULL_RETURN(output1);

  int output0_len = 0;
  Unique(input, in_tensors_.at(0)->ElementsNum(), output0, &output0_len, output1);

  std::vector<int> out_shape = out_tensors_.at(0)->shape();
  out_shape.at(out_shape.size() - 1) = output0_len;
  out_tensors_.at(0)->set_shape(out_shape);
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Unique, LiteKernelCreator<UniqueCPUKernel>)
}  // namespace mindspore::kernel
