/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/cpu/fp32/unique_fp32.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/fp32/unique_fp32.h"
#ifdef ENABLE_FP16
#include "nnacl/fp16/unique_fp16.h"
#endif

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Unique;

namespace mindspore::kernel {
int UniqueCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), C2NUM);
  CHECK_NULL_RETURN(in_tensors_[0]);
  CHECK_NULL_RETURN(out_tensors_[0]);
  CHECK_NULL_RETURN(out_tensors_[1]);
  return RET_OK;
}

int UniqueCPUKernel::ReSize() { return RET_OK; }

int UniqueCPUKernel::Run() {
  auto input = in_tensors_[0]->MutableData();
  CHECK_NULL_RETURN(input);
  auto output0 = out_tensors_[0]->MutableData();
  CHECK_NULL_RETURN(output0);
  auto output1 = reinterpret_cast<int *>(out_tensors_[1]->MutableData());
  CHECK_NULL_RETURN(output1);

  int output0_len = 0;
  if (in_tensors_[0]->data_type() == kNumberTypeFloat16) {
#ifdef ENABLE_FP16
    UniqueFp16(static_cast<float16_t *>(input), in_tensors_[0]->ElementsNum(), static_cast<float16_t *>(output0),
               &output0_len, output1);
#endif
  } else if (in_tensors_[0]->data_type() == kNumberTypeInt32) {
    UniqueInt(static_cast<int *>(input), in_tensors_[0]->ElementsNum(), static_cast<int *>(output0), &output0_len,
              output1);
  } else {
    Unique(static_cast<float *>(input), in_tensors_[0]->ElementsNum(), static_cast<float *>(output0), &output0_len,
           output1);
  }

  std::vector<int> out_shape = out_tensors_[0]->shape();
  out_tensors_[0]->set_shape_changed(out_shape.back() != output0_len);
  out_shape[out_shape.size() - 1] = output0_len;
  out_tensors_[0]->set_shape(out_shape);
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Unique, LiteKernelCreator<UniqueCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Unique, LiteKernelCreator<UniqueCPUKernel>)
#ifdef ENABLE_FP16
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Unique, LiteKernelCreator<UniqueCPUKernel>)
#endif
}  // namespace mindspore::kernel
