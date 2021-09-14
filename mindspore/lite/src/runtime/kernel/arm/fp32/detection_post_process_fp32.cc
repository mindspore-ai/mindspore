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
#include "src/runtime/kernel/arm/fp32/detection_post_process_fp32.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/int8/quant_dtype_cast_int8.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_DetectionPostProcess;

namespace mindspore::kernel {
int DetectionPostProcessCPUKernel::GetInputData() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  if ((in_tensors_.at(0)->data_type() != kNumberTypeFloat32 && in_tensors_.at(0)->data_type() != kNumberTypeFloat) ||
      (in_tensors_.at(1)->data_type() != kNumberTypeFloat32 && in_tensors_.at(1)->data_type() != kNumberTypeFloat)) {
    MS_LOG(ERROR) << "Input data type error";
    return RET_ERROR;
  }
  CHECK_NULL_RETURN(in_tensors_.at(0)->data());
  CHECK_NULL_RETURN(in_tensors_.at(1)->data());
  input_boxes_ = reinterpret_cast<float *>(in_tensors_.at(0)->data());
  input_scores_ = reinterpret_cast<float *>(in_tensors_.at(1)->data());
  return RET_OK;
}
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_DetectionPostProcess,
           LiteKernelCreator<DetectionPostProcessCPUKernel>)
}  // namespace mindspore::kernel
