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

#include "src/litert/kernel/cpu/fp32/range_fp32.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/fp32/range_fp32.h"
#ifdef ENABLE_FP16
#include "nnacl/fp16/range_fp16.h"
#endif

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Range;

namespace mindspore::kernel {
int RangeCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_[kInputIndex]);
  CHECK_NULL_RETURN(out_tensors_[kInputIndex]);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int RangeCPUKernel::ReSize() { return RET_OK; }

int RangeCPUKernel::Run() {
  TypeId data_type = in_tensors_[kInputIndex]->data_type();
  void *input_data = in_tensors_[kInputIndex]->data();
  CHECK_NULL_RETURN(input_data);
  void *output_data = out_tensors_[kOutputIndex]->data();
  CHECK_NULL_RETURN(output_data);
  int output_num = out_tensors_[kOutputIndex]->DimensionSize(FIRST_INPUT);
  if (in_tensors_.size() == C3NUM) {
    CHECK_NULL_RETURN(in_tensors_[THIRD_INPUT]);
    void *delta_data = in_tensors_[THIRD_INPUT]->data();
    CHECK_NULL_RETURN(delta_data);
    if (data_type == kNumberTypeFloat32) {
      Range(static_cast<float *>(output_data), *static_cast<float *>(input_data), *static_cast<float *>(delta_data),
            output_num);
    } else if (data_type == kNumberTypeFloat16) {
#ifdef ENABLE_FP16
      RangeFp16(static_cast<float16_t *>(output_data), *static_cast<float16_t *>(input_data),
                *static_cast<float16_t *>(delta_data), output_num);
#endif
    } else {
      RangeInt(static_cast<int *>(output_data), *static_cast<int *>(input_data), *static_cast<int *>(delta_data),
               output_num);
    }
  } else {
    if (data_type == kNumberTypeInt32) {
      RangeInt(static_cast<int *>(output_data), (reinterpret_cast<RangeParameter *>(op_parameter_))->start_,
               (reinterpret_cast<RangeParameter *>(op_parameter_))->delta_, output_num);
    } else {
      MS_LOG(ERROR) << "Unsupported data type : " << data_type;
      return RET_ERROR;
    }
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Range, LiteKernelCreator<RangeCPUKernel>)
#ifdef ENABLE_FP16
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Range, LiteKernelCreator<RangeCPUKernel>)
#endif
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Range, LiteKernelCreator<RangeCPUKernel>)
}  // namespace mindspore::kernel
