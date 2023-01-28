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
#include "src/litert/kernel/cpu/fp32/broadcast_to_fp32.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_BroadcastTo;

namespace mindspore::kernel {
int BroadcastToCPUKernel::ReSize() {
  auto input_shape = in_tensors_.at(0)->shape();
  for (size_t i = 0; i < input_shape.size(); ++i) {
    shape_info_.input_shape_[i] = input_shape[i];
  }
  auto output_shape = out_tensors_.at(0)->shape();
  for (size_t i = 0; i < output_shape.size(); ++i) {
    shape_info_.output_shape_[i] = output_shape[i];
  }
  shape_info_.input_shape_size_ = static_cast<int>(input_shape.size());
  shape_info_.output_shape_size_ = static_cast<int>(output_shape.size());

  return RET_OK;
}

int BroadcastToCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int BroadcastToCPUKernel::Run() {
  const auto input_data = in_tensors_.at(0)->data();
  auto output_data = out_tensors_.at(0)->data();
  CHECK_NULL_RETURN(input_data);
  CHECK_NULL_RETURN(output_data);

  auto data_type = in_tensors_.at(0)->data_type();
  switch (data_type) {
    case kNumberTypeFloat32:
      return BroadcastToSize32(input_data, &shape_info_, output_data);
    case kNumberTypeFloat16:
      return BroadcastToSize16(input_data, &shape_info_, output_data);
    case kNumberTypeInt32:
    case kNumberTypeInt:
      return BroadcastToSize32(input_data, &shape_info_, output_data);
    case kNumberTypeBool:
      return BroadcastToSize8(input_data, &shape_info_, output_data);
    default:
      MS_LOG(ERROR) << "UnSupported data type: " << data_type;
      return RET_ERROR;
  }
}

REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_BroadcastTo, LiteKernelCreator<BroadcastToCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_BroadcastTo, LiteKernelCreator<BroadcastToCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_BroadcastTo, LiteKernelCreator<BroadcastToCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_BroadcastTo, LiteKernelCreator<BroadcastToCPUKernel>)
}  // namespace mindspore::kernel
