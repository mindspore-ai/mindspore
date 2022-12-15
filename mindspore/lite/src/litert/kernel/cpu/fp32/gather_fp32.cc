/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/cpu/fp32/gather_fp32.h"
#include <limits>
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Gather;

namespace mindspore::kernel {
int GatherCPUKernel::Run() {
  CHECK_NULL_RETURN(in_tensors_.at(FIRST_INPUT));
  CHECK_NULL_RETURN(in_tensors_.at(SECOND_INPUT));
  CHECK_NULL_RETURN(out_tensors_.at(FIRST_INPUT));
  return GatherBaseCPUKernel::Run();
}

int GatherCPUKernel::AssignIndicesData(bool isIndicesInt32) {
  auto indices_tensor = in_tensors_[SECOND_INPUT];
  auto indices_num = indices_tensor->ElementsNum();
  CHECK_NULL_RETURN(indices_tensor->data());
  if (!isIndicesInt32) {
    if (indices_num >= std::numeric_limits<int>::max() / static_cast<int>(sizeof(int))) {
      MS_LOG(ERROR) << "Input indices_num is invalid, indices_num: " << indices_num;
      return RET_ERROR;
    }
    indices_data_ = reinterpret_cast<int32_t *>(ms_context_->allocator->Malloc(sizeof(int32_t) * indices_num));
    if (indices_data_ == nullptr) {
      MS_LOG(ERROR) << "Memory allocation failed";
      return RET_ERROR;
    }
    switch (indices_tensor->data_type()) {
      case kNumberTypeInt64:
        for (int i = 0; i < indices_num; i++) {
          indices_data_[i] = static_cast<int>(reinterpret_cast<int64_t *>(indices_tensor->MutableData())[i]);
        }
        break;
      case kNumberTypeFloat:
      case kNumberTypeFloat32:
        for (int i = 0; i < indices_num; i++) {
          indices_data_[i] = static_cast<int>(reinterpret_cast<float *>(indices_tensor->MutableData())[i]);
        }
        break;
      case kNumberTypeBool:
        for (int i = 0; i < indices_num; i++) {
          indices_data_[i] = static_cast<int>(reinterpret_cast<bool *>(indices_tensor->MutableData())[i]);
        }
        break;
      default:
        MS_LOG(ERROR) << "Does not support data type: " << indices_tensor->data_type();
        return RET_ERROR;
    }
  } else {
    indices_data_ = reinterpret_cast<int32_t *>(indices_tensor->MutableData());
    CHECK_NULL_RETURN(indices_data_);
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Gather, LiteKernelCreator<GatherCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Gather, LiteKernelCreator<GatherCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Gather, LiteKernelCreator<GatherCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_Gather, LiteKernelCreator<GatherCPUKernel>)
}  // namespace mindspore::kernel
