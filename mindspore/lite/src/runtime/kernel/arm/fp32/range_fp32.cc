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

#include "src/runtime/kernel/arm/fp32/range_fp32.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Range;

namespace mindspore::kernel {
int RangeCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int RangeCPUKernel::ReSize() {
  if (in_tensors_[0]->data_type() == kNumberTypeFloat32 || in_tensors_[0]->data_type() == kNumberTypeFloat16 ||
      in_tensors_[0]->data_type() == kNumberTypeFloat) {
    data_type_ = kDataTypeFloat;
  } else {
    data_type_ = kDataTypeInt;
  }
  return RET_OK;
}

int RangeCPUKernel::Run() {
  if (in_tensors_.size() == 3) {
    if (data_type_ == kDataTypeInt) {
      RangeInt(reinterpret_cast<int *>(out_tensors_.at(0)->data_c()),
               *reinterpret_cast<int *>(in_tensors_.at(0)->data_c()),
               *reinterpret_cast<int *>(in_tensors_.at(2)->data_c()), out_tensors_.at(0)->shape()[0]);
    } else {
      Range(reinterpret_cast<float *>(out_tensors_.at(0)->data_c()),
            *reinterpret_cast<float *>(in_tensors_.at(0)->data_c()),
            *reinterpret_cast<float *>(in_tensors_.at(2)->data_c()), out_tensors_.at(0)->shape()[0]);
    }
  } else {
    if (data_type_ == kDataTypeInt) {
      RangeInt(reinterpret_cast<int *>(out_tensors_.at(0)->data_c()),
               (reinterpret_cast<RangeParameter *>(op_parameter_))->start_,
               (reinterpret_cast<RangeParameter *>(op_parameter_))->delta_, out_tensors_.at(0)->shape()[0]);
    } else {
      MS_LOG(ERROR) << "Unsupported parameter type : " << in_tensors_.at(0)->data_type() << ".";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

kernel::LiteKernel *CpuRangeFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                              const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                              const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                              const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Range);

  auto *kernel = new (std::nothrow) RangeCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new RangeCPUKernel fail!";
    free(opParameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Range, CpuRangeFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat, PrimitiveType_Range, CpuRangeFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Range, CpuRangeFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt, PrimitiveType_Range, CpuRangeFp32KernelCreator)
}  // namespace mindspore::kernel
