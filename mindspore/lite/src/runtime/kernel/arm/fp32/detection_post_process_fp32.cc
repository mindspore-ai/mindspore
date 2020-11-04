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
  if ((in_tensors_.at(0)->data_type() != kNumberTypeFloat32 && in_tensors_.at(0)->data_type() != kNumberTypeFloat) ||
      (in_tensors_.at(1)->data_type() != kNumberTypeFloat32 && in_tensors_.at(1)->data_type() != kNumberTypeFloat)) {
    MS_LOG(ERROR) << "Input data type error";
    return RET_ERROR;
  }
  input_boxes_ = reinterpret_cast<float *>(in_tensors_.at(0)->MutableData());
  input_scores_ = reinterpret_cast<float *>(in_tensors_.at(1)->MutableData());
  return RET_OK;
}

kernel::LiteKernel *CpuDetectionPostProcessFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                             const std::vector<lite::Tensor *> &outputs,
                                                             OpParameter *opParameter, const lite::InnerContext *ctx,
                                                             const kernel::KernelKey &desc,
                                                             const mindspore::lite::PrimitiveC *primitive) {
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Create kernel failed, opParameter is nullptr, type: PrimitiveType_DetectionPostProcess. ";
    return nullptr;
  }
  MS_ASSERT(desc.type == schema::PrimitiveType_DetectionPostProcess);
  auto *kernel = new (std::nothrow) DetectionPostProcessCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new DetectionPostProcessCPUKernel fail!";
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_DetectionPostProcess, CpuDetectionPostProcessFp32KernelCreator)
}  // namespace mindspore::kernel
