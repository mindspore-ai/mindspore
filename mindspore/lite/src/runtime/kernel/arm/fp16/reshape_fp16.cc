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

#include "src/runtime/kernel/arm/fp16/reshape_fp16.h"
#include <vector>
#include "nnacl/fp16/cast_fp16.h"
#include "nnacl/reshape.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Reshape;

namespace mindspore::kernel {

int ReshapeFp16CPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << ret;
    return ret;
  }
  auto in_tensor = in_tensors_.at(kInputIndex);
  auto out_tensor = out_tensors_.at(kOutputIndex);
  auto input_ptr = in_tensor->Data();
  auto output_ptr = out_tensor->Data();
  size_t data_size = out_tensor->Size();

  auto in_datatype = in_tensor->data_type();
  auto out_datatype = out_tensor->data_type();
  if (in_datatype != out_datatype) {
    if (in_datatype == kNumberTypeFloat32 && out_datatype == kNumberTypeFloat16) {
      input_ptr = context_->allocator->Malloc(in_tensor->ElementsNum() * sizeof(float16_t));
      if (input_ptr == nullptr) {
        MS_LOG(ERROR) << "malloc in tensor fail!";
        return mindspore::lite::RET_MEMORY_FAILED;
      }
      Float32ToFloat16(reinterpret_cast<float *>(in_tensor->Data()), reinterpret_cast<float16_t *>(input_ptr),
                       in_tensor->ElementsNum());
    } else if ((in_datatype == kNumberTypeFloat16 && out_datatype == kNumberTypeFloat32)) {
      input_ptr = context_->allocator->Malloc(in_tensor->ElementsNum() * sizeof(float));
      if (input_ptr == nullptr) {
        MS_LOG(ERROR) << "malloc in tensor fail!";
        return mindspore::lite::RET_MEMORY_FAILED;
      }
      Float16ToFloat32(reinterpret_cast<float16_t *>(in_tensor->Data()), reinterpret_cast<float *>(input_ptr),
                       in_tensor->ElementsNum());
    } else {
      MS_LOG(ERROR) << "unsupported data type, in_datatype: " << in_datatype << ",out_datatype: " << out_datatype;
      return RET_ERROR;
    }
  }

  Reshape(input_ptr, output_ptr, data_size);
  if (in_datatype != out_datatype) {
    context_->allocator->Free(input_ptr);
  }
  return RET_OK;
}

kernel::LiteKernel *CpuReshapeFp16KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                const std::vector<lite::tensor::Tensor *> &outputs,
                                                OpParameter *opParameter, const Context *ctx,
                                                const kernel::KernelKey &desc,
                                                const mindspore::lite::PrimitiveC *primitive) {
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Input opParameter is nullptr!";
    return nullptr;
  }
  MS_ASSERT(desc.type == schema::PrimitiveType_Reshape);
  auto *kernel = new (std::nothrow) ReshapeFp16CPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new ReshapeFp16CPUKernel fail!";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Reshape, CpuReshapeFp16KernelCreator)
}  // namespace mindspore::kernel
