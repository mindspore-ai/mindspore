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
#include "include/errorcode.h"
#include "include/ms_tensor.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/fp32/TensorListGetItem.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_TensorListGetItem;

namespace mindspore::kernel {

int TensorListGetItemCPUKernel::Init() {
  auto input0 = reinterpret_cast<int *>(in_tensors_[0]->data_c());
  size_t dim0 = *input0;
  int in_dtype = *(input0 + 1);
  if (dtype_ != in_dtype) {
    MS_LOG(ERROR) << "op dtype:" << dtype_ << " is not equal in_tensors dtype:" << in_dtype;
    return RET_ERROR;
  }
  index_ = *(reinterpret_cast<int *>(in_tensors_[dim0 + 2]->data_c()));
  if (index_ < 0) {
    MS_LOG(ERROR) << "index tensor:[" << index_ << "] must be greater than or equal to 0";
    return RET_ERROR;
  }
  if (index_ > dim0) {
    MS_LOG(ERROR) << "index tensor:[" << index_ << "] must be less than dim0:" << dim0;
    return RET_ERROR;
  }
  index_ += 2;
  return RET_OK;
}

int TensorListGetItemCPUKernel::Run() {
  if (in_tensors_[index_]->data_type() != kTypeUnknown) {
    auto status = out_tensors_[0]->CopyTensorData(*in_tensors_[index_]);  // tensorlist shape
    if (status == RET_ERROR) {
      MS_LOG(ERROR) << "copy tensor data failed!";
      return RET_ERROR;
    }
  } else {
    // reset 0 and dtype = dtype_
    auto out_ptr = reinterpret_cast<char *>(out_tensors_[0]->MutableData());
    memset(out_ptr, 0, lite::DataTypeSize(dtype_) * out_tensors_[0]->ElementsNum());
  }
  return RET_OK;
}

int TensorListGetItemCPUKernel::ReSize() { return RET_OK; }

kernel::LiteKernel *CpuTensorListGetItemFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                          const std::vector<lite::Tensor *> &outputs,
                                                          OpParameter *op_parameter, const lite::InnerContext *ctx,
                                                          const kernel::KernelKey &desc,
                                                          const mindspore::lite::PrimitiveC *primitive) {
  if (op_parameter == nullptr) {
    MS_LOG(ERROR) << "Input op_parameter is nullptr!";
    return nullptr;
  }
  if (ctx == nullptr) {
    MS_LOG(ERROR) << "Input context is nullptr!";
    free(op_parameter);
    return nullptr;
  }
  MS_ASSERT(desc.type == schema::PrimitiveType_TensorListGetItem);
  auto *kernel = new (std::nothrow) TensorListGetItemCPUKernel(op_parameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new TensorListGetItemCPUKernel fail!";
    free(op_parameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed! name: " << op_parameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(op_parameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_TensorListGetItem, CpuTensorListGetItemFp32KernelCreator)
}  // namespace mindspore::kernel
