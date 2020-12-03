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
#include <vector>
#include "include/errorcode.h"
#include "ir/dtype/type_id.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/fp32/TensorListStack.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_TensorListStack;

namespace mindspore::kernel {

int TensorListStackCPUKernel::CheckParam() {
  auto in0_dtype = in_tensors_.at(0)->data_type();
  if (in0_dtype != kNumberTypeInt) {
    MS_LOG(ERROR) << "in_tensors_[0]->data_type():" << in0_dtype
                  << " must be equal \"kNumberTypeInt\":" << kNumberTypeInt;
  }
  auto in0_ptr = reinterpret_cast<int *>(in_tensors_.at(0)->data_c());
  if (in0_ptr[1] != dtype_) {
    MS_LOG(ERROR) << "in_tensors_[0].data_type:[" << in0_ptr[1] << "] must be equal "
                  << "param.data_type:[" << dtype_ << "]";
    return RET_ERROR;
  }
  if (num_element_ != -1 && in0_ptr[0] != num_element_) {
    MS_LOG(ERROR) << "in_tensors_[0].dim0:[" << in0_ptr[0] << "] must be equal "
                  << "param.elements_num:[" << num_element_ << "]";
    return RET_ERROR;
  }
  num_element_ = in0_ptr[0];
  return RET_OK;
}

int TensorListStackCPUKernel::Init() {
  output0_ = out_tensors_.at(0);
  if (output0_->format() != schema::Format_NC) {  // shape().size() = 2
    MS_LOG(ERROR) << "out_tensor_[0] format must be \"Format:NC\", but now is:" << output0_->format();
    return RET_ERROR;
  }
  int dim0 = output0_->shape().at(0);
  if (dim0 != 1) {  // dim0 must be 1
    MS_LOG(ERROR) << "out_tensor_[0] dim0 must be 1, but now is:" << dim0;
    return RET_ERROR;
  }
  return CheckParam();
}

int TensorListStackCPUKernel::Run() {
  size_t in_ele_num = 0;
  for (int i = 0; i < num_element_; ++i) {
    in_ele_num += in_tensors_.at(i + 2)->ElementsNum();
  }
  size_t out_ele_num = out_tensors_.at(0)->ElementsNum();
  if (in_ele_num > out_ele_num) {
    MS_LOG(ERROR) << "out_tensors_[0]->ElementsNum():" << out_ele_num << "must greater than or equal to in_ele_num"
                  << in_ele_num;
    return RET_ERROR;
  }
  size_t index = 0;
  auto out_ptr = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());
  for (int i = 0; i < num_element_; ++i) {
    auto in_ptr = reinterpret_cast<float *>(in_tensors_.at(i + 2)->data_c());
    size_t in_size = in_tensors_.at(i + 2)->ElementsNum();
    memcpy(out_ptr + index, in_ptr, in_size * sizeof(float));
    index += in_size;
  }
  return RET_OK;
}

int TensorListStackCPUKernel::ReSize() { return RET_OK; }

kernel::LiteKernel *CpuTensorListStackFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
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
  MS_ASSERT(desc.type == schema::PrimitiveType_TensorListStack);
  auto *kernel = new (std::nothrow) TensorListStackCPUKernel(op_parameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new TensorListStackCPUKernel fail!";
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_TensorListStack, CpuTensorListStackFp32KernelCreator)
}  // namespace mindspore::kernel
