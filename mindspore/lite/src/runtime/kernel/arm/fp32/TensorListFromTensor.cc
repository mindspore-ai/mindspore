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
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/fp32/TensorListFromTensor.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_TensorListFromTensor;

namespace mindspore::kernel {

bool TensorListFromTensorCPUKernel::IsCompatibleShape() {
  if (input1_->data_type() != kNumberTypeInt) {  // element_shape
    MS_LOG(ERROR) << "in_tensors_[1] data type is must be \"kNumberTypeInt\", but now is:" << input1_->data_type();
    return false;
  }
  int in1_ele_num = input1_->ElementsNum();
  std::vector<int> tensor_shape = input0_->shape();
  if (static_cast<int>(tensor_shape.size() - 1) != in1_ele_num) {
    MS_LOG(ERROR) << "in_tensors_[0].shape() - 1:" << tensor_shape.size() - 1
                  << " must be equal in_tensors_[1].ElementsNum():" << in1_ele_num;
    return false;
  }
  int *elements_shape = reinterpret_cast<int *>(input1_->data_c());  // element shape in tensor data
  for (int i = 0; i < in1_ele_num; ++i) {
    const int dim0 = tensor_shape.at(i + 1);
    const int dim1 = *(elements_shape + i);
    if (dim0 >= 0 && dim1 >= 0 && dim0 != dim1) {
      MS_LOG(ERROR) << "input0_->shape()[" << i + 1 << "]:" << dim0 << " is not equal input1_->data_c()[" << i
                    << "]:" << dim1;
      return false;
    }
  }
  return true;
}

int TensorListFromTensorCPUKernel::Init() {
  input0_ = in_tensors_.at(0);  // row tensor
  input1_ = in_tensors_.at(1);  // element_shape tensor
  output0_ = out_tensors_.at(0);
  output1_ = out_tensors_.at(1);
  return IsCompatibleShape();
}

int TensorListFromTensorCPUKernel::ReSize() { return RET_OK; }

int TensorListFromTensorCPUKernel::Run() {
  int dim0 = input0_->shape().at(0);
  size_t devision_dim0 = input0_->ElementsNum() / dim0;
  auto out0_ptr = reinterpret_cast<int *>(output0_->MutableData());
  *out0_ptr = dim0;
  *(out0_ptr + 1) = input0_->data_type();
  auto status = output1_->CopyTensorData(*input1_);
  if (status == RET_ERROR) {
    MS_LOG(ERROR) << "copy tensor data failed!";
    return RET_ERROR;
  }
  if (dim0 != static_cast<int>(out_tensors_.size() - 2)) {
    MS_LOG(ERROR) << "out_tensors_.size() - 2:[" << out_tensors_.size() - 2
                  << "] must be equal in_tensors_[0].shape()[0]:[" << dim0 << "]";
    return RET_ERROR;
  }
  auto in_ptr = reinterpret_cast<float *>(input0_);
  size_t index = 0;
  for (int i = 0; i < dim0; ++i) {
    auto out_ptr = reinterpret_cast<float *>(out_tensors_.at(i + 2)->MutableData());
    memcpy(out_ptr, in_ptr + index, devision_dim0 * sizeof(float));
    index += devision_dim0;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuTensorListFromTensorFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
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
  MS_ASSERT(desc.type == schema::PrimitiveType_TensorListFromTensor);
  op_parameter->thread_num_ = ctx->thread_num_;
  auto *kernel = new (std::nothrow) TensorListFromTensorCPUKernel(op_parameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new TensorListFromTensorCPUKernel fail!";
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_TensorListFromTensor, CpuTensorListFromTensorFp32KernelCreator)
}  // namespace mindspore::kernel
