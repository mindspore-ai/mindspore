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
#include "src/runtime/kernel/arm/fp32/tensorlist_fromtensor_fp32.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_TensorListFromTensor;

namespace mindspore::kernel {

int TensorListFromTensorCPUKernel::IsCompatibleShape() {
  if (input1_->data_type() != kNumberTypeInt && input1_->data_type() != kNumberTypeInt32) {  // element_shape
    MS_LOG(ERROR) << "in_tensors_[1] data type is must be int";
    return RET_ERROR;
  }
  int in1_ele_num = input1_->ElementsNum();
  std::vector<int> tensor_shape = input0_->shape();
  if (static_cast<int>(tensor_shape.size() - 1) != in1_ele_num) {
    MS_LOG(ERROR) << "in_tensors_[0].shape().size() - 1:" << tensor_shape.size() - 1
                  << " must be equal in_tensors_[1].ElementsNum():" << in1_ele_num;
    return RET_ERROR;
  }
  int *elements_shape = reinterpret_cast<int *>(input1_->data_c());  // element shape in tensor data
  for (int i = 0; i < in1_ele_num; ++i) {
    int dim0 = tensor_shape[i + 1];
    int dim1 = elements_shape[i];
    if (dim0 >= 0 && dim1 >= 0 && dim0 != dim1) {
      MS_LOG(ERROR) << "input0_->shape()[" << i + 1 << "]:" << dim0 << " is not equal input1_->data_c()[" << i
                    << "]:" << dim1;
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int TensorListFromTensorCPUKernel::Init() {
  input0_ = in_tensors_[0];  // row tensor
  input1_ = in_tensors_[1];  // element_shape tensor
  output0_ = out_tensors_[0];
  return IsCompatibleShape();
}

int TensorListFromTensorCPUKernel::ReSize() {
  auto ret = this->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed!";
    return ret;
  }
  return RET_OK;
}

int TensorListFromTensorCPUKernel::Run() {
  input0_ = in_tensors_[0];  // row tensor
  input1_ = in_tensors_[1];  // element_shape tensor
  output0_ = out_tensors_[0];
  if (input0_->shape().size() == 0) {
    MS_LOG(ERROR) << "input0_->shape().size():" << input0_->shape().size() << " must be greater than 0";
  }
  int dim0 = input0_->shape()[0];
  if (dim0 <= 0) {
    MS_LOG(ERROR) << "input0_->shape()[0]:" << dim0 << " must be greater than 0!";
    return RET_ERROR;
  }
  auto output0 = reinterpret_cast<lite::TensorList *>(output0_);
  if (dim0 != output0->ElementsNum()) {
    MS_LOG(ERROR) << "output0_->ElementsNum():" << output0->ElementsNum() << " must be equal to dim0:" << dim0;
    return RET_ERROR;
  }
  int devision_dim0 = input0_->ElementsNum() / dim0;
  auto in_ptr = reinterpret_cast<float *>(input0_->data_c());
  // copy data from input0(tensor) to output(tensorlist) vector<*tensor>
  for (int i = 0; i < dim0; ++i) {
    auto out_ptr = output0->GetTensor(i);
    MS_ASSERT(out_ptr != nullptr);
    if (out_ptr->ElementsNum() != devision_dim0) {
      MS_LOG(ERROR) << "tensors_[" << i << "].ElementsNum():" << out_ptr->ElementsNum()
                    << " must be euqal to devision_dim0:" << devision_dim0;
      return RET_ERROR;
    }
    memcpy(reinterpret_cast<float *>(out_ptr->MutableData()), in_ptr, devision_dim0 * sizeof(float));
    in_ptr += devision_dim0;
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
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_TensorListFromTensor, CpuTensorListFromTensorFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_TensorListFromTensor, CpuTensorListFromTensorFp32KernelCreator)
}  // namespace mindspore::kernel
