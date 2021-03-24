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
#include "src/runtime/kernel/arm/base/tensorlist_fromtensor.h"
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

int TensorListFromTensorCPUKernel::Init() { return RET_OK; }

int TensorListFromTensorCPUKernel::ReSize() { return RET_OK; }

int TensorListFromTensorCPUKernel::Run() {
  input0_ = in_tensors_[0];  // row tensor
  input1_ = in_tensors_[1];  // element_shape tensor
  output0_ = out_tensors_[0];
  if (IsCompatibleShape() != RET_OK) {
    MS_LOG(ERROR) << "IsNotCompatibleShape!";
    return RET_ERROR;
  }
  dtype_ = in_tensors_[0]->data_type();
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
  auto data_offset = devision_dim0 * lite::DataTypeSize(dtype_);
  auto in_data = reinterpret_cast<char *>(input0_->data_c());
  MS_ASSERT(in_data != nullptr);
  // copy data from input0(tensor) to output(tensorlist) vector<*tensor>
  for (int i = 0; i < dim0; ++i) {
    auto out_ptr = output0->GetTensor(i);
    MS_ASSERT(out_ptr != nullptr);
    if (out_ptr->ElementsNum() != devision_dim0) {
      MS_LOG(ERROR) << "tensors_[" << i << "].ElementsNum():" << out_ptr->ElementsNum()
                    << " must be euqal to devision_dim0:" << devision_dim0;
      return RET_ERROR;
    }
    auto out_data = out_ptr->data_c();
    MS_ASSERT(out_data != nullptr);
    memcpy(out_data, in_data, data_offset);
    out_ptr->set_data_type(dtype_);
    in_data += data_offset;
  }
  output0->set_tensors_data_type(dtype_);
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_TensorListFromTensor,
           LiteKernelCreator<TensorListFromTensorCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_TensorListFromTensor, LiteKernelCreator<TensorListFromTensorCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_TensorListFromTensor,
           LiteKernelCreator<TensorListFromTensorCPUKernel>)
}  // namespace mindspore::kernel
