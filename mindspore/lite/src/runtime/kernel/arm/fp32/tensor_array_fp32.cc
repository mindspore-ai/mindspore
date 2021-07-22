/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include <memory>
#include "include/errorcode.h"
#include "src/runtime/kernel/arm/fp32/tensor_array_fp32.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/tensorlist.h"
#include "src/common/log_util.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_TensorArray;
using mindspore::schema::PrimitiveType_TensorArrayRead;
using mindspore::schema::PrimitiveType_TensorArrayWrite;

namespace mindspore::kernel {
constexpr int kTensorArrayReadInSize = 3;
constexpr int kTensorArrayWriteInSize = 4;
constexpr int kHandleIndex = 0;
// input index for tensor arrya write/read
constexpr int kIndexInputIdx = 1;
constexpr int kValueIndex = 2;

int TensorArrayCPUKernel::Init() {
  MSLITE_CHECK_PTR(this->ta_param_);
  int *element_shape = this->ta_param_->element_shape_;
  MSLITE_CHECK_PTR(element_shape);
  int element_shape_size = this->ta_param_->element_shape_size_;
  // element shape to vector
  std::vector<int> element_shape_v(element_shape, element_shape + element_shape_size);
  // check inputs' size
  if (this->in_tensors_.size() != 1) {
    MS_LOG(ERROR) << "invalid number of tensor array!";
    return RET_ERROR;
  }
  // get size from input
  lite::Tensor *input = InnerKernel::in_tensors_.at(kInputIndex);
  // check input tensor's datatype is int or not
  if (input->data_type() != TypeId::kNumberTypeInt32 || input->ElementsNum() != 1) {
    MS_LOG(ERROR) << "checked invalid tensor array's input!";
    return RET_ERROR;
  }
  std::vector<int> shape = {*(static_cast<int *>(input->data()))};
  this->tensor_list_ = std::make_unique<lite::TensorList>(shape, element_shape_v);
  std::vector<std::vector<int>> tensor_shape(shape.front(), element_shape_v);
  this->tensor_list_->MallocTensorListData(TypeId::kNumberTypeFloat32, tensor_shape);
  this->tensor_list_->MallocData();
  return RET_OK;
}

inline int TensorArrayCPUKernel::Run() {
  // set handle to outputs, fake malloc, call set_data
  lite::Tensor *output = out_tensors_.at(kOutputIndex);
  void *tensor_list = static_cast<void *>(this->tensor_list_.get());
  void *delta = InnerKernel::ms_context_->allocator->Malloc(sizeof(tensor_list));
  MSLITE_CHECK_PTR(delta);
  memcpy(delta, &tensor_list, sizeof(tensor_list));
  output->set_data(delta);
  return RET_OK;
}

/**
 * read operate just copy handle(tensor buffer) to output,
 * on the contrary, write just copy output to buffer.
 */
int TensorArrayBaseCPUKernel::Init() {
  // check index_tensor
  lite::Tensor *input_y = in_tensors_.at(kIndexInputIdx);
  if (input_y->category() != lite::Tensor::Category::CONST_TENSOR) {
    MS_LOG(ERROR) << "invalid category of index input";
    return RET_ERROR;
  }
  MSLITE_CHECK_PTR(input_y->data());
  index_ = *(static_cast<int *>(input_y->data()));
  return RET_OK;
}

int TensorArrayBaseCPUKernel::Run() {
  lite::Tensor *input_x = in_tensors_.at(kHandleIndex);
  // check output shape is same as handle
  lite::TensorList **delta = static_cast<lite::TensorList **>(input_x->data());
  lite::TensorList *tensor_list = *delta;
  if (tensor_list == nullptr) {
    MS_LOG(ERROR) << "get tensor list failed!";
    return RET_ERROR;
  }
  this->handle_ = tensor_list->GetTensor(index_);
  MSLITE_CHECK_PTR(this->handle_);
  return RET_OK;
}

int TensorArrayReadCPUKernel::Init() {
  // just check
  if (in_tensors_.size() != kTensorArrayReadInSize) {
    MS_LOG(ERROR) << "invalid input numbers of TensorArrayReadCPUKernel";
    return RET_ERROR;
  }
  // check index_tensor
  TensorArrayBaseCPUKernel::Init();
  return RET_OK;
}

int TensorArrayReadCPUKernel::Run() {
  TensorArrayBaseCPUKernel::Run();
  lite::Tensor *output = out_tensors_.at(kOutputIndex);
  lite::Tensor::CopyTensorData(*(TensorArrayBaseCPUKernel::handle_), output);
  return RET_OK;
}

int TensorArrayWriteCPUKernel::Init() {
  // just check
  if (in_tensors_.size() != kTensorArrayWriteInSize) {
    MS_LOG(ERROR) << "invalid input numbers of TensorArrayWriteCPUKernel";
    return RET_ERROR;
  }
  TensorArrayBaseCPUKernel::Init();
  return RET_OK;
}

int TensorArrayWriteCPUKernel::Run() {
  TensorArrayBaseCPUKernel::Run();
  lite::Tensor *value = in_tensors_.at(kValueIndex);
  lite::Tensor::CopyTensorData(*value, TensorArrayBaseCPUKernel::handle_);
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_TensorArray, LiteKernelCreator<TensorArrayCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_TensorArrayRead, LiteKernelCreator<TensorArrayReadCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_TensorArrayWrite, LiteKernelCreator<TensorArrayWriteCPUKernel>)
}  // namespace mindspore::kernel
