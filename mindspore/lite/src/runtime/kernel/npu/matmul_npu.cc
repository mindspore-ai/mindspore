/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/npu/matmul_npu.h"
#include <memory>
#include "src/kernel_registry.h"
#include "src/runtime/agent/npu/npu_converter_utils.h"

using mindspore::kernel::KERNEL_ARCH::kNPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_MatMul;

namespace mindspore::kernel {
int MatMulNPUKernel::IsSupport(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                               OpParameter *opParameter) {
  if (inputs.size() == 3) {
    if (inputs[2]->shape().size() != 1) {
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int MatMulNPUKernel::SetNPUInputs(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                                  const std::vector<ge::Operator *> &npu_inputs) {
  op_ = new (std::nothrow) hiai::op::MatMul(name_);
  op_->set_input_x1(*npu_inputs[0]);
  op_->set_input_x2(*npu_inputs[1]);
  if (npu_inputs.size() == 3) {
    matmul_parameter_->has_bias_ = true;
    add_op_ = new (std::nothrow) hiai::op::Add(name_ + "_add");
    if (add_op_ == nullptr) {
      MS_LOG(ERROR) << "new add op failed.";
      return RET_ERROR;
    }
    add_op_->set_input_x1(*op_);
    auto bias_shape = inputs[2]->shape();
    auto bias_tensor = std::make_shared<ge::Tensor>();
    if (bias_tensor == nullptr) {
      MS_LOG(ERROR) << "new bias_tensor failed.";
      return RET_ERROR;
    }
    ge::TensorDesc bias_tensor_desc(lite::ConverterToNPUShape({1, bias_shape[0], 1, 1}), ge::FORMAT_NCHW,
                                    lite::ConverterToNPUDataType(inputs[2]->data_type()));
    if (outputs[0]->shape().size() == 2) {
      bias_tensor_desc.SetShape(lite::ConverterToNPUShape({1, bias_shape[0]}));
    }
    bias_tensor->SetTensorDesc(bias_tensor_desc);
    bias_tensor->SetData(reinterpret_cast<const uint8_t *>(inputs[2]->data_c()), inputs[2]->Size());
    bias_ = new (std::nothrow) hiai::op::Const(name_ + "_bias");
    if (bias_ == nullptr) {
      MS_LOG(ERROR) << "new bias const failed.";
      return RET_ERROR;
    }
    bias_->set_attr_value(bias_tensor);
    add_op_->set_input_x2(*bias_);
  }

  op_->set_attr_transpose_x1(matmul_parameter_->a_transpose_);
  op_->set_attr_transpose_x2(matmul_parameter_->b_transpose_);
  return RET_OK;
}

ge::Operator *mindspore::kernel::MatMulNPUKernel::GetNPUOp() {
  if (matmul_parameter_->has_bias_) {
    return add_op_;
  }
  return op_;
}

MatMulNPUKernel::~MatMulNPUKernel() {
  if (op_ != nullptr) {
    delete op_;
    op_ = nullptr;
  }
  if (add_op_ != nullptr) {
    delete add_op_;
    add_op_ = nullptr;
  }
  if (bias_ != nullptr) {
    delete bias_;
    bias_ = nullptr;
  }
}

REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_MatMul, NPUKernelCreator<MatMulNPUKernel>)
}  // namespace mindspore::kernel
