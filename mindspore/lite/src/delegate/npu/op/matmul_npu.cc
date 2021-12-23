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

#include "src/delegate/npu/op/matmul_npu.h"
#include "src/delegate/npu/npu_converter_utils.h"
namespace mindspore {
constexpr int BIAS_INDEX = 2;
constexpr int MATMUL_OUTPUT_DIM = 2;
constexpr int MATMUL_INPUT_SIZE = 3;

int MatMulNPUOp::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                           const std::vector<mindspore::MSTensor> &out_tensors) {
  if (in_tensors.size() == MATMUL_INPUT_SIZE) {
    if (in_tensors[BIAS_INDEX].Shape().size() != 1) {
      return RET_NOT_SUPPORT;
    }
  }
  return RET_OK;
}

int MatMulNPUOp::Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                      const std::vector<mindspore::MSTensor> &out_tensors) {
  matmul_ = new (std::nothrow) hiai::op::MatMul(name_);
  if (matmul_ == nullptr) {
    MS_LOG(ERROR) << "New matmul npu operator for op " << name_ << " failed.";
    return RET_ERROR;
  }
  if (in_tensors.size() == MATMUL_INPUT_SIZE) {
    has_bias_ = true;
  }
  auto matmul_prim = primitive->value_as_MatMulFusion();
  if (matmul_prim == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }
  matmul_->set_attr_transpose_x1(matmul_prim->transpose_a());
  matmul_->set_attr_transpose_x2(matmul_prim->transpose_b());
  return RET_OK;
}

int MatMulNPUOp::SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                              const std::vector<mindspore::MSTensor> &out_tensors,
                              const std::vector<ge::Operator *> &npu_inputs) {
  matmul_->set_input_x1(*npu_inputs[0]);
  matmul_->set_input_x2(*npu_inputs[1]);
  if (has_bias_) {
    add_op_ = new (std::nothrow) hiai::op::Add(name_ + "_add");
    if (add_op_ == nullptr) {
      MS_LOG(ERROR) << "new add op failed.";
      return RET_ERROR;
    }
    add_op_->set_input_x1(*matmul_);
    auto bias_shape = in_tensors[BIAS_INDEX].Shape();
    auto bias_tensor = ConverterToNPUTensor(in_tensors[BIAS_INDEX]);
    if (bias_tensor == nullptr) {
      MS_LOG(ERROR) << "Get bias_tensor failed.";
      return RET_ERROR;
    }

    ge::TensorDesc bias_tensor_desc(ConverterToNPUShape({1, bias_shape[0], 1, 1}));
    if (out_tensors[0].Shape().size() == MATMUL_OUTPUT_DIM) {
      bias_tensor_desc.SetShape(ConverterToNPUShape({1, bias_shape[0]}));
    }
    bias_tensor->SetTensorDesc(bias_tensor_desc);

    bias_ = new (std::nothrow) hiai::op::Const(name_ + "_bias");
    if (bias_ == nullptr) {
      MS_LOG(ERROR) << "new bias const failed.";
      return RET_ERROR;
    }
    bias_->set_attr_value(bias_tensor);
    add_op_->set_input_x2(*bias_);
  }
  return RET_OK;
}

ge::Operator *MatMulNPUOp::GetNPUOp() {
  if (has_bias_) {
    return add_op_;
  }
  return matmul_;
}

MatMulNPUOp::~MatMulNPUOp() {
  if (matmul_ != nullptr) {
    delete matmul_;
    matmul_ = nullptr;
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
}  // namespace mindspore
