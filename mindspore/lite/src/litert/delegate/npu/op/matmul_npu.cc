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

#include "src/litert/delegate/npu/op/matmul_npu.h"
#include "src/litert/delegate/npu/npu_converter_utils.h"
namespace mindspore::lite {
constexpr int BIAS_INDEX = 2;
constexpr int MATMUL_COMMON_DIM = 2;
constexpr int MATMUL_INPUT_SIZE = 3;
constexpr int BATCH_MATMUL_MAX_SHAPE = 1024;

int MatMulNPUOp::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                           const std::vector<mindspore::MSTensor> &out_tensors) {
  CHECK_LESS_RETURN(in_tensors.size(), MATMUL_INPUT_SIZE - 1);

  MS_CHECK_TRUE_RET(in_tensors.size() >= MATMUL_COMMON_DIM, RET_ERROR);
  if (in_tensors.front().Shape().size() > MATMUL_COMMON_DIM || in_tensors.at(1).Shape().size() > MATMUL_COMMON_DIM) {
    // The size of each input dim should be less than 1024 in batchmatmul, whose input dim exceeds 2.
    bool is_exceed_dim = std::any_of(in_tensors.begin(), in_tensors.begin() + 1, [](const MSTensor &input) {
      return std::any_of(input.Shape().begin(), input.Shape().end(),
                         [](int64_t size) { return size > BATCH_MATMUL_MAX_SHAPE; });
    });
    if (is_exceed_dim) {
      return RET_NOT_SUPPORT;
    }
  }
  if (in_tensors.size() == MATMUL_INPUT_SIZE) {
    if (in_tensors[BIAS_INDEX].Shape().size() != 1) {
      return RET_NOT_SUPPORT;
    }
  }
  return RET_OK;
}

int MatMulNPUOp::SetActivation(const ge::Operator *input) {
  act_op_->set_input_x(*input);
  auto act_mode = ConverterToNPUActivationMode(act_type_);
  if (act_mode == ACTIVATION_INVALID) {
    MS_LOG(ERROR) << "Unsupported activation type for matmul op " << name_;
    return RET_ERROR;
  }
  act_op_->set_attr_mode(act_mode);
  return RET_OK;
}

int MatMulNPUOp::Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                      const std::vector<mindspore::MSTensor> &out_tensors) {
  auto matmul_prim = primitive->value_as_MatMulFusion();
  if (matmul_prim == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }
  auto input_dims_a = in_tensors[0].Shape().size();
  auto input_dims_b = in_tensors[1].Shape().size();
  if (input_dims_a == NPU_SHAPE_SIZE && input_dims_b == NPU_SHAPE_SIZE) {
    use_batch_matmul_ = true;
    batch_matmul_ = new (std::nothrow) hiai::op::BatchMatMul(name_);
    if (batch_matmul_ == nullptr) {
      MS_LOG(ERROR) << "New batch_matmul npu operator for op " << name_ << " failed.";
      return RET_ERROR;
    }
    batch_matmul_->set_attr_adj_x1(matmul_prim->transpose_a());
    batch_matmul_->set_attr_adj_x2(matmul_prim->transpose_b());
    actual_matmul_ = batch_matmul_;
  } else {
    matmul_ = new (std::nothrow) hiai::op::MatMul(name_);
    if (matmul_ == nullptr) {
      MS_LOG(ERROR) << "New matmul npu operator for op " << name_ << " failed.";
      return RET_ERROR;
    }
    matmul_->set_attr_transpose_x1(matmul_prim->transpose_a());
    matmul_->set_attr_transpose_x2(matmul_prim->transpose_b());
    actual_matmul_ = matmul_;
  }

  if (in_tensors.size() == MATMUL_INPUT_SIZE) {
    has_bias_ = true;
    add_op_ = new (std::nothrow) hiai::op::Add(name_ + "_add");
    if (add_op_ == nullptr) {
      MS_LOG(ERROR) << "new add op failed.";
      return RET_ERROR;
    }
  }

  act_type_ = matmul_prim->activation_type();
  if (act_type_ != schema::ActivationType_NO_ACTIVATION) {
    act_op_ = new (std::nothrow) hiai::op::Activation(name_ + "_act");
    if (act_op_ == nullptr) {
      MS_LOG(ERROR) << "New activation npu operator for op " << name_ << " failed.";
      return RET_ERROR;
    }
  }

  return RET_OK;
}

int MatMulNPUOp::SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                              const std::vector<mindspore::MSTensor> &out_tensors,
                              const std::vector<ge::Operator *> &npu_inputs) {
  CHECK_LESS_RETURN(npu_inputs.size(), MATMUL_INPUT_SIZE - 1);
  if (use_batch_matmul_) {
    batch_matmul_->set_input_x1(*npu_inputs[0]);
    batch_matmul_->set_input_x2(*npu_inputs[1]);
  } else {
    matmul_->set_input_x1(*npu_inputs[0]);
    matmul_->set_input_x2(*npu_inputs[1]);
  }
  if (has_bias_) {
    add_op_->set_input_x1(*actual_matmul_);
    auto bias_shape = in_tensors[BIAS_INDEX].Shape();
    auto bias_tensor = ConverterToNPUTensor(in_tensors[BIAS_INDEX]);
    if (bias_tensor == nullptr) {
      MS_LOG(ERROR) << "Get bias_tensor failed.";
      return RET_ERROR;
    }

    ge::TensorDesc bias_tensor_desc(ConverterToNPUShape({1, bias_shape[0], 1, 1}));
    if (out_tensors[0].Shape().size() == MATMUL_COMMON_DIM) {
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
  if (act_type_ != schema::ActivationType_NO_ACTIVATION) {
    int ret = RET_ERROR;
    if (has_bias_) {
      ret = SetActivation(add_op_);
    } else {
      ret = SetActivation(actual_matmul_);
    }
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "New activation npu operator for op " << name_ << " failed.";
      return ret;
    }
  }
  return RET_OK;
}

ge::Operator *MatMulNPUOp::GetNPUOp() {
  if (act_type_ == schema::ActivationType_NO_ACTIVATION) {
    if (has_bias_) {
      return add_op_;
    }
    return actual_matmul_;
  } else {
    return act_op_;
  }
}

MatMulNPUOp::~MatMulNPUOp() {
  if (actual_matmul_ != nullptr) {
    delete actual_matmul_;
    actual_matmul_ = nullptr;
  }
  if (add_op_ != nullptr) {
    delete add_op_;
    add_op_ = nullptr;
  }
  if (bias_ != nullptr) {
    delete bias_;
    bias_ = nullptr;
  }
  if (act_op_ != nullptr) {
    delete act_op_;
    act_op_ = nullptr;
  }
}
}  // namespace mindspore::lite
