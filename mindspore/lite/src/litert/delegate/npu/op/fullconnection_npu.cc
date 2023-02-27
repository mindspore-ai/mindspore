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

#include "src/litert/delegate/npu/op/fullconnection_npu.h"
#include "src/litert/delegate/npu/op/fullconnection_int8_npu.h"
#include <memory>
#include "src/litert/delegate/npu/npu_converter_utils.h"

namespace mindspore::lite {
constexpr int FC_INPUT_DIM = 2;
constexpr int FC_INPUT_SIZE = 3;

int FullconnectionNPUOp::Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                              const std::vector<mindspore::MSTensor> &out_tensors) {
  auto fc_prim = primitive->value_as_FullConnection();
  if (fc_prim == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }
  act_type_ = fc_prim->activation_type();
  CHECK_LESS_RETURN(in_tensors.size(), 1);
  auto input_shape = in_tensors[0].Shape();
  reshape_ = new (std::nothrow) hiai::op::Reshape(name_ + "_reshape");
  if (reshape_ == nullptr) {
    MS_LOG(ERROR) << "New reshape operator for fullconnection op " << name_ << " failed.";
    return RET_ERROR;
  }

  int col = 1;
  for (int i = 1; i < input_shape.size(); i++) {
    col *= input_shape[i];
  }
  reshape_op_ = new (std::nothrow) hiai::op::Const(name_ + "_reshape_data");
  if (reshape_op_ == nullptr) {
    MS_LOG(ERROR) << "New Const operator for fullconnection op " << name_ << " failed.";
    return RET_ERROR;
  }
  std::vector<int> reshape_data = {static_cast<int>(input_shape[0]), col};
  ge::TensorDesc reshape_tensor_desc(ge::Shape({FC_INPUT_DIM}), ge::FORMAT_NCHW, ge::DT_INT32);
  ge::TensorPtr reshape_tensor = std::make_shared<hiai::Tensor>(reshape_tensor_desc);
  reshape_tensor->SetData(reinterpret_cast<uint8_t *>(reshape_data.data()), FC_INPUT_DIM * sizeof(int32_t));
  reshape_op_->set_attr_value(reshape_tensor);
  reshape_->set_input_shape(*reshape_op_);

  fc_ = new (std::nothrow) hiai::op::MatMul(name_);
  if (fc_ == nullptr) {
    MS_LOG(ERROR) << "New matmul operator for fullconnection op " << name_ << " failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int FullconnectionNPUOp::SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                                      const std::vector<mindspore::MSTensor> &out_tensors,
                                      const std::vector<ge::Operator *> &npu_inputs) {
  CHECK_LESS_RETURN(npu_inputs.size(), 1);
  reshape_->set_input_x(*npu_inputs[0]);
  fc_->set_input_x1(*reshape_);

  weight_ = new (std::nothrow) hiai::op::Const(name_ + "_w");
  if (weight_ == nullptr) {
    MS_LOG(ERROR) << "New weight const failed.";
    return RET_ERROR;
  }
  CHECK_LESS_RETURN(in_tensors.size(), kInputSize1);
  auto weight_tensor = ConverterToNPUTensor(in_tensors[1]);
  weight_->set_attr_value(weight_tensor);
  fc_->set_input_x2(*weight_).set_attr_transpose_x2(true);

  if (in_tensors.size() >= FC_INPUT_SIZE) {
    has_bias_ = true;
  }
  if (has_bias_) {
    biasadd_ = new (std::nothrow) hiai::op::BiasAdd(name_ + "_biasadd");
    if (biasadd_ == nullptr) {
      MS_LOG(ERROR) << "New biasadd operator for fullconnection op " << name_ << " failed.";
      return RET_ERROR;
    }

    auto ret = InitBiasConst(in_tensors);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Set bias for convolution op " << name_ << " failed when running npu";
      return RET_ERROR;
    }
    biasadd_->set_input_x(*fc_).set_input_bias(*bias_);
  }

  if (act_type_ != schema::ActivationType_NO_ACTIVATION) {
    auto ret = biasadd_ == nullptr ? SetActivation(fc_, act_type_) : SetActivation(biasadd_, act_type_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "New activation npu operator for op " << name_ << " failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

ge::Operator *FullconnectionNPUOp::GetNPUOp() {
  if (act_type_ != schema::ActivationType_NO_ACTIVATION) {
    return act_;
  }
  if (has_bias_) {
    return biasadd_;
  }
  return fc_;
}

FullconnectionNPUOp::~FullconnectionNPUOp() {
  if (reshape_ != nullptr) {
    delete reshape_;
    reshape_ = nullptr;
  }
  if (fc_ != nullptr) {
    delete fc_;
    fc_ = nullptr;
  }
  if (biasadd_ != nullptr) {
    delete biasadd_;
    biasadd_ = nullptr;
  }
  if (reshape_op_ != nullptr) {
    delete reshape_op_;
    reshape_op_ = nullptr;
  }
}
NPUOp *GetNPUFCOp(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                  const std::vector<mindspore::MSTensor> &out_tensors, std::string name) {
  auto shape = out_tensors.front().Shape();
  if (std::find(shape.begin(), shape.end(), -1) != shape.end()) {
    MS_LOG(ERROR) << "NPU does not support runtime inference shape.";
    return nullptr;
  }

  if (in_tensors[0].DataType() != DataType::kNumberTypeFloat32 &&
      in_tensors[0].DataType() != DataType::kNumberTypeFloat16) {
    MS_LOG(ERROR) << "Npu does not support datatype " << static_cast<int>(in_tensors[0].DataType());
    return nullptr;
  }

  NPUOp *op = nullptr;
  if (in_tensors[1].DataType() == DataType::kNumberTypeInt8) {
    op = new (std::nothrow) FullconnectionINT8NPUOp(primitive, in_tensors, out_tensors, name);
  } else {
    op = new (std::nothrow) FullconnectionNPUOp(primitive, in_tensors, out_tensors, name);
  }

  if (op == nullptr) {
    MS_LOG(ERROR) << "create conv op for Npu failed.";
    return nullptr;
  }

  auto ret = op->IsSupport(primitive, in_tensors, out_tensors);
  if (ret != RET_OK) {
    delete op;
    return nullptr;
  }
  ret = op->Init(primitive, in_tensors, out_tensors);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "NPU op init failed.";
    delete op;
    return nullptr;
  }
  return op;
}
}  // namespace mindspore::lite
