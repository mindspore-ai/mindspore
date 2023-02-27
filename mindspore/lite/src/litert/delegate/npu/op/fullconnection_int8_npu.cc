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

#include "src/litert/delegate/npu/op/fullconnection_int8_npu.h"
#include <memory>
#include "src/litert/delegate/npu/npu_converter_utils.h"

namespace mindspore::lite {
constexpr int FC_INPUT_DIM = 2;
constexpr int FC_INPUT_SIZE = 3;

void SetFCQuantParam(hiai::op::QuantizedFullyConnection *fc, const std::vector<mindspore::MSTensor> &in_tensors) {
  fc->set_attr_x_quant_scale(in_tensors.at(0).QuantParams().front().scale);
  fc->set_attr_x_quant_offset(in_tensors.at(0).QuantParams().front().zero_point);
  fc->set_attr_x_quant_type(1);

  std::vector<float> filter_scales(in_tensors.at(WEIGHT_INDEX).QuantParams().size());
  for (size_t i = 0; i < in_tensors.at(WEIGHT_INDEX).QuantParams().size(); i++) {
    filter_scales[i] = in_tensors.at(WEIGHT_INDEX).QuantParams().at(i).scale;
  }
  fc->set_attr_w_quant_scales(filter_scales);
  fc->set_attr_w_quant_type(1);
}

int FullconnectionINT8NPUOp::Init(const schema::Primitive *primitive,
                                  const std::vector<mindspore::MSTensor> &in_tensors,
                                  const std::vector<mindspore::MSTensor> &out_tensors) {
  auto fc_prim = primitive->value_as_FullConnection();
  if (fc_prim == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }
  act_type_ = fc_prim->activation_type();
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

  fc_ = new (std::nothrow) hiai::op::QuantizedFullyConnection(name_);
  if (fc_ == nullptr) {
    MS_LOG(ERROR) << "New matmul operator for fullconnection op " << name_ << " failed.";
    return RET_ERROR;
  }
  CHECK_LESS_RETURN(in_tensors.size(), FC_INPUT_SIZE - 1);
  SetFCQuantParam(fc_, in_tensors);
  return RET_OK;
}

int FullconnectionINT8NPUOp::SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                                          const std::vector<mindspore::MSTensor> &out_tensors,
                                          const std::vector<ge::Operator *> &npu_inputs) {
  reshape_->set_input_x(*npu_inputs[0]);
  fc_->set_input_x(*reshape_);

  weight_ = new (std::nothrow) hiai::op::Const(name_ + "_w");
  if (weight_ == nullptr) {
    MS_LOG(ERROR) << "New weight const failed.";
    return RET_ERROR;
  }
  auto weight_tensor = ConverterToNPUTensor(in_tensors[1], true);
  weight_->set_attr_value(weight_tensor);
  fc_->set_input_w(*weight_);
  if (in_tensors.size() >= FC_INPUT_SIZE) {
    bias_ = new (std::nothrow) hiai::op::Const(name_ + "_b");
    if (bias_ == nullptr) {
      MS_LOG(ERROR) << "New weight const failed.";
      return RET_ERROR;
    }
    auto bias_tensor = ConverterToNPUTensor(in_tensors[kBiasIndex], true);
    bias_->set_attr_value(bias_tensor);
    fc_->set_input_b(*bias_);
  }
  fc_->set_attr_num_output(out_tensors.front().ElementNum());

  if (act_type_ != schema::ActivationType_NO_ACTIVATION) {
    auto ret = SetActivation(fc_, act_type_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "New activation npu operator for op " << name_ << " failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

ge::Operator *FullconnectionINT8NPUOp::GetNPUOp() {
  if (act_type_ != schema::ActivationType_NO_ACTIVATION) {
    return act_;
  }
  return fc_;
}

FullconnectionINT8NPUOp::~FullconnectionINT8NPUOp() {
  if (reshape_ != nullptr) {
    delete reshape_;
    reshape_ = nullptr;
  }
  if (fc_ != nullptr) {
    delete fc_;
    fc_ = nullptr;
  }
  if (reshape_op_ != nullptr) {
    delete reshape_op_;
    reshape_op_ = nullptr;
  }
  if (bias_ != nullptr) {
    delete bias_;
    bias_ = nullptr;
  }
}
}  // namespace mindspore::lite
