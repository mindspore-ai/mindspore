/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "src/litert/delegate/coreml/op/coreml_op.h"
#include "nnacl/base/cast_base.h"
namespace mindspore::lite {
int CoreMLOp::Init() {
  auto ret = InitParams();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "CoreML op " << name_ << "'s parameter initialization failed.";
    return RET_ERROR;
  }
  op_ = std::make_unique<CoreML::Specification::NeuralNetworkLayer>();
  if (op_ == nullptr) {
    MS_LOG(ERROR) << "New CoreML op " << name_ << " failed.";
    return RET_ERROR;
  }
  op_->set_name("CoreML_" + name_);
  return RET_OK;
}

int CoreMLOp::SetActivation(schema::ActivationType act_type) {
  act_op_ = std::make_unique<CoreML::Specification::NeuralNetworkLayer>();
  if (act_op_ == nullptr) {
    MS_LOG(ERROR) << "New CoreML op " << name_ << "_activation failed.";
    return RET_ERROR;
  }
  act_op_->set_name("CoreML_" + name_ + "_activation");
  switch (act_type) {
    case schema::ActivationType_RELU:
      act_op_->mutable_activation()->mutable_relu();
      break;
    case schema::ActivationType_RELU6: {
      auto clip_param = act_op_->mutable_clip();
      clip_param->set_minval(0);
      clip_param->set_maxval(kValueThreshold6);
      break;
    }
    case schema::ActivationType_TANH:
      act_op_->mutable_activation()->mutable_tanh();
      break;
    case schema::ActivationType_SIGMOID:
      act_op_->mutable_activation()->mutable_sigmoid();
      break;
    default:
      MS_LOG(ERROR) << "Unsupported activation type.";
      return RET_ERROR;
  }
  return RET_OK;
}

int CoreMLOp::SetPadding(std::vector<int> pad_list) {
  pad_op_ = std::make_unique<CoreML::Specification::NeuralNetworkLayer>();
  if (pad_op_ == nullptr) {
    MS_LOG(ERROR) << "New CoreML op " << name_ << "_pad failed.";
    return RET_ERROR;
  }
  pad_op_->set_name("CoreML_" + name_ + "_pad");
  auto pad_param = pad_op_->mutable_padding();
  pad_param->mutable_constant();
  auto height_border = pad_param->mutable_paddingamounts()->add_borderamounts();
  auto width_border = pad_param->mutable_paddingamounts()->add_borderamounts();
  height_border->set_startedgesize(pad_list[PAD_UP]);
  height_border->set_endedgesize(pad_list[PAD_DOWN]);
  width_border->set_startedgesize(pad_list[PAD_LEFT]);
  width_border->set_endedgesize(pad_list[PAD_RIGHT]);
  return RET_OK;
}

int CoreMLOp::SetConstInput(const mindspore::MSTensor &in_tensor) {
  MS_CHECK_TRUE_MSG(in_tensor.IsConst(), RET_ERROR, "Only constant tensor can be set as CoreML Const op.");
  std::string const_op_name = "CoreML_" + in_tensor.Name() + "_const";
  auto const_op = std::make_unique<CoreML::Specification::NeuralNetworkLayer>();
  if (const_op == nullptr) {
    MS_LOG(ERROR) << "New CoreML const op " << const_op_name << " for op " << name_ << " failed.";
    return RET_ERROR;
  }
  const_op->set_name(const_op_name);
  auto const_param = const_op->mutable_loadconstantnd();
  for (auto i : in_tensor.Shape()) {
    const_param->add_shape(static_cast<uint64_t>(i));
  }
  if (in_tensor.Shape().empty()) {
    const_param->add_shape(1);
  }
  // set const data
  auto org_data = in_tensor.Data().get();
  auto *ml_data_container = const_param->mutable_data()->mutable_floatvalue();
  ml_data_container->Resize(in_tensor.ElementNum(), 0);
  auto *ml_data = reinterpret_cast<float *>(ml_data_container->mutable_data());
  if (in_tensor.DataType() == DataType::kNumberTypeInt32) {
    Int32ToFloat32(reinterpret_cast<const int *>(org_data), ml_data, in_tensor.ElementNum());
  } else if (in_tensor.DataType() == DataType::kNumberTypeFloat32) {
    memcpy(ml_data, org_data, in_tensor.DataSize());
  } else {
    MS_LOG(ERROR) << "Unsupported const input data type: " << static_cast<int>(in_tensor.DataType());
    return RET_ERROR;
  }
  const_ops_[in_tensor.Name()] = std::move(const_op);
  return RET_OK;
}

void CoreMLOp::SetMLOpInOut() {
  MS_ASSERT(op_ != nullptr);
  auto input_name = in_tensors_.at(0).Name();
  if (pad_op_ != nullptr) {
    std::string pad_name = op_->name() + "_pad_0";
    pad_op_->add_input(input_name);
    pad_op_->add_output(pad_name);
    op_->add_input(pad_name);
  } else {
    op_->add_input(input_name);
  }
  auto output_name = out_tensors_.at(0).Name();
  if (act_op_ != nullptr) {
    std::string act_name = op_->name() + "_act_0";
    op_->add_output(act_name);
    act_op_->add_input(act_name);
    act_op_->add_output(output_name);
  } else {
    op_->add_output(output_name);
  }
}

std::vector<CoreML::Specification::NeuralNetworkLayer *> CoreMLOp::GetLayers() {
  MS_ASSERT(op_ != nullptr);
  std::vector<CoreML::Specification::NeuralNetworkLayer *> ret_ops;
  if (pad_op_ != nullptr) {
    ret_ops.push_back(pad_op_.release());
  }
  if (!const_ops_.empty()) {
    for (auto it = const_ops_.begin(); it != const_ops_.end(); it++) {
      ret_ops.push_back(it->second.release());
    }
  }
  ret_ops.push_back(op_.release());
  if (act_op_ != nullptr) {
    ret_ops.push_back(act_op_.release());
  }
  return ret_ops;
}
}  // namespace mindspore::lite
