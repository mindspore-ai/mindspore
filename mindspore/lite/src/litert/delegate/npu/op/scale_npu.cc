/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "src/litert/delegate/npu/op/scale_npu.h"
#include <memory>
#include "src/litert/delegate/npu/npu_converter_utils.h"

namespace mindspore::lite {
constexpr int INPUT_INDEX = 0;
constexpr int SCALE_INDEX = 1;
constexpr int BIAS_INDEX = 2;

int ScaleNPUOp::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                          const std::vector<mindspore::MSTensor> &out_tensors) {
  auto scale_prim = primitive->value_as_ScaleFusion();
  if (scale_prim == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op: " << name_;
    return RET_ERROR;
  }
  CHECK_LESS_RETURN(in_tensors.size(), 1);
  auto input_dims = in_tensors.at(INPUT_INDEX).Shape().size();
  axis_ = scale_prim->axis();
  if (axis_ < 0) {
    axis_ = axis_ + input_dims;
  }
  if (axis_ != NHWC_C && axis_ != NCHW_C) {
    if (in_tensors.size() <= BIAS_INDEX) {
      MS_LOG(INFO) << "Npu Scale op does not support axis: " << axis_ << ", trying to convert to Mul op.";
      use_mul_ = true;
      return RET_OK;
    } else {
      MS_LOG(WARNING) << "Npu Scale axis attr only support 1 or channel, now is " << axis_;
      return RET_NOT_SUPPORT;
    }
  }
  if (input_dims < NPU_SHAPE_SIZE) {
    need_expand_ = true;
  }
  return RET_OK;
}

int ScaleNPUOp::Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                     const std::vector<mindspore::MSTensor> &out_tensors) {
  auto scale_prim = primitive->value_as_ScaleFusion();
  if (scale_prim == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }

  if (use_mul_) {
    mul_ = new (std::nothrow) hiai::op::Mul(name_ + "_mul");
    if (mul_ == nullptr) {
      MS_LOG(ERROR) << "New Mul npu operator for op " << name_ << "_mul failed.";
      return RET_ERROR;
    }
    scale_ops_.emplace_back(mul_);
  } else {
    // note that Scale only support the default axis(i.e., 1), setting axis is meaningless.
    scale_ = new (std::nothrow) hiai::op::Scale(name_);
    if (scale_ == nullptr) {
      MS_LOG(ERROR) << "New Scale npu operator for op " << name_ << " failed.";
      return RET_ERROR;
    }
    scale_ops_.emplace_back(scale_);
  }

  if (need_expand_) {
    out_reshape_ = new (std::nothrow) hiai::op::Reshape(name_ + "_restore");
    if (out_reshape_ == nullptr) {
      MS_LOG(ERROR) << "New Reshape npu operator for op " << name_ << "_restore failed.";
      return RET_ERROR;
    }
    scale_ops_.emplace_back(out_reshape_);
  }

  act_type_ = scale_prim->activation_type();
  if (act_type_ != schema::ActivationType_NO_ACTIVATION) {
    act_ = new (std::nothrow) hiai::op::Activation(name_ + "_act");
    if (act_ == nullptr) {
      MS_LOG(ERROR) << "New activation npu operator for op " << name_ << " failed.";
      return RET_ERROR;
    }
    scale_ops_.emplace_back(act_);
  }
  return RET_OK;
}

ge::Operator *ScaleNPUOp::GetNPUOp() {
  if (act_type_ != schema::ActivationType_NO_ACTIVATION) {
    return act_;
  } else if (use_mul_) {
    return mul_;
  } else if (need_expand_) {
    return out_reshape_;
  } else {
    return scale_;
  }
}

int ScaleNPUOp::SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                             const std::vector<mindspore::MSTensor> &out_tensors,
                             const std::vector<ge::Operator *> &npu_inputs) {
  if (use_mul_) {
    auto ret = ConvertScaleToMul(npu_inputs, in_tensors);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Convert Scale to Mul failed, op name: " << name_;
      return RET_ERROR;
    }
  } else {
    auto ret = Adopt4DScale(npu_inputs, in_tensors);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Adopt 4D Scale op failed, op name: " << name_;
      return RET_ERROR;
    }
  }
  if (act_type_ != schema::ActivationType_NO_ACTIVATION) {
    auto ret = SetActivation();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Set Activation failed, op name: " << name_;
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int ScaleNPUOp::SetActivation() {
  ge::Operator *act_input = nullptr;
  if (use_mul_) {
    act_input = mul_;
  } else if (need_expand_) {
    act_input = out_reshape_;
  } else {
    act_input = scale_;
  }
  MS_CHECK_TRUE_MSG(act_input != nullptr, RET_ERROR, "Scale activation input is nullptr.");
  act_->set_input_x(*act_input);
  auto act_mode = ConverterToNPUActivationMode(act_type_);
  if (act_mode == ACTIVATION_INVALID) {
    MS_LOG(ERROR) << "Unsupported activation type for scale op " << name_;
    return RET_ERROR;
  }
  act_->set_attr_mode(act_mode);
  return RET_OK;
}

int ScaleNPUOp::ConvertScaleToMul(const std::vector<ge::Operator *> &npu_inputs,
                                  const std::vector<mindspore::MSTensor> &in_tensors) {
  auto input_shape = in_tensors.at(INPUT_INDEX).Shape();
  auto scale_shape = in_tensors.at(SCALE_INDEX).Shape();
  mul_->set_input_x1(*npu_inputs.at(INPUT_INDEX));
  if (input_shape.size() == scale_shape.size()) {
    mul_->set_input_x2(*npu_inputs.at(SCALE_INDEX));
  } else {
    int64_t valid_dims = input_shape.size();
    std::vector<int> valid_shape(valid_dims, 1);
    for (size_t i = 0; i < scale_shape.size(); i++) {
      valid_shape[axis_ + i] = static_cast<int>(scale_shape[i]);
    }
    auto reshape = new (std::nothrow) hiai::op::Reshape(name_ + "_mul_reshape");
    if (reshape == nullptr) {
      MS_LOG(ERROR) << "New Reshape npu operator for op " << name_ << "_mul_reshape failed.";
      return RET_ERROR;
    }
    scale_ops_.emplace_back(reshape);
    auto valid_data_ptr = reinterpret_cast<const uint8_t *>(valid_shape.data());
    auto shape = GetNPUConst<int>(valid_data_ptr, {valid_dims}, ge::DT_INT32, name_ + "_mul_expand_shape");
    if (shape == nullptr) {
      MS_LOG(ERROR) << "Get shape const for op " << name_ << "_mul failed.";
      return RET_ERROR;
    }
    scale_ops_.emplace_back(shape);
    reshape->set_input_x(*npu_inputs.at(SCALE_INDEX));
    reshape->set_input_shape(*shape);
    mul_->set_input_x2(*reshape);
  }
  return RET_OK;
}

int ScaleNPUOp::Adopt4DScale(const std::vector<ge::Operator *> &npu_inputs,
                             const std::vector<mindspore::MSTensor> &in_tensors) {
  MS_ASSERT(scale_ != nullptr);
  // handle input
  auto org_input_tensor = in_tensors.at(INPUT_INDEX);
  ge::Operator *actual_input = npu_inputs.at(INPUT_INDEX);
  std::vector<int64_t> org_input_shape = org_input_tensor.Shape();
  if (need_expand_) {
    actual_input = ChangeDims(npu_inputs.at(INPUT_INDEX), org_input_shape, name_ + "_expand_input", true);
    if (actual_input == nullptr) {
      MS_LOG(ERROR) << "Change Scale op input dims failed.";
      return RET_ERROR;
    }
  }
  scale_->set_input_x(*actual_input);

  // handle scale, note that the scale axis can only be 1.
  auto org_scale_tensor = in_tensors.at(SCALE_INDEX);
  ge::Operator *actual_scale = npu_inputs.at(SCALE_INDEX);
  if (org_scale_tensor.Shape().size() == DIMENSION_2D) {
    std::vector<int64_t> expand_scale_shape = org_scale_tensor.Shape();
    expand_scale_shape.emplace_back(1);
    actual_scale = ChangeDims(npu_inputs.at(SCALE_INDEX), expand_scale_shape, name_ + "_expand_scale");
    if (actual_scale == nullptr) {
      MS_LOG(ERROR) << "Change Scale op scale dims failed.";
      return RET_ERROR;
    }
  }
  scale_->set_input_scale(*actual_scale);

  // handle bias
  if (in_tensors.size() > BIAS_INDEX) {
    auto org_bias_tensor = in_tensors.at(BIAS_INDEX);
    ge::Operator *actual_bias = npu_inputs.at(BIAS_INDEX);
    if (org_bias_tensor.Shape().size() == DIMENSION_2D) {
      std::vector<int64_t> expand_bias_shape = org_bias_tensor.Shape();
      expand_bias_shape.emplace_back(1);
      actual_bias = ChangeDims(npu_inputs.at(BIAS_INDEX), expand_bias_shape, name_ + "_expand_bias");
      if (actual_bias == nullptr) {
        MS_LOG(ERROR) << "Change Scale op bias dims failed.";
        return RET_ERROR;
      }
    }
    scale_->set_input_bias(*actual_bias);
  }

  // restore to origin input shape
  if (need_expand_) {
    int64_t dims = org_input_shape.size();
    std::vector<int> valid_shape;
    for (int i = 0; i < dims; i++) {
      valid_shape.emplace_back(static_cast<int>(org_input_shape.at(i)));
    }
    auto valid_data_ptr = reinterpret_cast<const uint8_t *>(valid_shape.data());
    auto shape = GetNPUConst<int>(valid_data_ptr, {dims}, ge::DT_INT32, name_ + "_restore_shape");
    if (shape == nullptr) {
      MS_LOG(ERROR) << "Get NPU Const for shape restoration failed.";
      return RET_ERROR;
    }
    scale_ops_.emplace_back(shape);
    out_reshape_->set_input_x(*scale_);
    out_reshape_->set_input_shape(*shape);
  }
  return RET_OK;
}

ge::Operator *ScaleNPUOp::ChangeDims(const ge::Operator *input, std::vector<int64_t> dst_shape, std::string name,
                                     bool need_expand_4d) {
  MS_ASSERT(input != nullptr);
  auto reshape = new (std::nothrow) hiai::op::Reshape(name);
  if (reshape == nullptr) {
    MS_LOG(ERROR) << "New Reshape NPU operator failed.";
    return nullptr;
  }
  scale_ops_.emplace_back(reshape);
  MS_CHECK_LE(dst_shape.size(), NPU_SHAPE_SIZE, nullptr);
  int64_t actual_dim = need_expand_4d ? NPU_SHAPE_SIZE : dst_shape.size();
  std::vector<int> valid_shape(actual_dim, 1);
  for (int i = 0; i < dst_shape.size(); i++) {
    valid_shape[i] = static_cast<int>(dst_shape.at(i));
  }
  auto valid_data_ptr = reinterpret_cast<const uint8_t *>(valid_shape.data());
  auto shape = GetNPUConst<int>(valid_data_ptr, {actual_dim}, ge::DT_INT32, name_ + "_shape");
  if (shape == nullptr) {
    MS_LOG(ERROR) << "Get NPU Const for shape restoration failed.";
    return nullptr;
  }
  scale_ops_.emplace_back(shape);
  reshape->set_input_x(*input);
  reshape->set_input_shape(*shape);
  return reshape;
}

ScaleNPUOp::~ScaleNPUOp() {
  for (auto op : scale_ops_) {
    if (op != nullptr) {
      delete op;
      op = nullptr;
    }
  }
}
}  // namespace mindspore::lite
