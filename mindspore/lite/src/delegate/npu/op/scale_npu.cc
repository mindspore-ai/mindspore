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

#include "src/delegate/npu/op/scale_npu.h"
#include <memory>
#include "src/delegate/npu/npu_converter_utils.h"

namespace mindspore {
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
  axis_ = scale_prim->axis();
  if (axis_ < 0) {
    axis_ = axis_ + in_tensors[INPUT_INDEX].Shape().size();
  }
  if (axis_ != NHWC_C && axis_ != NCHW_C) {
    if (in_tensors.size() <= BIAS_INDEX) {
      MS_LOG(INFO) << "Npu Scale op does not support axis: " << axis_ << ", try to convert to Mul op.";
      use_mul_ = true;
    } else {
      MS_LOG(WARNING) << "Npu Scale axis attr only support 1 or channel, now is " << axis_;
      return RET_NOT_SUPPORT;
    }
  }
  return RET_OK;
}

int ScaleNPUOp::Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                     const std::vector<mindspore::MSTensor> &out_tensors) {
  if (!use_mul_) {
    // note that Scale only support the default axis(i.e., 1), setting axis is meaningless.
    op_ = new (std::nothrow) hiai::op::Scale(name_);
  } else {
    op_ = new (std::nothrow) hiai::op::Mul(name_);
  }
  if (op_ == nullptr) {
    MS_LOG(ERROR) << name_ << " op is nullptr";
    return RET_ERROR;
  }

  auto scale_prim = primitive->value_as_ScaleFusion();
  if (scale_prim == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }
  act_type_ = scale_prim->activation_type();
  if (act_type_ != schema::ActivationType_NO_ACTIVATION) {
    auto ret = SetActivation(op_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "New activation npu operator for op " << name_ << " failed.";
      return ret;
    }
  }
  return RET_OK;
}

int ScaleNPUOp::SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                             const std::vector<mindspore::MSTensor> &out_tensors,
                             const std::vector<ge::Operator *> &npu_inputs) {
  MS_ASSERT(in_tensors.size() > SCALE_INDEX);
  if (use_mul_) {
    auto ret = ConvertScaleToMul(npu_inputs, op_, in_tensors);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Convert Scale to Mul failed, op name: " << name_;
    }
    return ret;
  }

  auto scale_op = reinterpret_cast<hiai::op::Scale *>(op_);
  scale_op->set_input_x(*npu_inputs.at(INPUT_INDEX));
  scale_op->set_input_scale(*npu_inputs.at(SCALE_INDEX));
  if (in_tensors.size() > BIAS_INDEX && in_tensors[BIAS_INDEX] != nullptr) {
    scale_op->set_input_bias(*npu_inputs.at(BIAS_INDEX));
  }
  return RET_OK;
}

ge::Operator *ScaleNPUOp::GetNPUOp() {
  if (act_type_ == schema::ActivationType_NO_ACTIVATION) {
    return op_;
  } else {
    return act_;
  }
}

int ScaleNPUOp::SetActivation(const ge::Operator *input) {
  act_ = new (std::nothrow) hiai::op::Activation(name_ + "_act");
  if (act_ == nullptr) {
    MS_LOG(ERROR) << "New activation npu operator for op " << name_ << " failed.";
    return RET_ERROR;
  }
  act_->set_input_x(*input);

  auto act_mode = ConverterToNPUActivationMode(act_type_);
  if (act_mode == ACTIVATION_INVALID) {
    MS_LOG(ERROR) << "Unsupported activation type for scale op " << name_;
    return RET_ERROR;
  }
  act_->set_attr_mode(act_mode);
  return RET_OK;
}

int ScaleNPUOp::ConvertScaleToMul(const std::vector<ge::Operator *> &npu_inputs, ge::Operator *cur_op,
                                  const std::vector<mindspore::MSTensor> &in_tensors) {
  auto input_shape = in_tensors[INPUT_INDEX].Shape();
  auto scale_shape = in_tensors[SCALE_INDEX].Shape();
  auto mul_op = reinterpret_cast<hiai::op::Mul *>(cur_op);
  mul_op->set_input_x1(*npu_inputs.at(INPUT_INDEX));
  if (input_shape.size() == scale_shape.size()) {
    mul_op->set_input_x2(*npu_inputs.at(SCALE_INDEX));
  } else {
    int valid_shape[4] = {1, 1, 1, 1};
    for (size_t i = 0; i < scale_shape.size(); i++) {
      valid_shape[axis_ + i] = static_cast<int>(scale_shape[i]);
    }
    reshape_ = new (std::nothrow) hiai::op::Reshape(name_ + "_reshape");
    if (reshape_ == nullptr) {
      MS_LOG(ERROR) << "New Reshape npu operator for op " << name_ << " failed.";
      return RET_ERROR;
    }
    std::shared_ptr<ge::Tensor> shape_tensor = std::make_shared<ge::Tensor>();
    if (shape_tensor == nullptr) {
      MS_LOG(ERROR) << "new shape_tensor failed.";
      return RET_ERROR;
    }
    ge::TensorDesc tensor_desc(ge::Shape({NPU_SHAPE_SIZE}), ge::FORMAT_ND, ge::DT_INT32);
    shape_tensor->SetTensorDesc(tensor_desc);
    shape_tensor->SetData(reinterpret_cast<const uint8_t *>(valid_shape), NPU_SHAPE_SIZE * sizeof(int));
    shape_ = new (std::nothrow) hiai::op::Const(name_ + "_reshape_1");
    if (shape_ == nullptr) {
      MS_LOG(ERROR) << "New shape const for op " << name_ << " failed.";
      return RET_ERROR;
    }
    shape_->set_attr_value(shape_tensor);
    reshape_->set_input_x(*npu_inputs.at(SCALE_INDEX));
    reshape_->set_input_shape(*shape_);
    mul_op->set_input_x2(*reshape_);
  }
  return RET_OK;
}

ScaleNPUOp::~ScaleNPUOp() {
  if (op_ != nullptr) {
    delete op_;
    op_ = nullptr;
  }
  if (scale_ != nullptr) {
    delete scale_;
    scale_ = nullptr;
  }
  if (bias_ != nullptr) {
    delete bias_;
    bias_ = nullptr;
  }
  if (act_ != nullptr) {
    delete act_;
    act_ = nullptr;
  }
  if (reshape_ != nullptr) {
    delete reshape_;
    reshape_ = nullptr;
  }
  if (shape_ != nullptr) {
    delete shape_;
    shape_ = nullptr;
  }
}
}  // namespace mindspore
