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

#include "src/litert/delegate/npu/op/pad_npu.h"
#include <memory>
#include "src/litert/delegate/npu/npu_converter_utils.h"

namespace mindspore::lite {
constexpr int PAD_PAIR_SIZE = 2;
constexpr int PAD_SIZE = 8;
constexpr int PAD_INPUT_SIZE = 2;
constexpr int PAD_EXPAND_DIM = 2;

int PadNPUOp::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                        const std::vector<mindspore::MSTensor> &out_tensors) {
  auto pad_prim = primitive->value_as_PadFusion();
  if (pad_prim == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }
  if (pad_prim->padding_mode() != schema::PaddingMode_CONSTANT) {
    MS_LOG(WARNING) << "NPU only support CONSTANT padding mode";
    return RET_NOT_SUPPORT;
  }
  if (pad_prim->paddings() != nullptr) {
    return RET_OK;
  }
  if (in_tensors.size() >= PAD_INPUT_SIZE && in_tensors[1].Data() != nullptr) {
    return RET_OK;
  }
  MS_LOG(WARNING) << "NPU pad only support constant pad size.";
  return RET_ERROR;
}

int PadNPUOp::Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                   const std::vector<mindspore::MSTensor> &out_tensors) {
  pad_ = new (std::nothrow) hiai::op::PadV2(name_);
  if (pad_ == nullptr) {
    MS_LOG(ERROR) << name_ << " op is nullptr";
    return RET_ERROR;
  }
  auto pad_prim = primitive->value_as_PadFusion();
  if (pad_prim == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }
  if (pad_prim->paddings() != nullptr) {
    auto fb_paddings = pad_prim->paddings()->data();
    if (fb_paddings == nullptr) {
      MS_LOG(ERROR) << "paddings is nullptr";
      return RET_ERROR;
    }
    for (auto fb_padding : *fb_paddings) {
      auto paddings_data = fb_padding->data();
      if (paddings_data == nullptr) {
        MS_LOG(ERROR) << "paddings_data is nullptr";
        return RET_ERROR;
      }
      auto paddings = std::vector<int64_t>(paddings_data->begin(), paddings_data->end());
      paddings_vec_.insert(paddings_vec_.end(), paddings.begin(), paddings.end());
    }
  } else if (in_tensors.size() >= PAD_INPUT_SIZE && in_tensors[1].Data() != nullptr) {
    for (int i = 0; i < in_tensors[1].ElementNum(); i++) {
      paddings_vec_.push_back(static_cast<const int *>(in_tensors[1].Data().get())[i]);
    }
  } else {
    MS_LOG(ERROR) << "NPU pad only support constant pad size.";
    return RET_ERROR;
  }

  ge::TensorDesc constant_values_tensor_desc(ge::Shape({1}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  ge::TensorPtr constant_values_tensor = std::make_shared<hiai::Tensor>(constant_values_tensor_desc);
  std::vector<float> constant_values_data_value = {pad_prim->constant_value()};
  constant_values_tensor->SetData(reinterpret_cast<uint8_t *>(constant_values_data_value.data()), 1 * sizeof(float));
  constant_value_ = new (std::nothrow) hiai::op::Const(name_ + "constant");
  if (constant_value_ == nullptr) {
    MS_LOG(ERROR) << "create const NPU op failed for " << name_;
    return RET_ERROR;
  }
  constant_value_->set_attr_value(constant_values_tensor);
  pad_->set_input_constant_values(*constant_value_);
  return RET_OK;
}

int PadNPUOp::SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                           const std::vector<mindspore::MSTensor> &out_tensors,
                           const std::vector<ge::Operator *> &npu_inputs) {
  int size = static_cast<int>(paddings_vec_.size());
  ge::TensorDesc padding_tensor_desc(ge::Shape({size / PAD_PAIR_SIZE, PAD_PAIR_SIZE}), ge::FORMAT_NCHW, ge::DT_INT32);
  ge::TensorPtr padding_tensor = std::make_shared<hiai::Tensor>(padding_tensor_desc);
  padding_tensor->SetData(reinterpret_cast<uint8_t *>(paddings_vec_.data()), size * sizeof(int));
  paddings_ = new (std::nothrow) hiai::op::Const(name_ + "paddings");
  if (paddings_ == nullptr) {
    MS_LOG(ERROR) << "create padding_tensor const NPU op failed for " << name_;
    return RET_ERROR;
  }
  paddings_->set_attr_value(padding_tensor);
  pad_->set_input_paddings(*paddings_);
  CHECK_LESS_RETURN(npu_inputs.size(), 1);
  pad_->set_input_x(*npu_inputs[0]);
  return RET_OK;
}

ge::Operator *PadNPUOp::GetNPUOp() { return this->pad_; }

int PadNPUOp::HandleAxisAndConstantInputs(std::vector<mindspore::MSTensor *> *all_tensors) {
  if (paddings_vec_.size() != PAD_SIZE) {
    return RET_ERROR;
  }
  int c1 = paddings_vec_[NHWC_C * PAD_EXPAND_DIM];
  int c2 = paddings_vec_[NHWC_C * PAD_EXPAND_DIM + 1];
  // 0 1 2 3 4 5 6 7
  // n n h h w w c c
  // n n c c h h w w
  paddings_vec_[NCHW_H * PAD_EXPAND_DIM] = paddings_vec_[NHWC_H * PAD_EXPAND_DIM];
  paddings_vec_[NCHW_H * PAD_EXPAND_DIM + 1] = paddings_vec_[NHWC_H * PAD_EXPAND_DIM + 1];
  paddings_vec_[NCHW_W * PAD_EXPAND_DIM] = paddings_vec_[NHWC_W * PAD_EXPAND_DIM];
  paddings_vec_[NCHW_W * PAD_EXPAND_DIM + 1] = paddings_vec_[NHWC_W * PAD_EXPAND_DIM + 1];
  paddings_vec_[NCHW_C * PAD_EXPAND_DIM] = c1;
  paddings_vec_[NCHW_C * PAD_EXPAND_DIM + 1] = c2;
  return RET_OK;
}

PadNPUOp::~PadNPUOp() {
  if (pad_ != nullptr) {
    delete pad_;
    pad_ = nullptr;
  }
  if (paddings_ != nullptr) {
    delete paddings_;
    paddings_ = nullptr;
  }
  if (constant_value_ != nullptr) {
    delete constant_value_;
    constant_value_ = nullptr;
  }
}
}  // namespace mindspore::lite
