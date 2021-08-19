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

#include "src/delegate/npu/op/batchnorm_npu.h"
#include "include/graph/op/all_ops.h"
#include "src/delegate/npu/npu_converter_utils.h"

namespace mindspore {
constexpr int SCALE_INDEX = 1;
constexpr int OFFSET_INDEX = 2;
constexpr int MEAN_INDEX = 3;
constexpr int VARIANCE_INDEX = 4;

int BatchnormNPUOp::Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                         const std::vector<mindspore::MSTensor> &out_tensors) {
  batchnorm_ = new (std::nothrow) ge::op::BatchNormExt2(name_);
  if (batchnorm_ == nullptr) {
    MS_LOG(ERROR) << "New batchnorm npu operator for batchnorm op " << name_ << " failed.";
    return RET_ERROR;
  }
  auto batchnorm_prim = primitive->value_as_FusedBatchNorm();
  if (batchnorm_prim == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }
  batchnorm_->set_attr_epsilon(batchnorm_prim->epsilon());
  batchnorm_->set_attr_momentum(batchnorm_prim->momentum());
  batchnorm_->set_attr_mode(1);
  return RET_OK;
}

int BatchnormNPUOp::SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                                 const std::vector<mindspore::MSTensor> &out_tensors,
                                 const std::vector<ge::Operator *> &npu_inputs) {
  batchnorm_->set_input_x(*npu_inputs[0]);
  scale_ = new (std::nothrow) hiai::op::Const(name_ + "_scale");
  if (scale_ == nullptr) {
    MS_LOG(ERROR) << "New scale const failed.";
    return RET_ERROR;
  }
  auto scale_tensor = ConverterToNPUTensor(in_tensors[SCALE_INDEX]);
  scale_->set_attr_value(scale_tensor);
  batchnorm_->set_input_scale(*scale_);

  offset_ = new (std::nothrow) hiai::op::Const(name_ + "_offset");
  if (offset_ == nullptr) {
    MS_LOG(ERROR) << "New offset const failed.";
    return RET_ERROR;
  }
  auto offset_tensor = ConverterToNPUTensor(in_tensors[OFFSET_INDEX]);
  offset_->set_attr_value(offset_tensor);
  batchnorm_->set_input_offset(*offset_);

  mean_ = new (std::nothrow) hiai::op::Const(name_ + "_mean");
  if (mean_ == nullptr) {
    MS_LOG(ERROR) << "New mean const failed.";
    return RET_ERROR;
  }
  auto mean_tensor = ConverterToNPUTensor(in_tensors[MEAN_INDEX]);
  mean_->set_attr_value(mean_tensor);
  batchnorm_->set_input_mean(*mean_);

  variance_ = new (std::nothrow) hiai::op::Const(name_ + "_variance");
  if (variance_ == nullptr) {
    MS_LOG(ERROR) << "New variance const failed.";
    return RET_ERROR;
  }
  auto variance_tensor = ConverterToNPUTensor(in_tensors[VARIANCE_INDEX]);
  variance_->set_attr_value(variance_tensor);
  batchnorm_->set_input_variance(*variance_);
  return RET_OK;
}

ge::Operator *BatchnormNPUOp::GetNPUOp() { return batchnorm_; }

BatchnormNPUOp::~BatchnormNPUOp() {
  if (batchnorm_ != nullptr) {
    delete batchnorm_;
    batchnorm_ = nullptr;
  }
  if (scale_ != nullptr) {
    delete scale_;
    scale_ = nullptr;
  }
  if (offset_ != nullptr) {
    delete offset_;
    offset_ = nullptr;
  }
  if (mean_ != nullptr) {
    delete mean_;
    mean_ = nullptr;
  }
  if (variance_ != nullptr) {
    delete variance_;
    variance_ = nullptr;
  }
}
}  // namespace mindspore
