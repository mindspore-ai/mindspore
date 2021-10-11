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

#include "src/delegate/npu/op/reshape_npu.h"
#include "include/graph/op/all_ops.h"
#include "src/delegate/npu/npu_converter_utils.h"
namespace mindspore {
int ReshapeNPUOp::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                            const std::vector<mindspore::MSTensor> &out_tensors) {
  if (in_tensors.size() != 2) {
    MS_LOG(WARNING) << "Npu op should have w2 input tensors.";
    return RET_NOT_SUPPORT;
  }
  auto shape_tensor = in_tensors.at(1);
  if (shape_tensor.Data() == nullptr) {
    MS_LOG(WARNING) << "Npu reshape op only supports const shape.";
    return RET_NOT_SUPPORT;
  }
  return RET_OK;
}

int ReshapeNPUOp::Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                       const std::vector<mindspore::MSTensor> &out_tensors) {
  reshape_ = new (std::nothrow) hiai::op::Reshape(name_);
  if (reshape_ == nullptr) {
    MS_LOG(ERROR) << name_ << " op is nullptr";
    return RET_ERROR;
  }
  return RET_OK;
}

int ReshapeNPUOp::SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                               const std::vector<mindspore::MSTensor> &out_tensors,
                               const std::vector<ge::Operator *> &npu_inputs) {
  reshape_->set_input_x(*npu_inputs[0]);
  reshape_->set_input_shape(*npu_inputs[1]);
  return RET_OK;
}

int ReshapeNPUOp::SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                               const std::vector<mindspore::MSTensor> &out_tensors,
                               const std::vector<ge::Operator *> &npu_inputs,
                               const std::unordered_map<int, std::pair<ge::Operator *, int>> &index2_multi_out_index) {
  for (auto pair : index2_multi_out_index) {
    auto in_op = pair.second.first;
    MS_CHECK_TRUE_RET(in_op != nullptr, RET_ERROR);
    reshape_->SetInput(pair.first, *in_op, pair.second.second);
  }
  for (int i = 0; i < npu_inputs.size(); ++i) {
    if (index2_multi_out_index.find(i) == index2_multi_out_index.end()) {
      reshape_->SetInput(i, *npu_inputs[i], 0);
    }
  }
  return RET_OK;
}

ge::Operator *ReshapeNPUOp::GetNPUOp() { return this->reshape_; }

ReshapeNPUOp::~ReshapeNPUOp() {
  if (reshape_ != nullptr) {
    delete reshape_;
    reshape_ = nullptr;
  }
}
}  // namespace mindspore
