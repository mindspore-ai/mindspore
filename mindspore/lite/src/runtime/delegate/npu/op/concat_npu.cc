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

#include "src/runtime/delegate/npu/op/concat_npu.h"
#include "src/runtime/delegate/npu/npu_converter_utils.h"

namespace mindspore {
int ConcatNPUOp::Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                      const std::vector<mindspore::MSTensor> &out_tensors) {
  concat_ = new (std::nothrow) hiai::op::ConcatD(name_);
  if (concat_ == nullptr) {
    MS_LOG(ERROR) << name_ << " op is nullptr";
    return RET_ERROR;
  }
  auto concat_prim = primitive->value_as_Concat();
  if (concat_prim == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }
  axis_ = concat_prim->axis();
  auto input_num = in_tensors.size();
  concat_->set_attr_N(input_num);
  concat_->create_dynamic_input_x(input_num);
  return RET_OK;
}

int ConcatNPUOp::SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                              const std::vector<mindspore::MSTensor> &out_tensors,
                              const std::vector<ge::Operator *> &npu_inputs,
                              const std::unordered_map<int, std::pair<ge::Operator *, int>> &index2_multi_out_index) {
  concat_->set_attr_concat_dim(axis_);
  for (auto pair : index2_multi_out_index) {
    auto in_op = pair.second.first;
    MS_CHECK_TRUE_RET(in_op != nullptr, RET_ERROR);
    concat_->SetInput(pair.first, *in_op, pair.second.second);
  }
  for (int i = 0; i < npu_inputs.size(); ++i) {
    if (index2_multi_out_index.find(i) == index2_multi_out_index.end()) {
      concat_->SetInput(i, *npu_inputs[i], 0);
    }
  }
  return RET_OK;
}

ge::Operator *ConcatNPUOp::GetNPUOp() { return this->concat_; }

int ConcatNPUOp::HandleAxisAndConstantInputs(std::vector<mindspore::MSTensor *> *all_tensors) {
  axis_ = TransFormAxis(axis_);
  if (axis_ == NCHW_INVALID) {
    MS_LOG(ERROR) << "Transform axis for concat op failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

ConcatNPUOp::~ConcatNPUOp() {
  if (concat_ != nullptr) {
    delete concat_;
    concat_ = nullptr;
  }
}
}  // namespace mindspore
