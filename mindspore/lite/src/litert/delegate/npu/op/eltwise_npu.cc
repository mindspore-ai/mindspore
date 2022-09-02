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

#include "src/litert/delegate/npu/op/eltwise_npu.h"
#include "include/graph/op/all_ops.h"
#include "src/litert/kernel_registry.h"
#include "src/litert/delegate/npu/npu_converter_utils.h"

namespace mindspore::lite {
int EltwiseNPUOp::Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                       const std::vector<mindspore::MSTensor> &out_tensors) {
  eltwise_ = new (std::nothrow) hiai::op::Eltwise(name_);
  if (eltwise_ == nullptr) {
    MS_LOG(ERROR) << name_ << " op is nullptr";
    return RET_ERROR;
  }
  auto eltwise_prim = primitive->value_as_Eltwise();
  if (eltwise_prim == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }
  eltwise_->set_attr_mode(ConverterToNPUEltwiseMode(eltwise_prim->mode()));
  auto input_num = in_tensors.size();
  eltwise_->create_dynamic_input_x(input_num);
  eltwise_->set_attr_N(input_num);
  return RET_OK;
}

int EltwiseNPUOp::SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                               const std::vector<mindspore::MSTensor> &out_tensors,
                               const std::vector<ge::Operator *> &npu_inputs,
                               const std::unordered_map<int, std::pair<ge::Operator *, int>> &index2_multi_out_index) {
  for (auto pair : index2_multi_out_index) {
    auto in_op = pair.second.first;
    MS_CHECK_TRUE_RET(in_op != nullptr, RET_ERROR);
    eltwise_->SetInput(pair.first, *in_op, pair.second.second);
  }
  for (int i = 0; i < npu_inputs.size(); ++i) {
    if (index2_multi_out_index.find(i) == index2_multi_out_index.end()) {
      CHECK_LESS_RETURN(npu_inputs.size(), 1);
      eltwise_->SetInput(i, *npu_inputs[i], 0);
    }
  }
  return RET_OK;
}

ge::Operator *EltwiseNPUOp::GetNPUOp() { return this->eltwise_; }

EltwiseNPUOp::~EltwiseNPUOp() {
  if (eltwise_ != nullptr) {
    delete eltwise_;
    eltwise_ = nullptr;
  }
}
}  // namespace mindspore::lite
