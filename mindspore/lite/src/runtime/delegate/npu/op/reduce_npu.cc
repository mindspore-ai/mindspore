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

#include "src/runtime/delegate/npu/op/reduce_npu.h"
#include "src/runtime/delegate/npu/npu_converter_utils.h"

namespace mindspore {
int ReduceNPUOp::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                           const std::vector<mindspore::MSTensor> &out_tensors) {
  auto reduce_prim = primitive->value_as_ReduceFusion();
  if (reduce_prim == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op: " << name_;
    return RET_ERROR;
  }
  CHECK_LESS_RETURN(in_tensors.size(), kInputSize1);
  auto reduce_axes = inputs_.at(1);
  if (!reduce_axes.IsConst()) {
    MS_LOG(WARNING) << "Npu Reduce op dose not support non-const axes.";
    return RET_NOT_SUPPORT;
  }
  reduce_mode_ = reduce_prim->mode();
  if (reduce_mode_ != schema::ReduceMode_ReduceMean && reduce_mode_ != schema::ReduceMode_ReduceSum) {
    MS_LOG(WARNING) << "Npu does not support reduce mode " << reduce_prim->mode() << " for op " << name_;
    return RET_NOT_SUPPORT;
  }
  if (reduce_prim->reduce_to_end()) {
    MS_LOG(WARNING) << "Npu Reduce op does not support attribute reduce_to_end";
    return RET_NOT_SUPPORT;
  }
  return RET_OK;
}

int ReduceNPUOp::Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                      const std::vector<mindspore::MSTensor> &out_tensors) {
  auto reduce_prim = primitive->value_as_ReduceFusion();
  if (reduce_prim == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }
  if (reduce_mode_ == schema::ReduceMode_ReduceMean) {
    auto reduce_mean = new (std::nothrow) hiai::op::ReduceMean(name_);
    if (reduce_mean == nullptr) {
      MS_LOG(ERROR) << "New reduce operator for op " << name_ << " failed.";
      return RET_ERROR;
    }
    reduce_mean->set_attr_keep_dims(reduce_prim->keep_dims());
    reduce_ = reduce_mean;
  } else if (reduce_mode_ == schema::ReduceMode_ReduceSum) {
    auto reduce_sum = new (std::nothrow) hiai::op::ReduceSum(name_);
    if (reduce_sum == nullptr) {
      MS_LOG(ERROR) << "New reduce operator for op " << name_ << " failed.";
      return RET_ERROR;
    }
    reduce_sum->set_attr_keep_dims(reduce_prim->keep_dims());
    reduce_ = reduce_sum;
  } else {
    MS_LOG(ERROR) << "Npu does not support reduce mode " << reduce_prim->mode() << " for op " << name_;
    return RET_ERROR;
  }
  return RET_OK;
}

int ReduceNPUOp::SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                              const std::vector<mindspore::MSTensor> &out_tensors,
                              const std::vector<ge::Operator *> &npu_inputs) {
  if (reduce_mode_ == schema::ReduceMode_ReduceMean) {
    auto reduce_mean = reinterpret_cast<hiai::op::ReduceMean *>(reduce_);
    reduce_mean->set_input_x(*npu_inputs[0]).set_input_axes(*npu_inputs[1]);
  } else if (reduce_mode_ == schema::ReduceMode_ReduceSum) {
    auto reduce_sum = reinterpret_cast<hiai::op::ReduceSum *>(reduce_);
    reduce_sum->set_input_x(*npu_inputs[0]).set_input_axes(*npu_inputs[1]);
  }
  return RET_OK;
}

ge::Operator *ReduceNPUOp::GetNPUOp() { return this->reduce_; }

int ReduceNPUOp::HandleAxisAndConstantInputs(std::vector<mindspore::MSTensor *> *all_tensors) {
  auto reduce_axes = inputs_.at(1);
  int num_axes = reduce_axes.Shape().at(0);
  MS_CHECK_TRUE_RET(reduce_axes.MutableData() != nullptr, RET_ERROR);
  int *axes_data = reinterpret_cast<int *>(reduce_axes.MutableData());
  for (int i = 0; i < num_axes; i++) {
    axes_data[i] = TransFormAxis(axes_data[i]);
    if (axes_data[i] == NCHW_INVALID) {
      MS_LOG(ERROR) << "Transform axis for Reduce op failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

ReduceNPUOp::~ReduceNPUOp() {
  if (reduce_ != nullptr) {
    delete reduce_;
    reduce_ = nullptr;
  }
}
}  // namespace mindspore
