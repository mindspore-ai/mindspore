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

#include "src/control_flow/actor/entrance_actor.h"
#include <algorithm>
#include "mindrt/include/mindrt.hpp"
#include "src/common/tensor_util.h"
#include "src/runtime/inner_allocator.h"

namespace {
const constexpr int kToExitIndex = 0;
}
namespace mindspore::lite {
void LiteEntranceOpActor::RunOpData(OpData<Tensor> *inputs, OpContext<Tensor> *context) {
  input_actor_id_data_[inputs->op_id_].push_back(inputs);
  if (input_actor_id_data_[inputs->op_id_].size() < kernel_->in_tensors().size()) {
    return;
  }

  entrance_input_aid_ = input_actor_id_data_[inputs->op_id_].front()->op_id_;
  for (auto item : input_actor_id_data_[inputs->op_id_]) {
    inputs_data_[item->index_] = item->data_;
  }

  InitInputData();
  input_actor_id_data_[inputs->op_id_].clear();
  AsyncOutput(context);
  SetOutputData(context);
  return;
}

int LiteEntranceOpActor::InitInputData() {
  auto ret = SetInputShape();

  for (size_t i = 0; i < inputs_data_.size(); ++i) {
    auto dst_tensor = kernel_->out_tensors()[i + 1];
    auto src_tensor = inputs_data_[i];
    dst_tensor->set_data_type(src_tensor->data_type());
    if (src_tensor->allocator() == nullptr || src_tensor->IsGraphInput()) {
      (void)SetTensorData(dst_tensor, src_tensor);
    } else {
      (void)MoveTensorData(dst_tensor, src_tensor);
    }
  }
  return ret;
}

int LiteEntranceOpActor::SetInputShape() {
  auto ret = RET_OK;
  for (size_t i = 0; i < inputs_data_.size(); ++i) {
    auto &output_tensor = kernel_->out_tensors()[i + 1];
    if (output_tensor->shape() == inputs_data_[i]->shape()) {
      continue;
    }
    ret = SetTensorShape(output_tensor, inputs_data_[i]);
    MS_CHECK_FALSE_MSG(ret != RET_OK, ret, "set input shape failed.");
  }
  return ret;
}

void LiteEntranceOpActor::AsyncOutput(OpContext<Tensor> *context) {
  // modify aid of data which is sending to exit actor
  to_exit_acotr_data_->op_id_ = entrance_input_aid_;

  for (size_t i = 0; i < output_data_arrows_.size(); i++) {
    auto data = outputs_data_.at(i);
    Async(output_data_arrows_[i]->to_op_id_, get_actor_mgr(), &mindspore::OpActor<Tensor>::RunOpData, data.get(),
          context);
  }
}

int LiteEntranceOpActor::PrepareOutputData() {
  // entrance actor has not calculating, so send input directly.
  outputs_data_.resize(output_data_arrows_.size());
  for (size_t i = 0; i < output_data_arrows_.size(); i++) {
    auto &arrow = output_data_arrows_[i];
    auto data = std::make_shared<OpData<Tensor>>(this->GetAID(), (kernel_->out_tensors()).at(arrow->from_output_index_),
                                                 static_cast<int>(arrow->to_input_index_));
    if (data == nullptr) {
      MS_LOG(ERROR) << "new output_data failed.";
      return RET_NULL_PTR;
    }
    if (arrow->from_output_index_ == kToExitIndex) {
      to_exit_acotr_data_ = data;
    }
    outputs_data_.at(i) = data;
  }
  return RET_OK;
}
}  // namespace mindspore::lite
