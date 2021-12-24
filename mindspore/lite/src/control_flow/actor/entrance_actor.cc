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

void LiteEntranceOpActor::InitInputData() {
  SetInputShape();

  for (size_t i = 0; i < inputs_data_.size(); ++i) {
    auto dst_tensor = kernel_->out_tensors()[i + 1];
    auto src_tensor = inputs_data_[i];
    if (dst_tensor->init_ref_count() == 0) {
      src_tensor->DecRefCount();
      continue;
    }

    if (NeedCastData(dst_tensor, src_tensor)) {
      CastInputData(dst_tensor, src_tensor);
      continue;
    }

    /* same data-type  */
    if (src_tensor->allocator() == nullptr || src_tensor->IsGraphInput()) {
      // delegate graph kernel output tensor
      SetInputData(dst_tensor, src_tensor);
    } else {
      MoveInputData(dst_tensor, src_tensor);
    }
  }
  return;
}

void LiteEntranceOpActor::SetInputShape() {
  for (size_t i = 0; i < inputs_data_.size(); ++i) {
    auto &output_tensor = kernel_->out_tensors()[i + 1];
    if (output_tensor->shape() == inputs_data_[i]->shape()) {
      continue;
    }
    MS_LOG(DEBUG) << "inputs_data_[" << i << "].shape: " << inputs_data_[i]->shape() << " vs kernel_->out_tensors()["
                  << i << "].shape: " << kernel_->out_tensors()[i]->shape() << " are not equal.";
    MS_LOG(DEBUG) << "this->kernel_->name(): " << this->kernel_->name();

    if (output_tensor->data_type() == kObjectTypeTensorType) {
#ifndef CONTROLFLOW_TENSORLIST_CLIP
      auto input_tensorlist = reinterpret_cast<TensorList *>(output_tensor);
      auto input_data_tensorlist = reinterpret_cast<TensorList *>(inputs_data_[i]);
      input_tensorlist->FreeTensorListData();
      input_tensorlist->set_element_shape(input_data_tensorlist->element_shape());
      input_tensorlist->set_shape(input_data_tensorlist->shape());
      std::vector<std::vector<int>> tensor_shape{};
      std::transform(input_data_tensorlist->tensors().begin(), input_data_tensorlist->tensors().end(),
                     std::back_inserter(tensor_shape), [](const Tensor *tensor_item) { return tensor_item->shape(); });
      input_tensorlist->MallocTensorListData(input_data_tensorlist->tensors_data_type(), tensor_shape);
#endif
    } else {
      output_tensor->set_shape(inputs_data_[i]->shape());
      output_tensor->set_format(inputs_data_[i]->format());
    }
  }
}

void LiteEntranceOpActor::AsyncOutput(OpContext<Tensor> *context) {
  // modify aid of data which is sending to exit actor
  to_exit_acotr_data_->op_id_ = entrance_input_aid_;

  for (size_t i = 0; i < output_data_arrows_.size(); i++) {
    auto data = outputs_data_.at(i);
    Async(output_data_arrows_[i]->to_op_id_, &mindspore::OpActor<Tensor>::RunOpData, data.get(), context);
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
