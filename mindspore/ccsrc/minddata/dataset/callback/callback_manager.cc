/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/callback/callback_manager.h"
#include "minddata/dataset/callback/ds_callback.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/engine/datasetops/dataset_op.h"

namespace mindspore {
namespace dataset {

void CallbackManager::AddCallbacks(std::vector<std::shared_ptr<DSCallback>> callbacks) {
  callbacks_.insert(callbacks_.end(), callbacks.begin(), callbacks.end());
}

Status CallbackManager::Init(DatasetOp *op) {
  RETURN_UNEXPECTED_IF_NULL(op);
  op_ = op;
  // turn the flag on if callback is set
  enabled_ = !callbacks_.empty();

  // error check for each of the callbacks
  for (auto &cb : callbacks_) {
    CHECK_FAIL_RETURN_UNEXPECTED(cb->step_size() > 0, "callback step_size needs to be greater than 0.");
  }

  return Status::OK();
}

Status CallbackManager::Begin(const CallbackParam &cb_param) {
  RETURN_OK_IF_TRUE(!enabled_);
  RETURN_UNEXPECTED_IF_NULL(op_);
  std::vector<size_t> callback_inds;
  // go through all callback functions to see if each function is needed
  for (size_t ind = 0; ind < callbacks_.size(); ind++) {
    if (callbacks_[ind]->IsBeginNeeded()) callback_inds.push_back(ind);
  }
  // return Status::OK() if no begin is needed
  RETURN_OK_IF_TRUE(callback_inds.empty());

  RETURN_IF_NOT_OK(op_->WaitForWorkers());

  // Now do the actual callback
  for (size_t ind : callback_inds) {
    RETURN_IF_NOT_OK(callbacks_[ind]->DSBegin(cb_param));
  }
  return Status::OK();
}

Status CallbackManager::EpochBegin(const CallbackParam &cb_param) {
  RETURN_OK_IF_TRUE(!enabled_);
  RETURN_UNEXPECTED_IF_NULL(op_);
  std::vector<size_t> callback_inds;
  // go through all callback functions to see if each function is needed
  for (size_t ind = 0; ind < callbacks_.size(); ind++) {
    if (callbacks_[ind]->IsEpochBeginNeeded()) callback_inds.push_back(ind);
  }
  // return Status::OK() if no epoch_begin is needed
  RETURN_OK_IF_TRUE(callback_inds.empty());

  RETURN_IF_NOT_OK(op_->WaitForWorkers());

  // Now do the actual callback
  for (size_t ind : callback_inds) {
    RETURN_IF_NOT_OK(callbacks_[ind]->DSEpochBegin(cb_param));
  }
  return Status::OK();
}

Status CallbackManager::StepBegin(const CallbackParam &cb_param) {
  RETURN_OK_IF_TRUE(!enabled_);
  RETURN_UNEXPECTED_IF_NULL(op_);
  std::vector<size_t> callback_inds;
  // go through all callback functions to see if each function is needed
  for (size_t ind = 0; ind < callbacks_.size(); ind++) {
    if (callbacks_[ind]->IsNStepBeginNeeded() && (cb_param.cur_epoch_step_num_ - 1) % callbacks_[ind]->step_size() == 0)
      callback_inds.push_back(ind);
  }
  // return Status::OK() if no step_begin is needed
  RETURN_OK_IF_TRUE(callback_inds.empty());

  RETURN_IF_NOT_OK(op_->WaitForWorkers());

  // Now do the actual callback
  for (size_t ind : callback_inds) {
    RETURN_IF_NOT_OK(callbacks_[ind]->DSNStepBegin(cb_param));
  }
  return Status::OK();
}

Status CallbackManager::End(const CallbackParam &cb_param) {
  RETURN_OK_IF_TRUE(!enabled_);
  RETURN_UNEXPECTED_IF_NULL(op_);
  std::vector<size_t> callback_inds;
  // go through all callback functions to see if each function is needed
  for (size_t ind = 0; ind < callbacks_.size(); ind++) {
    if (callbacks_[ind]->IsEndNeeded()) callback_inds.push_back(ind);
  }
  // return Status::OK() if no end is needed
  RETURN_OK_IF_TRUE(callback_inds.empty());

  RETURN_IF_NOT_OK(op_->WaitForWorkers());

  // Now do the actual callback
  for (size_t ind : callback_inds) {
    RETURN_IF_NOT_OK(callbacks_[ind]->DSEnd(cb_param));
  }
  return Status::OK();
}

Status CallbackManager::EpochEnd(const CallbackParam &cb_param) {
  RETURN_OK_IF_TRUE(!enabled_);
  RETURN_UNEXPECTED_IF_NULL(op_);
  std::vector<size_t> callback_inds;
  // go through all callback functions to see if each function is needed
  for (size_t ind = 0; ind < callbacks_.size(); ind++) {
    if (callbacks_[ind]->IsEpochEndNeeded()) callback_inds.push_back(ind);
  }
  // return Status::OK() if no epoch_end is needed
  RETURN_OK_IF_TRUE(callback_inds.empty());

  RETURN_IF_NOT_OK(op_->WaitForWorkers());

  // Now do the actual callback
  for (size_t ind : callback_inds) {
    RETURN_IF_NOT_OK(callbacks_[ind]->DSEpochEnd(cb_param));
  }
  return Status::OK();
}

Status CallbackManager::StepEnd(const CallbackParam &cb_param) {
  RETURN_OK_IF_TRUE(!enabled_);
  RETURN_UNEXPECTED_IF_NULL(op_);
  std::vector<size_t> callback_inds;
  // go through all callback functions to see if each function is needed
  for (size_t ind = 0; ind < callbacks_.size(); ind++) {
    if (callbacks_[ind]->IsNStepEndNeeded() && (cb_param.cur_epoch_step_num_ - 1) % callbacks_[ind]->step_size() == 0)
      callback_inds.push_back(ind);
  }
  // return Status::OK() if no step_end is needed
  RETURN_OK_IF_TRUE(callback_inds.empty());

  RETURN_IF_NOT_OK(op_->WaitForWorkers());

  // Now do the actual callback
  for (size_t ind : callback_inds) {
    RETURN_IF_NOT_OK(callbacks_[ind]->DSNStepEnd(cb_param));
  }
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
