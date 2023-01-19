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

#include "minddata/dataset/callback/callback_manager.h"
#include "minddata/dataset/callback/ds_callback.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/engine/datasetops/dataset_op.h"

namespace mindspore {
namespace dataset {

void CallbackManager::AddCallbacks(std::vector<std::shared_ptr<DSCallback>> callbacks) {
  callbacks_.insert(callbacks_.end(), callbacks.begin(), callbacks.end());
  for (size_t ind = 0; ind < callbacks_.size(); ind++) {
    callbacks.push_back(callbacks[ind]);
    if (callbacks[ind]->IsBeginNeeded()) {
      begin_indices_.push_back(ind);
    }
    if (callbacks[ind]->IsEndNeeded()) {
      end_indices_.push_back(ind);
    }
    if (callbacks[ind]->IsEpochBeginNeeded()) {
      epoch_begin_indices_.push_back(ind);
    }
    if (callbacks[ind]->IsEpochEndNeeded()) {
      epoch_end_indices_.push_back(ind);
    }
    if (callbacks[ind]->IsNStepBeginNeeded()) {
      step_begin_indices_.push_back(ind);
    }
    if (callbacks[ind]->IsNStepEndNeeded()) {
      step_end_indices_.push_back(ind);
    }
  }
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

  // Now do the actual callback
  for (size_t ind : begin_indices_) {
    RETURN_IF_NOT_OK(callbacks_[ind]->DSBegin(cb_param));
  }
  return Status::OK();
}

Status CallbackManager::EpochBegin(const CallbackParam &cb_param) {
  RETURN_OK_IF_TRUE(!enabled_);
  RETURN_UNEXPECTED_IF_NULL(op_);

  // only wait if there are callbacks to call
  if (epoch_begin_indices_.size() > 0) {
    RETURN_IF_NOT_OK(op_->WaitForWorkers());
  }

  // Now do the actual callback
  for (size_t ind : epoch_begin_indices_) {
    RETURN_IF_NOT_OK(callbacks_[ind]->DSEpochBegin(cb_param));
  }

  // wakeup all the workers threads and collector thread
  RETURN_IF_NOT_OK(op_->PostForWorkers());

  return Status::OK();
}

Status CallbackManager::StepBegin(const CallbackParam &cb_param) {
  RETURN_OK_IF_TRUE(!enabled_);
  RETURN_UNEXPECTED_IF_NULL(op_);

  // Now do the actual callback
  for (size_t ind : step_begin_indices_) {
    if ((cb_param.cur_epoch_step_num_ - 1) % callbacks_[ind]->step_size() == 0) {
      RETURN_IF_NOT_OK(callbacks_[ind]->DSNStepBegin(cb_param));
    }
  }
  return Status::OK();
}

Status CallbackManager::End(const CallbackParam &cb_param) {
  RETURN_OK_IF_TRUE(!enabled_);
  RETURN_UNEXPECTED_IF_NULL(op_);
  // return Status::OK() if no end is needed
  RETURN_OK_IF_TRUE(end_indices_.empty());

  // Now do the actual callback
  for (size_t ind : end_indices_) {
    RETURN_IF_NOT_OK(callbacks_[ind]->DSEnd(cb_param));
  }
  return Status::OK();
}

Status CallbackManager::EpochEnd(const CallbackParam &cb_param) {
  RETURN_OK_IF_TRUE(!enabled_);
  RETURN_UNEXPECTED_IF_NULL(op_);

  // Now do the actual callback
  for (size_t ind : epoch_end_indices_) {
    RETURN_IF_NOT_OK(callbacks_[ind]->DSEpochEnd(cb_param));
  }
  return Status::OK();
}

Status CallbackManager::StepEnd(const CallbackParam &cb_param) {
  RETURN_OK_IF_TRUE(!enabled_);
  RETURN_UNEXPECTED_IF_NULL(op_);

  // Now do the actual callback
  for (size_t ind : step_end_indices_) {
    if ((cb_param.cur_epoch_step_num_ - 1) % callbacks_[ind]->step_size() == 0) {
      RETURN_IF_NOT_OK(callbacks_[ind]->DSNStepEnd(cb_param));
    }
  }
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
