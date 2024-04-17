/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/engine/tree_modifier.h"

namespace mindspore {
namespace dataset {
Status AutotuneCallback::DSNStepBegin(const CallbackParam &cb_param) {
  // check if the queue is empty, no need to wait until a change request is ready
  if (!change_request_queue_->empty()) {
    ChangeRequestPtr change_request;
    RETURN_IF_NOT_OK(change_request_queue_->PopFront(&change_request));
    RETURN_IF_NOT_OK(change_request->ApplyChange(op_));
  }
  return Status::OK();
}

Status AutotuneCallback::DSBegin(const CallbackParam &cb_param) { return Status::OK(); }

Status AutotuneCallback::DSEpochBegin(const CallbackParam &cb_param) { return Status::OK(); }

Status AutotuneCallback::DSEnd(const CallbackParam &cb_param) { return Status::OK(); }

Status AutotuneCallback::DSEpochEnd(const CallbackParam &cb_param) { return Status::OK(); }

Status AutotuneCallback::DSNStepEnd(const CallbackParam &cb_param) { return Status::OK(); }

bool AutotuneCallback::IsBeginNeeded() { return false; }

bool AutotuneCallback::IsEpochBeginNeeded() { return false; }

bool AutotuneCallback::IsNStepBeginNeeded() { return true; }

bool AutotuneCallback::IsEndNeeded() { return false; }

bool AutotuneCallback::IsEpochEndNeeded() { return false; }

bool AutotuneCallback::IsNStepEndNeeded() { return false; }

Status AutotuneCallback::PushChangeRequest(ChangeRequestPtr change_request) {
  RETURN_IF_NOT_OK(change_request_queue_->Add(std::move(change_request)));
  return Status::OK();
}

Status ChangeNumWorkersRequest::ApplyChange(DatasetOp *op) {
  int32_t diff = num_workers_ - op->NumWorkers();
  if (diff > 0) {
    RETURN_IF_NOT_OK(op->AddNewWorkers(diff));
  } else if (diff < 0) {
    RETURN_IF_NOT_OK(op->RemoveWorkers(-1 * diff));
  }
  return Status::OK();
}

TreeModifier::TreeModifier(const TreeAdapter *adapter) : TreeModifier(adapter->tree_.get()) {}
}  // namespace dataset
}  // namespace mindspore
