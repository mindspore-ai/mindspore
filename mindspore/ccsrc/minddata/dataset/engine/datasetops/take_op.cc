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
#include "minddata/dataset/engine/datasetops/take_op.h"

#include "utils/ms_utils.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/engine/db_connector.h"

namespace mindspore {
namespace dataset {
// Constructor of the TakeOp.
TakeOp::TakeOp(int32_t count) : PipelineOp(0), max_takes_(count), take_count_(0) {}

// A print method typically used for debugging
void TakeOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    PipelineOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << " [takes: " << max_takes_ << "]\n";
  } else {
    // Call the super class for displaying any common detailed info
    PipelineOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nTake count: " << take_count_ << "\nMax takes: " << max_takes_ << "\n\n";
  }
}

Status TakeOp::operator()() { RETURN_STATUS_UNEXPECTED("Logic error. SkipOp is an inlined operator."); }

Status TakeOp::GetNextRow(TensorRow *row, int32_t worker_id, bool retry_if_eoe) {
  bool eoe_received = false;
  if (take_count_ < max_takes_) {
    RETURN_IF_NOT_OK(child_[0]->GetNextRow(row, worker_id, retry_if_eoe));
    if (row->eoe()) {
      eoe_received = true;
    } else {
      take_count_++;
      return Status::OK();
    }
  }
  if (take_count_ == max_takes_) {
    // drain
    while (!row->eoe()) {
      RETURN_IF_NOT_OK(child_[0]->GetNextRow(row, worker_id, retry_if_eoe));
    }
    eoe_received = true;
  }
  if (eoe_received) {
    UpdateRepeatAndEpochCounter();
    take_count_ = 0;
  }

  return Status::OK();
}

int32_t TakeOp::NumConsumers() const {
  if (parent_.empty()) {
    MS_LOG(DEBUG) << "Return operator, no parent node, assuming it's the root and returning 1.";
    return 1;
  } else if (parent_[0] == nullptr) {
    MS_LOG(DEBUG) << "Return operator, pointer to the first parent is null. Returning 0.";
    return 0;
  } else {
    return parent_[0]->NumConsumers();
  }
}

int32_t TakeOp::NumProducers() const {
  if (child_.empty() || child_[0] == nullptr) {
    MS_LOG(DEBUG) << "Return operator, pointer to child node is null. Returning 0.";
    return 0;
  } else {
    return child_[0]->NumProducers();
  }
}
}  // namespace dataset
}  // namespace mindspore
