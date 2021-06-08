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
#include <iostream>

#include "minddata/dataset/engine/datasetops/epoch_ctrl_op.h"

#include "minddata/dataset/util/log_adapter.h"

namespace mindspore {
namespace dataset {
// Constructor
EpochCtrlOp::EpochCtrlOp(int32_t num_epoch) : RepeatOp(num_epoch) { MS_LOG(INFO) << "Welcome to Epoch Ctrl Op."; }

// Destructor
EpochCtrlOp::~EpochCtrlOp() {}

// A print method typically used for debugging
void EpochCtrlOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    PipelineOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << " [epochs: " << num_repeats_ << "]\n";
  } else {
    // Call the super class for displaying any common detailed info
    RepeatOp::Print(out, show_all);
  }
}

Status EpochCtrlOp::GetNextRow(TensorRow *row, int32_t worker_id, bool retry_if_eoe) {
  if (child_.empty()) {
    RETURN_STATUS_UNEXPECTED("EpochCtrlOp can't be the leaf node(first operator) of pipeline.");
  }

  // `retry_if_eoe` is false because EpochCtrlOp does not eat EOE.
  RETURN_IF_NOT_OK(child_[0]->GetNextRow(row, worker_id, false));

  // Only intercept EOE for EoeReceived processing, after that the EOE is forwarded to next op.
  // Other TensorRows containing data or EOF will simply be forwarded.
  // EOF can simply be forwarded because this op does not spawn any thread, thus does not require clean up.
  if (row->eoe()) {
    RETURN_IF_NOT_OK(EoeReceived(worker_id));
  }

  return Status::OK();
}

Status EpochCtrlOp::EoeReceived(int32_t worker_id) {
  UpdateRepeatAndEpochCounter();
  repeat_count_++;
  MS_LOG(DEBUG) << "Epoch Control operator received end of epoch. Epoch count is now: " << repeat_count_
                << ". Max epochs: " << num_repeats_;

  // This will allow GetNextInput in DatasetOp class to pass EOE row instead of eating it.
  state_ = OpState::kDeOpIdle;

  if (repeat_count_ != num_repeats_) {
    for (auto &eoe_op : eoe_ops_) {
      MS_LOG(DEBUG) << "Epoch Control driving reset to op: " << eoe_op->id();
      RETURN_IF_NOT_OK(eoe_op->Reset());
    }
  }

  return Status::OK();
}

int64_t EpochCtrlOp::GetTreeRepeatCount() { return child_[0]->GetTreeRepeatCount(); }
}  // namespace dataset
}  // namespace mindspore
