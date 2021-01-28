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
#include <iomanip>
#include <iostream>
#include <utility>

#include "minddata/dataset/engine/datasetops/epoch_ctrl_op.h"
#include "minddata/dataset/engine/data_buffer.h"
#include "minddata/dataset/util/log_adapter.h"

namespace mindspore {
namespace dataset {

// The builder "build" method creates the final object.
Status EpochCtrlOp::Builder::Build(std::shared_ptr<EpochCtrlOp> *ptr) {
  RETURN_IF_NOT_OK(SanityCheck());
  *ptr = std::make_shared<EpochCtrlOp>(build_num_repeats_);
  return Status::OK();
}

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
    PipelineOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nCurrent epoch count: " << repeat_count_ << "\nMax epoch count: " << num_repeats_
        << "\nLeaf Nodes in execution path:";
    if (!eoe_ops_.empty()) {
      for (size_t i = 0; i < eoe_ops_.size(); i++) {
        out << "\n  Operator: " << eoe_ops_[i]->id();
      }
    } else {
      out << " None.";
    }
    out << "\n\n";
  }
}

Status EpochCtrlOp::GetNextBuffer(std::unique_ptr<DataBuffer> *p_buffer, int32_t worker_id, bool retry_if_eoe) {
  if (child_.empty()) {
    RETURN_STATUS_UNEXPECTED("EpochCtrlOp can't be the leaf node.");
  }

  std::unique_ptr<DataBuffer> buf;

  // `retry_if_eoe` is false because EpochCtrlOp does not eat EOE.
  RETURN_IF_NOT_OK(child_[0]->GetNextBuffer(&buf, worker_id, false));

  // Only intercept EOE for EoeReceived processing, after that the EOE is forwarded to next op.
  // Other databuffers containing data or EOF will simply be forwarded.
  // EOF can simply be forwarded because this op does not spawn any thread, thus does not require clean up.
  if (buf->eoe()) {
    RETURN_IF_NOT_OK(EoeReceived(worker_id));
  }

  *p_buffer = std::move(buf);
  return Status::OK();
}

Status EpochCtrlOp::EoeReceived(int32_t worker_id) {
  UpdateRepeatAndEpochCounter();
  repeat_count_++;
  MS_LOG(DEBUG) << "Epoch Control operator received end of epoch. Epoch count is now: " << repeat_count_
                << ". Max epochs: " << num_repeats_;

  // This will allow GetNextInput in DatasetOp class to pass EOE buffer instead of eating it.
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
