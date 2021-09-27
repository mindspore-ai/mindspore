/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/engine/datasetops/repeat_op.h"

#include "minddata/dataset/util/log_adapter.h"

namespace mindspore {
namespace dataset {
// Constructor of the RepeatOp.
RepeatOp::RepeatOp(int32_t count) : PipelineOp(0), num_repeats_(count), repeat_count_(0) {}

// Destructor
RepeatOp::~RepeatOp() {}

// A print method typically used for debugging
void RepeatOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    PipelineOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << " [repeats: " << num_repeats_ << "]\n";
  } else {
    // Call the super class for displaying any common detailed info
    PipelineOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nCurrent count: " << repeat_count_ << "\nMax count: " << num_repeats_ << "\nLeaf Nodes in execution path:";
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

// This function returns the row that is at the top of our output connector. The caller is
// typically our parent node, when the parent is asking us to provide the next row of data.
// Since RepeatOp is an inlined op, getting a row from us will simply bounce you to get
// a row from our child.
// This function sets the `retryIfEoe` flag when popping from the child connector. This way,
// this function will retry to pop the connector again and will get the non-EOE row if any.
Status RepeatOp::GetNextRow(TensorRow *row, int32_t worker_id, bool retry_if_eoe) {
  if (child_.empty()) {
    RETURN_STATUS_UNEXPECTED("Pipeline init failed, RepeatOp can't be the first op in pipeline.");
  }

  RETURN_IF_NOT_OK(child_[0]->GetNextRow(row, worker_id, true));
  // Loop until non EOE is received
  while (row->eoe()) {
    RETURN_IF_NOT_OK(EoeReceived(worker_id));
    if (state_ == OpState::kDeOpIdle) {
      return Status::OK();
    }
    RETURN_IF_NOT_OK(child_[0]->GetNextRow(row, worker_id, true));
  }
  // Check if the last buf is next eof
  if (row->eof()) {
    RETURN_IF_NOT_OK(EofReceived(worker_id));
  }
  return Status::OK();
}

// Base-class override for handling cases when an eoe is received.
Status RepeatOp::EoeReceived(int32_t worker_id) {
  UpdateRepeatAndEpochCounter();
  repeat_count_++;
  MS_LOG(DEBUG) << "Repeat operator (" << operator_id_
                << ") end of epoch message received. Repeat count is now: " << repeat_count_ << ".";

  if (repeat_count_ == num_repeats_) {
    repeat_count_ = 0;
    state_ = OpState::kDeOpIdle;
    return Status::OK();
  } else {
    state_ = OpState::kDeOpRunning;
  }

  // Invoke a reset against the eoe nodes only.
  for (auto &eoe_op : eoe_ops_) {
    MS_LOG(DEBUG) << "Repeat operator sending reset to operator: " << eoe_op->id();
    RETURN_IF_NOT_OK(eoe_op->Reset());
  }

  return Status::OK();
}

// Class functor operator () override.
// Most dataset ops operate by launching a thread (see ExecutionTree).
// However, the RepeatOp is defined as a inlined operator, so it is invalid to launch the
// functor since this op runs inlined inside another operator.  The function is overloaded to
// ensure that it is not called by mistake (it will generate an error).
Status RepeatOp::operator()() { RETURN_STATUS_UNEXPECTED("Logic error. RepeatOp is an inlined operator."); }

// Base-class override for handling cases when an eof is received.
Status RepeatOp::EofReceived(int32_t worker_id) {
  MS_LOG(DEBUG) << "Repeat operator EOF received, do nothing now.";
  return Status::OK();
}

int32_t RepeatOp::NumConsumers() const {
  if (parent_.empty()) {
    MS_LOG(DEBUG) << "Repeat operator, no parent node, assuming it's root and returning 1.";
    return 1;
  } else if (parent_[0] == nullptr) {
    MS_LOG(DEBUG) << "Repeat operator, pointer to the first parent is null. Returning 0.";
    return 0;
  } else {
    return parent_[0]->NumConsumers();
  }
}

// Drive reset actions if needed
Status RepeatOp::Reset() {
  // If there's nested repeats, an ascendant repeat may have ourself listed as an eoe op.
  // In that case, we now have to bounce the reset down to our own eoe ops.
  MS_LOG(DEBUG) << "Repeat operator " << operator_id_ << " got reset.";
  for (auto &eoe_op : eoe_ops_) {
    MS_LOG(DEBUG) << "Nested repeat operator bouncing a reset to operator: " << eoe_op->id();
    RETURN_IF_NOT_OK(eoe_op->Reset());
  }
  state_ = OpState::kDeOpRunning;
  return Status::OK();
}

int32_t RepeatOp::NumProducers() const {
  if (child_.empty() || child_[0] == nullptr) {
    MS_LOG(DEBUG) << "Repeat operator, pointer to child node is null. Returning 0.";
    return 0;
  } else {
    return child_[0]->NumProducers();
  }
}

int64_t RepeatOp::GetTreeRepeatCount() { return num_repeats_; }
}  // namespace dataset
}  // namespace mindspore
