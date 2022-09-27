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
#include "minddata/dataset/engine/datasetops/skip_op.h"

#include <iostream>

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/util/log_adapter.h"

namespace mindspore {
namespace dataset {
// Constructor of the SkipOp.
SkipOp::SkipOp(int32_t count) : PipelineOp(0), max_skips_(count), skip_count_(0) {}

// Destructor
SkipOp::~SkipOp() {}

// A print method typically used for debugging
void SkipOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    PipelineOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << " [skips: " << max_skips_ << "]\n";
  } else {
    // Call the super class for displaying any common detailed info
    PipelineOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nSkip count: " << skip_count_ << "\nMax skips: " << max_skips_ << "\n\n";
  }
}

Status SkipOp::operator()() { RETURN_STATUS_UNEXPECTED("[Internal ERROR] SkipOp is an inlined operator."); }

Status SkipOp::GetNextRow(TensorRow *row) {
  RETURN_UNEXPECTED_IF_NULL(row);
  bool eoe_received = false;
  while (skip_count_ < max_skips_) {
    RETURN_IF_NOT_OK(child_[0]->GetNextRow(row));
    if (row->eoe() && !once_only_) {
      eoe_received = true;
      break;
    }
    if (!row->eoe()) {
      skip_count_++;
    }
  }
  if (!eoe_received) {
    RETURN_IF_NOT_OK(child_[0]->GetNextRow(row));
  }
  if (row->eoe()) {
    UpdateRepeatAndEpochCounter();
    if (!once_only_) {
      skip_count_ = 0;
    }
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
