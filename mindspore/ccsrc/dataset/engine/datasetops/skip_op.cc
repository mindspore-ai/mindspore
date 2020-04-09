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
#include <iostream>
#include <utility>

#include "dataset/engine/data_buffer.h"
#include "dataset/engine/datasetops/skip_op.h"
#include "dataset/engine/db_connector.h"
#include "dataset/engine/execution_tree.h"

#include "utils/log_adapter.h"

namespace mindspore {
namespace dataset {
// Builder constructor.  Creates the builder object.
SkipOp::Builder::Builder(int32_t count) : build_max_skips_(count) {}

Status SkipOp::Builder::SanityCheck() const {
  if (build_max_skips_ < 0) {
    std::string err_msg("Skip count must be positive integer or 0.");
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}

// The builder "build" method creates the final object.
Status SkipOp::Builder::Build(std::shared_ptr<SkipOp> *ptr) {
  RETURN_IF_NOT_OK(SanityCheck());
  *ptr = std::make_shared<SkipOp>(build_max_skips_);
  return Status::OK();
}

// Constructor of the SkipOp.
SkipOp::SkipOp(int32_t count) : PipelineOp(0), max_skips_(count), skip_count_(0) {}

// Destructor
SkipOp::~SkipOp() {}

// A print method typically used for debugging
void SkipOp::Print(std::ostream &out, bool show_all) const {
  // Call base class printer first
  PipelineOp::Print(out, show_all);

  // Then display our own stuff
  out << "SkipOp:"
      << "\nCurrent skip count: " << skip_count_ << "\nMax skip count: " << max_skips_;
}

// Since the buffer may contain multi rows, this function will drop the rows
// that need to skip in it, and then return the buffer.
Status SkipOp::GetNextBuffer(std::unique_ptr<DataBuffer> *p_buffer, int32_t worker_id, bool retry_if_eoe) {
  if (child_.empty()) {
    RETURN_STATUS_UNEXPECTED("SkipOp can't be the leaf node.");
  }

  std::unique_ptr<DataBuffer> buf;
  // Drop first max_skips_ rows
  while (skip_count_ < max_skips_) {
    RETURN_IF_NOT_OK(child_[0]->GetNextBuffer(&buf, worker_id, true));
    if (buf->eoe() || buf->eof()) {
      break;
    }

    // Consider the rows of buffer more than 1
    TensorRow drop_row;
    int row_num = buf->NumRows();
    for (int i = 0; i < row_num; i++) {
      RETURN_IF_NOT_OK(buf->PopRow(&drop_row));
      if (++skip_count_ == max_skips_) {
        break;
      }
    }
  }

  // If buffer is none or the rows of buffer is 0,
  // then get a buffer from child.
  if (!buf || buf->NumRows() == 0) {
    RETURN_IF_NOT_OK(child_[0]->GetNextBuffer(&buf, worker_id, true));
  }

  // Handling eoe and eof
  if (buf->eoe() || buf->eof()) {
    RETURN_IF_NOT_OK(EoeReceived(worker_id));
    if (state_ == OpState::kDeOpIdle) {
      *p_buffer = std::move(buf);
      return Status::OK();
    }
  }

  *p_buffer = std::move(buf);
  return Status::OK();
}

// Base-class override for handling cases when an eoe is received.
Status SkipOp::EoeReceived(int32_t worker_id) {
  skip_count_ = 0;
  state_ = OpState::kDeOpIdle;
  return Status::OK();
}

// Class functor operator () override.
// Most dataset ops operate by launching a thread (see ExecutionTree).
// However, the SkipOp is defined as a inlined operator, so it is invalid to
// launch the functor since this op runs inlined inside another operator.  The
// function is overloaded to ensure that it is not called by mistake (it will
// generate an error).
Status SkipOp::operator()() { RETURN_STATUS_UNEXPECTED("Logic error. SkipOp is an inlined operator."); }

// Base-class override for handling cases when an eof is received.
Status SkipOp::EofReceived(int32_t worker_id) {
  MS_LOG(INFO) << "Skip operator EOF received, do nothing now.";
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
