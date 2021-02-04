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

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/engine/data_buffer.h"
#include "minddata/dataset/engine/datasetops/skip_op.h"
#include "minddata/dataset/engine/db_connector.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/util/log_adapter.h"

namespace mindspore {
namespace dataset {
// Builder constructor.  Creates the builder object.
SkipOp::Builder::Builder(int32_t count) : build_max_skips_(count) {
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  builder_op_connector_size_ = cfg->op_connector_size();
}

Status SkipOp::Builder::SanityCheck() const {
  if (build_max_skips_ < 0) {
    std::string err_msg("Invalid parameter, skip count should be greater than or equal to 0.");
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}

// The builder "build" method creates the final object.
Status SkipOp::Builder::Build(std::shared_ptr<SkipOp> *ptr) {
  RETURN_IF_NOT_OK(SanityCheck());
  *ptr = std::make_shared<SkipOp>(build_max_skips_, builder_op_connector_size_);
  return Status::OK();
}

// Constructor of the SkipOp.
SkipOp::SkipOp(int32_t count, int32_t op_connector_size)
    : PipelineOp(op_connector_size), max_skips_(count), skip_count_(0) {}

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

// Base-class override for handling cases when an eoe is received.
Status SkipOp::EoeReceived(int32_t worker_id) {
  skip_count_ = 0;
  state_ = OpState::kDeOpIdle;
  return Status::OK();
}

// main entry point for skip
Status SkipOp::operator()() {
  TaskManager::FindMe()->Post();
  std::unique_ptr<DataBuffer> curr_buffer;
  RETURN_IF_NOT_OK(GetNextInput(&curr_buffer));

  while (curr_buffer->eof() == false) {
    // Reset count
    skip_count_ = 0;
    while (curr_buffer->eoe() == false) {
      // Drop first count rows
      while (skip_count_ < max_skips_) {
        if (curr_buffer->eoe() || curr_buffer->eof()) {
          break;
        }
        // Consider the rows of buffer more than one
        TensorRow drop_row;
        int row_num = curr_buffer->NumRows();
        int drop_num = row_num + skip_count_ < max_skips_ ? row_num : max_skips_ - skip_count_;
        skip_count_ += drop_num;
        for (int i = 0; i < drop_num; i++) {
          RETURN_IF_NOT_OK(curr_buffer->PopRow(&drop_row));
        }
        if (curr_buffer->NumRows() == 0) {
          RETURN_IF_NOT_OK(GetNextInput(&curr_buffer));
        }
      }
      RETURN_IF_NOT_OK(out_connector_->Add(0, std::move(curr_buffer)));
      RETURN_IF_NOT_OK(GetNextInput(&curr_buffer));
    }
    // we got eoe, now try again until we got eof
    MS_LOG(DEBUG) << "Skip operator EOE Received.";
    RETURN_IF_NOT_OK(out_connector_->Add(0, std::move(std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOE))));
    RETURN_IF_NOT_OK(GetNextInput(&curr_buffer));
  }

  MS_LOG(DEBUG) << "Skip operator EOF Received.";
  RETURN_IF_NOT_OK(out_connector_->Add(0, std::move(std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOF))));
  return Status::OK();
}

// Base-class override for handling cases when an eof is received.
Status SkipOp::EofReceived(int32_t worker_id) {
  MS_LOG(DEBUG) << "Skip operator EOF received, do nothing now.";
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
