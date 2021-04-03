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
#include "minddata/dataset/engine/dataset_iterator.h"
#include "minddata/dataset/engine/datasetops/skip_op.h"
#include "minddata/dataset/engine/db_connector.h"
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

// main entry point for skip
Status SkipOp::operator()() {
  TaskManager::FindMe()->Post();
  child_iterator_ = std::make_unique<ChildIterator>(this, 0, 0);

  TensorRow new_row;
  RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&new_row));
  while (!new_row.eof()) {
    // Reset count
    skip_count_ = 0;
    while (!new_row.eoe()) {
      // Drop first count rows
      if (skip_count_ < max_skips_) {
        skip_count_++;
      } else {
        RETURN_IF_NOT_OK(out_connector_->Add(std::move(new_row)));
      }
      RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&new_row));
    }
    // we got eoe, now try again until we got eof
    MS_LOG(DEBUG) << "Skip operator EOE Received.";
    RETURN_IF_NOT_OK(out_connector_->SendEOE());
    RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&new_row));
  }
  MS_LOG(DEBUG) << "Skip operator EOF Received.";
  RETURN_IF_NOT_OK(out_connector_->SendEOF());
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
