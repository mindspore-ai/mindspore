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
#include <utility>

#include <algorithm>
#include "utils/ms_utils.h"
#include "minddata/dataset/core/config_manager.h"

#include "minddata/dataset/engine/dataset_iterator.h"
#include "minddata/dataset/engine/datasetops/take_op.h"
#include "minddata/dataset/engine/db_connector.h"
#include "minddata/dataset/engine/execution_tree.h"

namespace mindspore {
namespace dataset {
// Builder constructor. Creates the builder object.
TakeOp::Builder::Builder(int32_t count) : build_max_takes_(count) {
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  builder_op_connector_size_ = cfg->op_connector_size();
}

Status TakeOp::Builder::SanityCheck() const {
  if (build_max_takes_ <= 0) {
    std::string err_msg("Invalid parameter, take count must be greater than 0.");
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}

// The builder "build" method creates the final object.
Status TakeOp::Builder::Build(std::shared_ptr<TakeOp> *ptr) {
  RETURN_IF_NOT_OK(SanityCheck());
  *ptr = std::make_shared<TakeOp>(build_max_takes_, builder_op_connector_size_);
  return Status::OK();
}

// Constructor of the TakeOp.
TakeOp::TakeOp(int32_t count, int32_t op_connector_size)
    : PipelineOp(op_connector_size), max_takes_(count), take_count_(0) {}

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

// Main entry point for Take
Status TakeOp::operator()() {
  TaskManager::FindMe()->Post();
  child_iterator_ = std::make_unique<ChildIterator>(this, 0, 0);

  TensorRow new_row;
  RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&new_row));

  while (!new_row.eof()) {
    while (!new_row.eoe()) {
      if (take_count_ < max_takes_) {
        RETURN_IF_NOT_OK(out_connector_->Add(std::move(new_row)));
        take_count_++;
        RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&new_row));
      }
      if (take_count_ == max_takes_) {
        RETURN_IF_NOT_OK(child_iterator_->Drain());
        break;
      }
    }
    UpdateRepeatAndEpochCounter();
    take_count_ = 0;
    RETURN_IF_NOT_OK(out_connector_->SendEOE());
    RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&new_row));
  }

  take_count_ = 0;
  MS_LOG(DEBUG) << "Meet the end and push-back eof row.";
  RETURN_IF_NOT_OK(out_connector_->SendEOF());
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
