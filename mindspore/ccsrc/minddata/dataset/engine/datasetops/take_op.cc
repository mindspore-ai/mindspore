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
#include "minddata/dataset/engine/data_buffer.h"
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
  std::unique_ptr<DataBuffer> buf;
  RETURN_IF_NOT_OK(child_[0]->GetNextBuffer(&buf));

  while (buf->eof() == false) {
    if (take_count_ == max_takes_) {
      // Do drain Operation
      while (!buf->eoe() && !buf->eof()) {
        RETURN_IF_NOT_OK(child_[0]->GetNextBuffer(&buf));
      }
    }

    // Loop until non EOE is received
    if (buf->eoe()) {
      UpdateRepeatAndEpochCounter();
      take_count_ = 0;
      RETURN_IF_NOT_OK(out_connector_->Add(0, std::move(buf)));
      RETURN_IF_NOT_OK(child_[0]->GetNextBuffer(&buf));
      continue;
    }

    // Get buffer and push back when take_count is still small
    if (take_count_ < max_takes_) {
      std::unique_ptr<DataBuffer> p_buffer;
      RETURN_IF_NOT_OK(FillBuffer(&buf, &p_buffer));
      RETURN_IF_NOT_OK(out_connector_->Add(0, std::move(p_buffer)));
    }
    RETURN_IF_NOT_OK(child_[0]->GetNextBuffer(&buf));
  }

  take_count_ = 0;
  MS_LOG(DEBUG) << "Meet the end and push-back eof buffer.";
  auto eof_buffer = std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOF);
  RETURN_IF_NOT_OK(out_connector_->Add(0, std::move(eof_buffer)));
  return Status::OK();
}

// Function FillBuffer mainly prepare the buffer for returning
Status TakeOp::FillBuffer(std::unique_ptr<DataBuffer> *buffer, std::unique_ptr<DataBuffer> *data_buffer) {
  int32_t buffer_size = (*buffer)->NumRows();
  if (take_count_ + buffer_size < max_takes_) {
    *data_buffer = std::move(*buffer);
    take_count_ = take_count_ + buffer_size;
  } else {
    MS_LOG(DEBUG) << "In last buffer: Push one buffer.";
    std::unique_ptr<TensorQTable> new_tensor_table = std::make_unique<TensorQTable>();
    while (take_count_ < max_takes_) {
      TensorRow new_row;
      RETURN_IF_NOT_OK((*buffer)->PopRow(&new_row));
      take_count_++;
      new_tensor_table->push_back(new_row);
    }
    (*buffer)->set_tensor_table(std::move(new_tensor_table));
    *data_buffer = std::move(*buffer);
  }
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
