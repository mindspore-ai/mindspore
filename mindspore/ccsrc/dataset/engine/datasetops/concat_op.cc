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
#include <iomanip>
#include <utility>

#include "common/utils.h"
#include "dataset/core/config_manager.h"
#include "dataset/engine/data_buffer.h"
#include "dataset/engine/datasetops/concat_op.h"
#include "dataset/engine/db_connector.h"
#include "dataset/engine/execution_tree.h"

namespace mindspore {
namespace dataset {
// Builder constructor. Creates the builder object.
ConcatOp::Builder::Builder() {
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  builder_op_connector_size_ = cfg->op_connector_size();
}

// The builder "build" method creates the final object.
Status ConcatOp::Builder::Build(std::shared_ptr<ConcatOp> *ptr) {
  *ptr = std::make_shared<ConcatOp>(builder_op_connector_size_);
  return Status::OK();
}

// Constructor of the ConcatOp.
ConcatOp::ConcatOp(int32_t op_connector_size) : PipelineOp(op_connector_size), children_num_(0) {}

// A function that prints info about the Operator
void ConcatOp::Print(std::ostream &out, bool show_all) const {
  // Always show the id and name as first line regardless if this is summary or detailed print
  out << "(" << std::setw(2) << operator_id_ << ") <ConcatOp>:";
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    PipelineOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    PipelineOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nDatasets: " << children_num_ << "\n\n";
  }
}

// Main entry point for Concat
Status ConcatOp::operator()() {
  // The children_num_ parameter needs to be put here
  children_num_ = static_cast<int32_t>(child_.size());

  TaskManager::FindMe()->Post();
  std::unique_ptr<DataBuffer> buf;
  RETURN_IF_NOT_OK(child_[0]->GetNextBuffer(&buf));

  // Obtain columns_name_id_map from child_[0]
  column_name_id_map_ = child_[0]->column_name_id_map();
  if (column_name_id_map_.empty()) {
    RETURN_STATUS_UNEXPECTED("Child column name map cannot be empty!");
  }

  int eof_count = 0;
  while (eof_count != children_num_) {
    for (int i = 0; i < children_num_; i++) {
      // 1. Throw the eof buffer when meet it
      if (buf->eof() || buf->eoe()) {
        RETURN_IF_NOT_OK(child_[i]->GetNextBuffer(&buf));
      }
      // 2. Do varification as for column name, column data type and rank of column data
      RETURN_IF_NOT_OK(Verify(i, buf));

      // 3. Put the data into output_connector
      while (!buf->eoe() && !buf->eof()) {
        RETURN_IF_NOT_OK(out_connector_->Add(0, std::move(buf)));
        RETURN_IF_NOT_OK(child_[i]->GetNextBuffer(&buf));
      }

      // 4. Throw the eoe buffer when meet it
      if (buf->eoe() && (!BitTest(op_ctrl_flags_, kDeOpRepeated) || BitTest(op_ctrl_flags_, kDeOpLastRepeat))) {
        RETURN_IF_NOT_OK(child_[i]->GetNextBuffer(&buf));
      }
      // 5. Add eoe buffer after get buffer from all child
      if (i == (children_num_ - 1)) {
        auto eoe_buffer = std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOE);
        RETURN_IF_NOT_OK(out_connector_->Add(0, std::move(eoe_buffer)));
      }
      if (buf->eof()) {
        eof_count++;
      }
    }
  }
  // 6. Add eof buffer in the end manually
  MS_LOG(DEBUG) << "Add the eof buffer manualy in the end.";
  auto eof_buffer = std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOF);
  RETURN_IF_NOT_OK(out_connector_->Add(0, std::move(eof_buffer)));

  return Status::OK();
}

Status ConcatOp::Verify(int32_t id, const std::unique_ptr<DataBuffer> &buf) {
  TensorRow new_row;
  buf->GetRow(0, &new_row);

  if (id == 0) {
    // Obtain the column name, data type and data rank in child[0]
    column_name_id_ = child_[id]->column_name_id_map();
    for (auto item : new_row) {
      data_type_.push_back(item->type());
      data_rank_.push_back(item->Rank());
    }
  } else {
    // Compare the column name, data type and data rank with these in child[0]
    if (child_[id]->column_name_id_map() != column_name_id_) {
      RETURN_STATUS_UNEXPECTED("The column name or column order is not the same with previous dataset.");
    }
    int32_t index = 0;
    for (auto item : new_row) {
      if ((item->type() != data_type_[index]) || item->Rank() != data_rank_[index++]) {
        RETURN_STATUS_UNEXPECTED("The data type or data rank is not the same with previous dataset.");
      }
    }
  }
  return Status::OK();
}

Status ConcatOp::PrepareNodePostAction() {
  RETURN_IF_NOT_OK(PipelineOp::PrepareNodePostAction());
  tree_->AddToRepeatStack(shared_from_this());
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
