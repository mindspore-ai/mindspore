/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "dataset/engine/datasetops/rename_op.h"
#include <iomanip>
#include <vector>
#include <utility>
#include <unordered_map>

#include "dataset/core/config_manager.h"
#include "dataset/core/constants.h"
#include "dataset/core/global_context.h"
#include "dataset/engine/data_buffer.h"
#include "dataset/engine/db_connector.h"
#include "dataset/engine/opt/pass.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace dataset {
// builds
RenameOp::Builder::Builder() {
  // Some arguments to the RenameOp constructor have a default argument that is taken
  // from the client config.
  // The user may choose to change these values for the construction of the RenameOp by
  // using the various builder set methods.

  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  builder_op_connector_size_ = cfg->op_connector_size();
}

Status RenameOp::Builder::SanityCheck() const { return Status::OK(); }

// build method for RenameOp
Status RenameOp::Builder::Build(std::shared_ptr<RenameOp> *ptr) {
  RETURN_IF_NOT_OK(SanityCheck());
  *ptr = std::make_shared<RenameOp>(builder_in_columns_, builder_out_columns_, builder_op_connector_size_);
  return Status::OK();
}

//  constructor
RenameOp::RenameOp(const std::vector<std::string> &in_col_names, const std::vector<std::string> &out_col_names,
                   int32_t op_connector_size)
    : PipelineOp(op_connector_size), in_columns_(in_col_names), out_columns_(out_col_names) {}

// destructor
RenameOp::~RenameOp() {}

// main entry point for rename
Status RenameOp::operator()() {
  TaskManager::FindMe()->Post();
  std::unique_ptr<DataBuffer> curr_buffer;
  RETURN_IF_NOT_OK(GetNextInput(&curr_buffer));
  if (curr_buffer->buffer_flags() != DataBuffer::kDeBFlagNone) {
    RETURN_IF_NOT_OK(out_connector_->Add(0, std::move(curr_buffer)));
    std::string err_msg = "Rename first buffer got was control signal";
    // if 1st eoe or eof, pass it on then return
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  // First, populate the column map from the input child.
  // This will not be the final map for output from this op.
  RETURN_IF_NOT_OK(DatasetOp::AssignColMapFromChild());
  // core rename functionality only needs to happen once, to identify the new column names/indexes
  RETURN_IF_NOT_OK(RenameColumns());

  while (curr_buffer->eof() == false) {
    while (curr_buffer->eoe() == false) {
      // push the renamed input buffer
      MS_LOG(DEBUG) << "Rename operator pushing next buffer.";
      RETURN_IF_NOT_OK(out_connector_->Add(0, std::move(curr_buffer)));
      RETURN_IF_NOT_OK(GetNextInput(&curr_buffer));
    }  // end of while eoe loop

    // we got eoe, now try again until we get eof
    MS_LOG(INFO) << "Rename operator EOE Received.";
    RETURN_IF_NOT_OK(out_connector_->Add(0, std::move(std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOE))));
    MS_LOG(DEBUG) << "Rename operator fetching buffer after EOE.";
    RETURN_IF_NOT_OK(GetNextInput(&curr_buffer));
  }  // end of while eof loop

  MS_LOG(INFO) << "Rename opeerator EOF Received.";
  RETURN_IF_NOT_OK(out_connector_->Add(0, std::move(std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOF))));
  return Status::OK();
}

// renames the columns
Status RenameOp::RenameColumns() {
  // iterate over my index in input vector, find the corresponding position
  std::unordered_map<std::string, int32_t> new_col_name_id_map = {};
  // parameter for input check
  size_t found = 0;

  // iterate over all the pairs and if there is a name match with rename, rename the column and add it to new map
  // by doing it this way we recreate a new ColNameIdMap and allow for switching
  for (const auto &pair : column_name_id_map_) {
    std::string name = pair.first;
    int32_t id = pair.second;
    // find name
    std::vector<std::string>::iterator it;
    it = std::find(in_columns_.begin(), in_columns_.end(), name);
    // for c input checks here we have to count the number of times we find the stuff in in_columns_
    // because we iterate over the mInputList n times
    if (it != in_columns_.end()) {
      // found
      found += 1;
      int index = std::distance(in_columns_.begin(), it);
      MS_LOG(INFO) << "Rename operator index found " << index << " value " << id << ".";

      new_col_name_id_map[out_columns_[index]] = id;
    } else {
      // not found
      MS_LOG(INFO) << "Rename operator index not found: " << id << " is the column id.";
      new_col_name_id_map[name] = id;
    }
  }
  // only checks number of renamed columns have been found, this input check doesn't check everything
  if (found != in_columns_.size()) {
    MS_LOG(INFO) << "Rename operator column names found: " << found << " out of " << in_columns_.size() << ".";
    std::string err_msg = "Renamed column doesn't exist in dataset";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  // Now, overwrite our column map with the new renamed columns/id's
  column_name_id_map_ = new_col_name_id_map;
  return Status::OK();
}

// prints rename
void RenameOp::Print(std::ostream &out,      // In: The output stream to print to
                     bool show_all) const {  // In: T/F if it should print everything
  // Always show the id and name as first line regardless if this summary or detailed print
  out << "(" << std::setw(2) << operator_id_ << ") <RenameOp>:";
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    PipelineOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    PipelineOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nIn columns:";
    for (size_t i = 0; i < in_columns_.size(); ++i) {
      out << "\n  " << in_columns_[i];
    }
    for (size_t i = 0; i < out_columns_.size(); ++i) {
      out << "\n  " << out_columns_[i];
    }
    out << "\n\n";
  }
}

Status RenameOp::EofReceived(int32_t) {
  MS_LOG(INFO) << "Rename operator EOF received, do nothing now.";
  return Status::OK();
}

Status RenameOp::EoeReceived(int32_t) {
  state_ = OpState::kDeOpIdle;
  return Status::OK();
}

// Visitor accept method for NodePass
Status RenameOp::Accept(NodePass *p, bool *modified) {
  // Downcast shared pointer then call visitor
  return p->RunOnNode(std::static_pointer_cast<RenameOp>(shared_from_this()), modified);
}
}  // namespace dataset
}  // namespace mindspore
