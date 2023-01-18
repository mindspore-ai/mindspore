/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/engine/datasetops/rename_op.h"

#include <set>
#include <unordered_map>
#include <vector>

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/util/log_adapter.h"

namespace mindspore {
namespace dataset {
//  constructor
RenameOp::RenameOp(const std::vector<std::string> &in_col_names, const std::vector<std::string> &out_col_names)
    : PipelineOp(0), in_columns_(in_col_names), out_columns_(out_col_names) {}

// destructor
RenameOp::~RenameOp() {}

// Gets a row from the child operator
Status RenameOp::GetNextRow(TensorRow *row) {
  RETURN_UNEXPECTED_IF_NULL(row);
  RETURN_IF_NOT_OK(child_[0]->GetNextRow(row));
  if (row->eoe()) {
    UpdateRepeatAndEpochCounter();
  }
  return Status::OK();
}

// For Pullmode, gets a row from the child operator
Status RenameOp::GetNextRowPullMode(TensorRow *row) {
  RETURN_UNEXPECTED_IF_NULL(row);
  RETURN_IF_NOT_OK(child_[0]->GetNextRowPullMode(row));
  if (row->eoe()) {
    UpdateRepeatAndEpochCounter();
  }
  return Status::OK();
}

Status RenameOp::operator()() { RETURN_STATUS_UNEXPECTED("[Internal ERROR] RenameOp is an inlined operator."); }

// Rename core functionality to compute the new column name id map.
// We need to overwrite the super class ComputeColMap here because we're making a modification of the
// map from the child map.
Status RenameOp::ComputeColMap() {
  if (column_name_id_map_.empty()) {
    column_name_id_map_ = child_[0]->column_name_id_map();
    // iterate over my index in input vector, find the corresponding position
    std::unordered_map<std::string, int32_t> new_col_name_id_map = {};
    // parameter for input check
    size_t found = 0;
    std::set<std::string> new_col_name;

    // iterate over all the pairs and if there is a name match with rename, rename the column and add it to new map
    // by doing it this way we recreate a new ColNameIdMap and allow for switching
    for (const auto &pair : column_name_id_map_) {
      std::string name = pair.first;
      int32_t id = pair.second;
      // find name
      std::vector<std::string>::iterator it = std::find(in_columns_.begin(), in_columns_.end(), name);
      // for c input checks here we have to count the number of times we find the stuff in in_columns_
      // because we iterate over the mInputList n times
      if (it != in_columns_.end()) {
        // found
        found += 1;
        int index = std::distance(in_columns_.begin(), it);
        MS_LOG(DEBUG) << "Rename operator index found " << index << " value " << id << ".";
        if (new_col_name.find(out_columns_[index]) != new_col_name.end()) {
          std::string err_msg(
            "Invalid column, rename operation does not support rename one column name into another already exist "
            "column name, existing column name is: " +
            out_columns_[index] + ".");
          RETURN_STATUS_UNEXPECTED(err_msg);
        }
        new_col_name_id_map[out_columns_[index]] = id;
        new_col_name.insert(out_columns_[index]);
      } else {
        // not found
        if (new_col_name.find(name) != new_col_name.end()) {
          std::string err_msg(
            "Invalid column, rename operation does not support rename one column name into another already exist "
            "column name, existing column name is: " +
            name + ".");
          RETURN_STATUS_UNEXPECTED(err_msg);
        }
        MS_LOG(DEBUG) << "Rename operator index not found: " << id << " is the column id.";
        new_col_name_id_map[name] = id;
        new_col_name.insert(name);
      }
    }
    // only checks number of renamed columns have been found, this input check doesn't check everything
    if (found != in_columns_.size()) {
      MS_LOG(DEBUG) << "Rename operator column names found: " << found << " out of " << in_columns_.size() << ".";
      std::string err_msg = "Invalid column, column to be renamed does not exist.";
      RETURN_STATUS_UNEXPECTED(err_msg);
    }

    // Now, overwrite our column map with the new renamed columns/id's
    column_name_id_map_ = new_col_name_id_map;
  } else {
    MS_LOG(WARNING) << "Column name map is already set!";
  }
  return Status::OK();
}

// prints rename
void RenameOp::Print(std::ostream &out,      // In: The output stream to print to
                     bool show_all) const {  // In: T/F if it should print everything
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
}  // namespace dataset
}  // namespace mindspore
