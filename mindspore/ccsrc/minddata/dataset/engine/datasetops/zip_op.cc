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
#include "minddata/dataset/engine/datasetops/zip_op.h"

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/util/log_adapter.h"

namespace mindspore {
namespace dataset {
// Construct ZipOp here, local variables initialized in operator due to tree construction restrictions
ZipOp::ZipOp() : PipelineOp(0) {}

// destructor
ZipOp::~ZipOp() {}

// fetches next zipped (merged) row
Status ZipOp::getNextZippedRow(TensorRow *const new_zip_row, int32_t *skip_child, bool is_pull_mode) const {
  *new_zip_row = {};
  // iterate over all iterators and generate a row
  for (size_t i = 0; i < child_.size(); ++i) {
    TensorRow new_row;
    if (!is_pull_mode) {
      RETURN_IF_NOT_OK(child_[i]->GetNextRow(&new_row));
    } else {
      RETURN_IF_NOT_OK(child_[i]->GetNextRowPullMode(&new_row));
    }
    if (new_row.eoe() || new_row.eof()) {
      *new_zip_row = new_row;
      *skip_child = static_cast<int32_t>(i);
      return Status::OK();
    } else {
      MS_LOG(DEBUG) << "Zip operator got row from child " << i << ". Num cols: " << new_row.size() << ".";
      new_zip_row->insert(new_zip_row->end(), new_row.begin(), new_row.end());
    }
  }
  return Status::OK();
}

// drain end of epoch messages from iterator for this epoch
Status ZipOp::drainPipeline(int32_t skip_child, bool is_pull_mode) const {
  for (size_t con = 0; con < child_.size(); ++con) {
    if (con == skip_child) {
      continue;
    }
    MS_LOG(DEBUG) << "Zip operator draining child at " << con << ".";
    TensorRow row;
    while (!row.eoe()) {
      if (!is_pull_mode) {
        RETURN_IF_NOT_OK(child_[con]->GetNextRow(&row));
      } else {
        RETURN_IF_NOT_OK(child_[con]->GetNextRowPullMode(&row));
      }
    }
  }
  // at this point all connectors don't contain end of epoch messages. next iteration should be clean
  return Status::OK();
}

// A function that prints info about the Operator
void ZipOp::Print(std::ostream &out,      // In: The output stream to print to
                  bool show_all) const {  // In: T/F if it should print everything
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    PipelineOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    PipelineOp::Print(out, show_all);
  }
  // Then show any custom derived-internal stuff
  out << "\nDatasets: " << child_.size() << "\n\n";
}

// overwrite function and handle eof
Status ZipOp::EofReceived(int32_t) {
  MS_LOG(DEBUG) << "Zip operator EOF received, do nothing now.";
  return Status::OK();
}

// overwrite function and handle eoe
Status ZipOp::EoeReceived(int32_t) {
  state_ = OpState::kDeOpIdle;
  return Status::OK();
}

Status ZipOp::ComputeColMap() {
  if (column_name_id_map_.empty()) {
    column_name_id_map_ = {};
    for (size_t i = 0; i < child_.size(); ++i) {
      // Initializing col_name_id_map from the child.
      const std::unordered_map<std::string, int32_t> col_name_id_map = child_[i]->column_name_id_map();
      int32_t colsCurrent = column_name_id_map_.size();
      // the update code below shouldn't do anything bad if the column name already exists.
      for (const auto &pair : col_name_id_map) {
        std::string name = pair.first;
        int32_t old_id = pair.second;
        // check if name already exists in column name descriptor
        if (column_name_id_map_.count(name) == 1) {
          RETURN_STATUS_UNEXPECTED("Invalid data, duplicate column " + name + " already exists when zipping datasets.");
        }
        column_name_id_map_[name] = old_id + colsCurrent;
      }
    }
    MS_LOG(DEBUG) << "Setting column map:\n" << this->ColumnNameMapAsString();
  } else {
    MS_LOG(WARNING) << "Column name map is already set!";
  }
  return Status::OK();
}

Status ZipOp::operator()() { RETURN_STATUS_UNEXPECTED("[Internal ERROR] ZipOp is an inlined operator."); }

Status ZipOp::GetNextRow(TensorRow *row) {
  RETURN_UNEXPECTED_IF_NULL(row);
  int32_t skip_child = -1;
  RETURN_IF_NOT_OK(getNextZippedRow(row, &skip_child, false));
  if (row->eoe()) {
    UpdateRepeatAndEpochCounter();
    MS_LOG(DEBUG) << "Zip operator is now draining child inputs.";
    RETURN_IF_NOT_OK(drainPipeline(skip_child, false));
  }
  return Status::OK();
}

Status ZipOp::GetNextRowPullMode(TensorRow *row) {
  RETURN_UNEXPECTED_IF_NULL(row);
  int32_t skip_child = -1;
  RETURN_IF_NOT_OK(getNextZippedRow(row, &skip_child, true));
  if (row->eoe()) {
    UpdateRepeatAndEpochCounter();
    MS_LOG(DEBUG) << "Zip operator in pull mode is now draining child inputs.";
    RETURN_IF_NOT_OK(drainPipeline(skip_child, true));
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
