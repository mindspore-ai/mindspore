/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include <algorithm>
#include <utility>
#include <iomanip>
#include "minddata/dataset/include/constants.h"

#include "minddata/dataset/engine/db_connector.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/util/log_adapter.h"

namespace mindspore {
namespace dataset {
ZipOp::Builder::Builder() {
  // Some arguments to the ZipOp constructor have a default argument that is taken
  // from the client config.
  // The user may choose to change these values for the construction of the ZipOp by
  // using the various builder set methods.

  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  builder_op_connector_size_ = cfg->op_connector_size();
}

Status ZipOp::Builder::SanityCheck() const { return Status::OK(); }

Status ZipOp::Builder::Build(std::shared_ptr<ZipOp> *ptr) {
  RETURN_IF_NOT_OK(SanityCheck());
  *ptr = std::make_shared<ZipOp>(builder_op_connector_size_);
  return Status::OK();
}

// Construct ZipOp here, local variables initialized in operator due to tree construction restrictions
ZipOp::ZipOp(int32_t op_connector_size)
    : PipelineOp(op_connector_size), children_num_(0), draining_(false), eof_(false) {}

// destructor
ZipOp::~ZipOp() {}

// Entry point for Zip, called by launch()
Status ZipOp::operator()() {
  // The children_num_ parameter needs to be put here
  children_num_ = child_.size();
  // Synchronize with TaskManager once the thread is created.
  TaskManager::FindMe()->Post();

  // initialize the iterators
  for (int32_t i = 0; i < children_num_; ++i) {
    // magic number 0 since Zip is not a parallel Op
    child_iterators_.push_back(std::make_unique<ChildIterator>(this, 0, i));
  }

  // Loop until eof is true
  while (!eof_) {
    // 1 Prepare new epoch
    RETURN_IF_NOT_OK(prepare());
    // 2 fetch first row
    TensorRow row;
    RETURN_IF_NOT_OK(getNextTensorRow(&row));

    // If an eof got picked up, then we're done
    if (eof_) {
      break;
    }
    while (!draining_) {
      // 3 send new row to the out connector
      MS_LOG(DEBUG) << "Zip operator finished one row, pushing, cols " << row.size() << ", map "
                    << column_name_id_map_.size() << ".";
      RETURN_IF_NOT_OK(out_connector_->Add(std::move(row)));
      // 4 fetch one more row
      RETURN_IF_NOT_OK(getNextTensorRow(&row));
    }
    // 5 handle drain state.
    if (draining_) {
      MS_LOG(DEBUG) << "Zip operator is now draining child inputs.";
      RETURN_IF_NOT_OK(drainPipeline());
      // Now that we have drained child inputs, send the eoe up.
      RETURN_IF_NOT_OK(out_connector_->SendEOE());
    }
  }

  // 6 handle eof
  MS_LOG(DEBUG) << "Zip operator got EOF, propagating.";
  RETURN_IF_NOT_OK(out_connector_->SendEOF());
  return Status::OK();
}

// Handles preprocessing of the main loop, used when starting new epoch
Status ZipOp::prepare() {
  MS_LOG(DEBUG) << "Zip operator prepares for new epoch.";
  draining_ = false;
  return Status::OK();
}

// fetches next zipped (merged) row
Status ZipOp::getNextTensorRow(TensorRow *const new_zip_row) {
  // iterate over all iterators and generate a row
  for (int32_t i = 0; i < children_num_; ++i) {
    TensorRow new_row = {};
    RETURN_IF_NOT_OK((child_iterators_[i])->FetchNextTensorRow(&new_row));
    // add each new row to iterator, check if row is empty, if row from iterator is empty return empty row
    if (new_row.empty()) {
      // If we did not get a row from any of the children, then it's the end of an epoch and we can move
      // to drain state.
      MS_LOG(DEBUG) << "Zip operator child iterator produced empty row.";
      draining_ = true;
      new_zip_row->clear();
      // If we picked up an eof here, then we are completely done.
      if ((child_iterators_[i])->eof_handled()) {
        MS_LOG(DEBUG) << "Zip operator iterator got EOF.";
        eof_ = true;
      }
      return Status::OK();
    } else {
      MS_LOG(DEBUG) << "Zip operator got row from child " << i << ". Num cols: " << new_row.size() << ".";
      // if row isn't empty then we can append the fetched row with new_zip_row
      new_zip_row->insert(new_zip_row->end(), new_row.begin(), new_row.end());
    }
  }
  MS_LOG(DEBUG) << "Zip operator builds a zipped row. Number of columns in row: " << new_zip_row->size() << ".";
  return Status::OK();
}

// drain end of epoch messages from iterator for this epoch
Status ZipOp::drainPipeline() {
  // we don't need to drain if we reached eof
  if (eof_) {
    return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__,
                  "ZipOp draining should not be done if already at eof!");
  }
  for (int32_t con = 0; con < children_num_; ++con) {
    MS_LOG(DEBUG) << "Zip operator draining child at " << con << ".";
    RETURN_IF_NOT_OK(child_iterators_[con]->Drain());
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
    // Then show any custom derived-internal stuff
    out << "\nDatasets: " << children_num_ << "\n\n";
  }
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
    for (int32_t i = 0; i < child_.size(); ++i) {
      // Initializing col_name_id_map from the child.
      const std::unordered_map<std::string, int32_t> col_name_id_map = child_[i]->column_name_id_map();
      int32_t colsCurrent = column_name_id_map_.size();
      // the update code below shouldn't do anything bad if the column name already exists.
      for (const auto &pair : col_name_id_map) {
        std::string name = pair.first;
        int32_t old_id = pair.second;
        // check if name already exists in column name descriptor
        if (column_name_id_map_.count(name) == 1) {
          RETURN_STATUS_UNEXPECTED("Invalid parameter, key: " + name + " already exists when zipping datasets.");
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
}  // namespace dataset
}  // namespace mindspore
