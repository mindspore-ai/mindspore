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
#include "minddata/dataset/engine/datasetops/concat_op.h"

#include <iomanip>
#include <utility>

#include "minddata/dataset/core/config_manager.h"

#include "minddata/dataset/engine/db_connector.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace dataset {
// Builder constructor. Creates the builder object.
ConcatOp::Builder::Builder() {
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  builder_op_connector_size_ = cfg->op_connector_size();
  builder_sampler_ = nullptr;
}

// The builder "build" method creates the final object.
Status ConcatOp::Builder::Build(std::shared_ptr<ConcatOp> *ptr) {
  if (builder_sampler_ == nullptr) {
    builder_sampler_ = std::make_shared<DistributedSamplerRT>(0, 1, 0, false);
  }
  *ptr = std::make_shared<ConcatOp>(builder_op_connector_size_, builder_sampler_, children_flag_and_nums_,
                                    children_start_end_index_);
  return Status::OK();
}

// Constructor of the ConcatOp.
ConcatOp::ConcatOp(int32_t op_connector_size, const std::shared_ptr<SamplerRT> &sampler,
                   const std::vector<std::pair<int, int>> &children_flag_and_nums,
                   const std::vector<std::pair<int, int>> &children_start_end_index)
    : PipelineOp(op_connector_size),
      children_num_(0),
      sampler_(sampler),
      children_flag_and_nums_(children_flag_and_nums),
      children_start_end_index_(children_start_end_index) {}

ConcatOp::ConcatOp(int32_t op_connector_size) : PipelineOp(op_connector_size), children_num_(0) {}

// A function that prints info about the Operator
void ConcatOp::Print(std::ostream &out, bool show_all) const {
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

// This definition is added to pass the cyclomatic complexity rule of <= 20 units
// The NOLINT directive is to disable cpplint check.
// Clang format and cpplint give conflicting recommendations on this line below.
#define f(fv, sv, shard_index)                                                                     \
  (((fv) == -1 && (sv) == -1) || ((fv) < (sv) && (shard_index) >= (fv) && (shard_index) < (sv)) || \
   ((fv) > (sv) && ((shard_index) >= (fv) || (shard_index) < (sv))))  // NOLINT

// Main entry point for Concat
Status ConcatOp::operator()() {
  TaskManager::FindMe()->Post();
  children_num_ = static_cast<int32_t>(child_.size());
  for (int32_t i = 0; i < children_num_; i++) {
    children_iterators_.push_back(std::make_unique<ChildIterator>(this, 0, i));
  }
  TensorRow new_row;
  int eof_count = 0;
  int sample_number = 0;
  bool is_not_mappable = true;
  bool is_not_mappable_or_second_ne_zero = true;
  int num_shard = 1;
  int shard_index = 0;
  std::shared_ptr<DistributedSamplerRT> distribute_sampler = std::dynamic_pointer_cast<DistributedSamplerRT>(sampler_);
  if (distribute_sampler != nullptr) {
    num_shard = distribute_sampler->GetDeviceNum();
    shard_index = distribute_sampler->GetDeviceID();
  }

  while (eof_count == 0) {
    for (int i = 0; i < children_num_; i++) {
      // 1. Read the first row
      RETURN_IF_NOT_OK(children_iterators_[i]->FetchNextTensorRow(&new_row));
      if (new_row.eof()) {
        eof_count++;
        continue;
      }
      // 2. Do verification as for column name, column data type and rank of column data
      if (!new_row.eoe()) {
        RETURN_IF_NOT_OK(Verify(i, new_row));
      }
      // 3. Put the data into output_connector
      if (!children_flag_and_nums_.empty()) {
        is_not_mappable = children_flag_and_nums_[i].first;
        is_not_mappable_or_second_ne_zero = is_not_mappable || (!children_flag_and_nums_[i].second);
      }
      while (!new_row.eoe() && !new_row.eof()) {
        // if dataset is not mappable or generator dataset which source is yield, cannot get the number of samples in
        // python layer), we use filtering to get data
        if (sample_number % num_shard == shard_index && is_not_mappable_or_second_ne_zero) {
          RETURN_IF_NOT_OK(out_connector_->Add(std::move(new_row)));
        } else if (!is_not_mappable_or_second_ne_zero) {
          // if dataset is mappable or generator dataset which source is not yield,
          // get the start and end subscripts of valid values
          int fv = children_start_end_index_[i].first, sv = children_start_end_index_[i].second;

          // determine whether the data allocated to the current shard id is false data
          if (f(fv, sv, shard_index)) {
            RETURN_IF_NOT_OK(out_connector_->Add(std::move(new_row)));
          }
        }

        // if dataset is not mappable or generator dataset which source is yield, sample_number+=1
        if (is_not_mappable_or_second_ne_zero) {
          sample_number++;
        }

        RETURN_IF_NOT_OK(children_iterators_[i]->FetchNextTensorRow(&new_row));
      }

      // if dataset is mappable,We don't use filtering to pick data.
      // so sample_number plus the length of the entire dataset
      if (!is_not_mappable_or_second_ne_zero) {
        sample_number += children_flag_and_nums_[i].second;
      }
    }

    // 4. Add eoe row after get rows from all child
    if (eof_count == 0) {
      RETURN_IF_NOT_OK(out_connector_->SendEOE());
    }
    UpdateRepeatAndEpochCounter();
  }
  CHECK_FAIL_RETURN_UNEXPECTED(eof_count == children_num_,
                               "Something went wrong, eof count does not match the number of children.");
  // 5. Add eof row in the end manually
  MS_LOG(DEBUG) << "Add the eof row manually in the end.";
  RETURN_IF_NOT_OK(out_connector_->SendEOF());
  return Status::OK();
}

Status ConcatOp::Verify(int32_t id, const TensorRow &new_row) {
  if (id == 0) {
    // Obtain the data type and data rank in child[0]
    for (auto item : new_row) {
      data_type_.push_back(item->type());
      data_rank_.push_back(item->Rank());
    }
  } else {
    // Compare the data type and data rank with these in child[0]
    int32_t index = 0;
    for (auto item : new_row) {
      if ((item->type() != data_type_[index]) || item->Rank() != data_rank_[index++]) {
        RETURN_STATUS_UNEXPECTED("Invalid data, data type or data rank is not the same with previous dataset.");
      }
    }
  }
  return Status::OK();
}

// We need to overwrite the super class ComputeColMap here because the number of children is more than 1.
Status ConcatOp::ComputeColMap() {
  if (column_name_id_map_.empty()) {
    // Obtain columns_name_id_map from child_[0]
    column_name_id_map_ = child_[0]->column_name_id_map();
    if (column_name_id_map_.empty()) {
      RETURN_STATUS_UNEXPECTED("Child column name map cannot be empty!");
    }
    // Verify all children have the same column name map
    for (int32_t i = 0; i < child_.size(); ++i) {
      if (child_[i]->column_name_id_map() != column_name_id_map_) {
        RETURN_STATUS_UNEXPECTED("Invalid data, column name or column order is not the same with previous dataset.");
      }
    }
  } else {
    MS_LOG(WARNING) << "Column name map is already set!";
  }
  return Status::OK();
}

// Gets the number of classes
Status ConcatOp::GetNumClasses(int64_t *num_classes) {
  int64_t max_num_classes = -1;
  for (const auto &child : child_) {
    // Choose a dataset which can get valid num_classes
    int64_t tmp_num_classes = -1;
    child->GetNumClasses(&tmp_num_classes);
    if (tmp_num_classes > max_num_classes) {
      max_num_classes = tmp_num_classes;
    }
  }
  *num_classes = max_num_classes;
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
