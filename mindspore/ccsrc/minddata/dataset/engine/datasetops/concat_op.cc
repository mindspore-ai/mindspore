/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "utils/ms_utils.h"

namespace mindspore {
namespace dataset {
// Constructor of the ConcatOp.
ConcatOp::ConcatOp(const std::shared_ptr<SamplerRT> &sampler,
                   const std::vector<std::pair<int, int>> &children_flag_and_nums,
                   const std::vector<std::pair<int, int>> &children_start_end_index)
    : ConcatOp() {
  children_flag_and_nums_ = children_flag_and_nums;
  children_start_end_index_ = children_start_end_index;
  std::shared_ptr<DistributedSamplerRT> distribute_sampler = std::dynamic_pointer_cast<DistributedSamplerRT>(sampler);
  if (distribute_sampler != nullptr) {
    num_shard_ = static_cast<int32_t>(distribute_sampler->GetDeviceNum());
    shard_index_ = static_cast<int32_t>(distribute_sampler->GetDeviceID());
  }
}

ConcatOp::ConcatOp()
    : PipelineOp(0), cur_child_(0), verified_(false), sample_number_(0), num_shard_(1), shard_index_(0) {}

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
    out << "\nDatasets: " << child_.size() << "\n\n";
  }
}

// This definition is added to pass the cyclomatic complexity rule of <= 20 units
// The NOLINT directive is to disable cpplint check.
// Clang format and cpplint give conflicting recommendations on this line below.
#define f(fv, sv, shard_index)                                                                     \
  (((fv) == -1 && (sv) == -1) || ((fv) < (sv) && (shard_index) >= (fv) && (shard_index) < (sv)) || \
   ((fv) > (sv) && ((shard_index) >= (fv) || (shard_index) < (sv))))  // NOLINT

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
      if (item->type() != data_type_[index]) {
        RETURN_STATUS_UNEXPECTED(
          "Concat: the data types of the two datasets to be concatenated should be the same, but got: " +
          data_type_[index].ToString() + " and " + item->type().ToString() + ".");
      }
      if (item->Rank() != data_rank_[index]) {
        RETURN_STATUS_UNEXPECTED(
          "Concat: the data tensor rank of the two datasets to be concatenated should be the same, but got: " +
          std::to_string(data_rank_[index]) + " and " + std::to_string(item->Rank()) + ".");
      }
      index++;
    }
  }
  verified_ = true;
  return Status::OK();
}

// We need to overwrite the super class ComputeColMap here because the number of children is more than 1.
Status ConcatOp::ComputeColMap() {
  if (column_name_id_map_.empty()) {
    // Obtain columns_name_id_map from child_[0]
    column_name_id_map_ = child_[0]->column_name_id_map();
    if (column_name_id_map_.empty()) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] Child column name map cannot be empty!");
    }
    // Verify all children have the same column name map
    for (size_t i = 0; i < child_.size(); ++i) {
      if (child_[i]->column_name_id_map() != column_name_id_map_) {
        RETURN_STATUS_UNEXPECTED(
          "Invalid columns, 'column name' or 'column order' of concat datasets should be the same.");
      }
    }
  } else {
    MS_LOG(WARNING) << "Column name map is already set!";
  }
  return Status::OK();
}

// Gets the number of classes
Status ConcatOp::GetNumClasses(int64_t *num_classes) {
  RETURN_UNEXPECTED_IF_NULL(num_classes);
  int64_t max_num_classes = -1;
  for (const auto &child : child_) {
    // Choose a dataset which can get valid num_classes
    int64_t tmp_num_classes = -1;
    RETURN_IF_NOT_OK(child->GetNumClasses(&tmp_num_classes));
    if (tmp_num_classes > max_num_classes) {
      max_num_classes = tmp_num_classes;
    }
  }
  *num_classes = max_num_classes;
  return Status::OK();
}
Status ConcatOp::operator()() { RETURN_STATUS_UNEXPECTED("[Internal ERROR] ConcatOp is an inlined operator."); }

bool ConcatOp::IgnoreSample() {
  bool is_not_mappable_or_second_ne_zero = true;

  if (!children_flag_and_nums_.empty()) {
    const bool is_not_mappable = children_flag_and_nums_[cur_child_].first != 0 ? true : false;
    const bool second_ne_zero = children_flag_and_nums_[cur_child_].second == 0 ? true : false;
    is_not_mappable_or_second_ne_zero = is_not_mappable || second_ne_zero;
  }
  bool ret = true;
  if (sample_number_ % num_shard_ == shard_index_ && is_not_mappable_or_second_ne_zero) {
    ret = false;
  } else if (!is_not_mappable_or_second_ne_zero) {
    // if dataset is mappable or generator dataset which source is not yield,
    // get the start and end subscripts of valid values
    int fv = children_start_end_index_[cur_child_].first, sv = children_start_end_index_[cur_child_].second;

    // determine whether the data allocated to the current shard id is false data
    if (f(fv, sv, shard_index_)) {
      ret = false;
    }
  }

  if (is_not_mappable_or_second_ne_zero) {
    sample_number_++;
  }
  return ret;
}

Status ConcatOp::GetNextRow(TensorRow *row) {
  RETURN_UNEXPECTED_IF_NULL(row);
  bool is_not_mappable_or_second_ne_zero = true;

  if (!children_flag_and_nums_.empty()) {
    const bool is_not_mappable = children_flag_and_nums_[cur_child_].first != 0 ? true : false;
    const bool second_ne_zero = children_flag_and_nums_[cur_child_].second == 0 ? true : false;
    is_not_mappable_or_second_ne_zero = is_not_mappable || second_ne_zero;
  }

  RETURN_IF_NOT_OK(child_[cur_child_]->GetNextRow(row));

  if (!row->eoe() && !row->eof()) {
    if (!verified_) {
      RETURN_IF_NOT_OK(Verify(static_cast<int32_t>(cur_child_), *row));
    }

    if (IgnoreSample()) {
      RETURN_IF_NOT_OK(GetNextRow(row));
    }

    return Status::OK();
  }
  if (row->eoe()) {
    // if last child, send out eoe and reset epoch
    if (cur_child_ == child_.size() - 1) {
      // reset
      cur_child_ = 0;
      verified_ = false;
      UpdateRepeatAndEpochCounter();
      return Status::OK();
    }
    if (!is_not_mappable_or_second_ne_zero) {
      sample_number_ += children_flag_and_nums_[cur_child_].second;
    }
    cur_child_++;
    verified_ = false;
    RETURN_IF_NOT_OK(GetNextRow(row));
    return Status::OK();
  }
  if (row->eof()) {
    CHECK_FAIL_RETURN_UNEXPECTED(cur_child_ == 0, "[Internal ERROR] Received an unexpected EOF.");
    for (size_t i = cur_child_ + 1; i < child_.size(); i++) {
      RETURN_IF_NOT_OK(child_[i]->GetNextRow(row));
      CHECK_FAIL_RETURN_UNEXPECTED(row->eof(), "[Internal ERROR] Row must be an EOF.");
    }
    return Status::OK();
  }

  return Status::OK();
}

Status ConcatOp::GetNextRowPullMode(TensorRow *const row) {
  RETURN_UNEXPECTED_IF_NULL(row);
  // Reset TensorRow (both vector and flags)
  row->reset();
  bool is_not_mappable_or_second_ne_zero = true;

  if (!children_flag_and_nums_.empty()) {
    const bool is_not_mappable = children_flag_and_nums_[cur_child_].first != 0 ? true : false;
    const bool second_ne_zero = children_flag_and_nums_[cur_child_].second == 0 ? true : false;
    is_not_mappable_or_second_ne_zero = is_not_mappable || second_ne_zero;
  }
  RETURN_IF_NOT_OK(child_[cur_child_]->GetNextRowPullMode(row));

  if (row->eoe()) {
    // if last child, send out eoe and reset epoch
    if (cur_child_ == child_.size() - 1) {
      // reset
      cur_child_ = 0;
      verified_ = false;
      UpdateRepeatAndEpochCounter();
      return Status::OK();
    }
    if (!is_not_mappable_or_second_ne_zero) {
      sample_number_ += children_flag_and_nums_[cur_child_].second;
    }
    cur_child_++;
    verified_ = false;
    RETURN_IF_NOT_OK(GetNextRowPullMode(row));
  } else {
    if (!verified_) {
      RETURN_IF_NOT_OK(Verify(static_cast<int32_t>(cur_child_), *row));
    }
    if (IgnoreSample()) {
      RETURN_IF_NOT_OK(GetNextRowPullMode(row));
    }
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
