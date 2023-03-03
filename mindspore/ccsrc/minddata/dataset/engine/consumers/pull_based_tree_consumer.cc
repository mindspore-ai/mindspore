/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/engine/consumers/pull_based_tree_consumer.h"

#include <algorithm>

namespace mindspore::dataset {
Status PullBasedIteratorConsumer::Init(std::shared_ptr<DatasetNode> root) {
  RETURN_UNEXPECTED_IF_NULL(root);
  return tree_adapter_lite_->Compile(std::move(root), num_epochs_);
}

std::vector<TensorRow> PullBasedIteratorConsumer::GetRows(int64_t num_rows) {
  std::vector<TensorRow> rows;
  for (int i = 0; i < num_rows; i++) {
    TensorRow row;
    RETURN_SECOND_IF_ERROR(tree_adapter_lite_->GetNextRow(&row), {});
    if (row.empty()) {
      break;
    }
    rows.push_back(row);
  }

  return rows;
}

Status PullBasedIteratorConsumer::GetNextAsVector(std::vector<TensorPtr> *const out) {
  RETURN_UNEXPECTED_IF_NULL(out);
  out->clear();

  TensorRow res;
  RETURN_IF_NOT_OK(tree_adapter_lite_->GetNextRow(&res));

  // Return empty vector if there's no data
  RETURN_OK_IF_TRUE(res.empty());

  std::copy(res.begin(), res.end(), std::back_inserter(*out));
  return Status::OK();
}

Status PullBasedIteratorConsumer::GetNextAsMap(std::unordered_map<std::string, TensorPtr> *const out_map) {
  RETURN_UNEXPECTED_IF_NULL(out_map);
  out_map->clear();

  TensorRow res;
  RETURN_IF_NOT_OK(tree_adapter_lite_->GetNextRow(&res));

  // Return empty map if there's no data
  RETURN_OK_IF_TRUE(res.empty());

  // Populate the out map from the row and return it
  for (const auto &colMap : tree_adapter_lite_->GetColumnNameMap()) {
    (*out_map)[colMap.first] = std::move(res[colMap.second]);
  }
  return Status::OK();
}

Status PullBasedIteratorConsumer::GetNextAsOrderedPair(
  std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> *const vec) {
  CHECK_FAIL_RETURN_UNEXPECTED(vec != nullptr && vec->empty(), "vec is null or non-empty.");

  TensorRow curr_row;

  RETURN_IF_NOT_OK(tree_adapter_lite_->GetNextRow(&curr_row));
  RETURN_OK_IF_TRUE(curr_row.empty());
  size_t num_cols = curr_row.size();  // num_cols is non-empty.
  // order the column names according to their ids
  if (column_order_.empty()) {
    const int32_t invalid_col_id = -1;
    column_order_.resize(num_cols, {std::string(), invalid_col_id});
    for (const auto &itr : tree_adapter_lite_->GetColumnNameMap()) {
      int32_t ind = itr.second;
      CHECK_FAIL_RETURN_UNEXPECTED(ind < num_cols && ind >= 0, "column id out of bounds. Expecting in the range [0," +
                                                                 std::to_string(num_cols) + "), but got " +
                                                                 std::to_string(ind));
      column_order_[ind] = std::make_pair(itr.first, ind);
    }
    // error check, make sure the ids in col_name_id_map are continuous and starts from 0
    for (const auto &col : column_order_) {
      if (col.second == invalid_col_id) {
        std::string err_msg = "Invalid column id encountered.";
        err_msg += " Note: It is unsupported and ambiguous to reuse the same column name for an output_column name";
        err_msg += " if it is an input_column name that will already appear as one of the output columns.";
        err_msg += " Use unique columns names.";
        MS_LOG(ERROR) << err_msg;
        RETURN_STATUS_UNEXPECTED(err_msg);
      }
    }
  }
  vec->reserve(num_cols);

  std::transform(column_order_.begin(), column_order_.end(), std::back_inserter(*vec),
                 [curr_row](const auto &col) { return std::make_pair(col.first, curr_row[col.second]); });

  return Status::OK();
}
}  // namespace mindspore::dataset
