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
Status PullBasedIteratorConsumer::Init(const std::shared_ptr<DatasetNode> &root) {
  return tree_adapter_lite_->Compile(root, num_epochs_);
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

  (void)std::copy(res.begin(), res.end(), std::back_inserter(*out));
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

TreeGetters::TreeGetters()
    : root_(nullptr),
      first_row_type_({}),
      first_row_shape_({}),
      estimated_row_shape_({}),
      init_flag_(false),
      first_row_obtained_(false) {
  tree_adapter_lite_ = std::make_unique<TreeAdapterLite>(TreeAdapterLite::UsageFlag::kDeGetter);
}

Status TreeGetters::Init(const std::shared_ptr<DatasetNode> &root) {
  root_ = root;
  return Status::OK();
}

Status TreeGetters::GetRow(TensorRow *row) {
  RETURN_UNEXPECTED_IF_NULL(row);
  Status get_next_status = tree_adapter_lite_->GetNextRow(row);
  return get_next_status;
}

Status TreeGetters::GetOutputTypes(std::vector<DataType> *types) {
  RETURN_UNEXPECTED_IF_NULL(types);
  RETURN_IF_NOT_OK(GetFirstRowShapeAndType());
  *types = first_row_type_;
  return Status::OK();
}

Status TreeGetters::GetOutputShapes(std::vector<TensorShape> *shapes, bool estimate) {
  RETURN_UNEXPECTED_IF_NULL(shapes);
  RETURN_IF_NOT_OK(GetFirstRowShapeAndType());
  *shapes = first_row_shape_;

  if (estimate) {
    estimated_row_shape_ = first_row_shape_;
    TensorRow row;
    RETURN_IF_NOT_OK(GetRow(&row));

    while (!row.empty()) {
      std::vector<TensorShape> cur_row_shape;
      (void)std::transform(row.begin(), row.end(), std::back_inserter(cur_row_shape),
                           [=](auto &t) { return t->shape(); });

      // calculate dynamic shape
      CHECK_FAIL_RETURN_SYNTAX_ERROR(cur_row_shape.size() == estimated_row_shape_.size(),
                                     "Inconsistent shapes, expect same shape for each data row, last data row: " +
                                       std::to_string(cur_row_shape.size()) +
                                       ", current data row: " + std::to_string(estimated_row_shape_.size()));
      size_t shape_size = cur_row_shape.size();
      std::vector<TensorShape> dynamic_shapes;
      for (size_t i = 0; i < shape_size; i++) {
        CHECK_FAIL_RETURN_SYNTAX_ERROR(
          cur_row_shape[i].Size() == estimated_row_shape_[i].Size(),
          "Inconsistent shapes, expect same shape for each data row, last data row: " + cur_row_shape[i].ToString() +
            ", current data row: " + estimated_row_shape_[i].ToString());

        std::vector<dsize_t> vec;
        for (size_t j = 0; j < estimated_row_shape_[i].Size(); j++) {
          dsize_t dim = cur_row_shape[i][j] == estimated_row_shape_[i][j] ? cur_row_shape[i][j] : -1;
          vec.push_back(dim);
        }
        dynamic_shapes.emplace_back(TensorShape(vec));
      }
      estimated_row_shape_ = dynamic_shapes;
      RETURN_IF_NOT_OK(GetRow(&row));
    }

    *shapes = estimated_row_shape_;
  }
  return Status::OK();
}

Status TreeGetters::GetBatchSize(int64_t *batch_size) {
  RETURN_UNEXPECTED_IF_NULL(batch_size);
  RETURN_IF_NOT_OK(InternalInit());
  std::shared_ptr<DatasetOp> root = std::shared_ptr<DatasetOp>(tree_adapter_lite_->GetRoot());
  RETURN_UNEXPECTED_IF_NULL(root);
  *batch_size = root->GetTreeBatchSize();
  CHECK_FAIL_RETURN_UNEXPECTED(*batch_size != 0, "GetBatchSize: Failed to find the batch size in Dataset pipeline.");
  return Status::OK();
}

Status TreeGetters::GetRepeatCount(int64_t *repeat_count) {
  RETURN_UNEXPECTED_IF_NULL(repeat_count);
  RETURN_IF_NOT_OK(InternalInit());
  std::shared_ptr<DatasetOp> root = std::shared_ptr<DatasetOp>(tree_adapter_lite_->GetRoot());
  RETURN_UNEXPECTED_IF_NULL(root);
  *repeat_count = root->GetTreeRepeatCount();
  return Status::OK();
}

Status TreeGetters::GetNumClasses(int64_t *num_classes) {
  RETURN_UNEXPECTED_IF_NULL(num_classes);
  RETURN_IF_NOT_OK(InternalInit());
  std::shared_ptr<DatasetOp> root = std::shared_ptr<DatasetOp>(tree_adapter_lite_->GetRoot());
  RETURN_UNEXPECTED_IF_NULL(root);
  RETURN_IF_NOT_OK(root->GetNumClasses(num_classes));
  return Status::OK();
}

Status TreeGetters::GetColumnNames(std::vector<std::string> *output) {
  RETURN_UNEXPECTED_IF_NULL(output);
  RETURN_IF_NOT_OK(InternalInit());
  std::shared_ptr<DatasetOp> root = std::shared_ptr<DatasetOp>(tree_adapter_lite_->GetRoot());
  RETURN_UNEXPECTED_IF_NULL(root);
  std::unordered_map<std::string, int32_t> column_name_id_map = root->column_name_id_map();
  CHECK_FAIL_RETURN_UNEXPECTED(!column_name_id_map.empty(), "GetColumnNames: column_name_id map can not be empty.");
  std::vector<std::pair<std::string, int32_t>> col_name_id_vec(column_name_id_map.begin(), column_name_id_map.end());
  std::sort(col_name_id_vec.begin(), col_name_id_vec.end(),
            [](const std::pair<std::string, int32_t> &a, const std::pair<std::string, int32_t> &b) {
              return a.second < b.second;
            });
  std::transform(col_name_id_vec.begin(), col_name_id_vec.end(), std::back_inserter(*output),
                 [](const std::pair<std::string, int32_t> &p) { return p.first; });
  return Status::OK();
}

Status TreeGetters::GetClassIndexing(std::vector<std::pair<std::string, std::vector<int32_t>>> *output_class_indexing) {
  RETURN_UNEXPECTED_IF_NULL(output_class_indexing);
  RETURN_IF_NOT_OK(InternalInit());
  std::shared_ptr<DatasetOp> root = std::shared_ptr<DatasetOp>(tree_adapter_lite_->GetRoot());
  RETURN_UNEXPECTED_IF_NULL(root);
  RETURN_IF_NOT_OK(root->GetClassIndexing(output_class_indexing));
  return Status::OK();
}

Status TreeGetters::InternalInit() {
  if (init_flag_) {
    return Status::OK();
  }

  Status s = tree_adapter_lite_->Compile(root_, 1);
  if (s.IsOk()) {
    init_flag_ = true;
  }
  return s;
}

Status TreeGetters::GetFirstRowShapeAndType() {
  RETURN_OK_IF_TRUE(first_row_obtained_);
  RETURN_IF_NOT_OK(InternalInit());
  TensorRow first_row;
  RETURN_IF_NOT_OK(GetRow(&first_row));
  std::transform(first_row.begin(), first_row.end(), std::back_inserter(first_row_type_),
                 [](const TensorPtr &t) { return t->type(); });
  std::transform(first_row.begin(), first_row.end(), std::back_inserter(first_row_shape_),
                 [](const TensorPtr &t) { return t->shape(); });
  first_row_obtained_ = true;
  return Status::OK();
}
}  // namespace mindspore::dataset
