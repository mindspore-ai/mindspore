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

#include "dataset/engine/datasetops/project_op.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "dataset/engine/data_buffer.h"
#include "dataset/engine/db_connector.h"
#include "dataset/engine/execution_tree.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace dataset {
ProjectOp::Builder::Builder(const std::vector<std::string> &columns_to_project)
    : builder_columns_to_project_(columns_to_project) {}

Status ProjectOp::Builder::SanityCheck() const {
  if (builder_columns_to_project_.empty()) {
    std::string err_msg("Columns to project is empty.");
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}

Status ProjectOp::Builder::Build(std::shared_ptr<ProjectOp> *ptr) {
  RETURN_IF_NOT_OK(SanityCheck());
  *ptr = std::make_shared<ProjectOp>(builder_columns_to_project_);
  return Status::OK();
}

ProjectOp::ProjectOp(const std::vector<std::string> &columns_to_project)
    : PipelineOp(0), columns_to_project_(columns_to_project) {}

void ProjectOp::Print(std::ostream &out, bool show_all) const {
  PipelineOp::Print(out, show_all);
  out << "ProjectOp: columns that are projected: ";
  for (size_t i = 0; i < columns_to_project_.size(); i++) {
    out << columns_to_project_[i] << " ";
  }
  out << '\n';
}

// Gets a buffer from the child operator and projects the buffer.
Status ProjectOp::GetNextBuffer(std::unique_ptr<DataBuffer> *p_buffer, int32_t worker_id, bool retry_if_eoe) {
  RETURN_IF_NOT_OK(child_[0]->GetNextBuffer(p_buffer, worker_id, retry_if_eoe));
  if (!((*p_buffer)->eoe()) && !((*p_buffer)->eof())) {
    RETURN_IF_NOT_OK(Project(p_buffer));
  }
  return Status::OK();
}

Status ProjectOp::Project(std::unique_ptr<DataBuffer> *data_buffer) {
  std::unordered_map<std::string, int32_t> column_name_mapping = (*data_buffer)->column_name_map();
  std::unordered_map<std::string, int32_t> new_column_name_mapping;
  std::vector<int32_t> projected_column_indices;
  for (size_t i = 0; i < columns_to_project_.size(); i++) {
    std::string &current_column = columns_to_project_[i];
    if (column_name_mapping.find(current_column) == column_name_mapping.end()) {
      std::string err_msg = "ProjectOp: column " + current_column + " does not exist in this buffer.";
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
    new_column_name_mapping[current_column] = i;
    projected_column_indices.push_back(column_name_mapping[current_column]);
  }
  std::unique_ptr<TensorQTable> new_tensor_table = std::make_unique<TensorQTable>();
  while ((*data_buffer)->NumRows() > 0) {
    TensorRow current_row;
    RETURN_IF_NOT_OK((*data_buffer)->PopRow(&current_row));
    TensorRow new_row;
    (void)std::transform(projected_column_indices.begin(), projected_column_indices.end(), std::back_inserter(new_row),
                         [&current_row](uint32_t x) { return current_row[x]; });
    new_tensor_table->push_back(new_row);
  }
  (*data_buffer)->set_tensor_table(std::move(new_tensor_table));
  (*data_buffer)->set_column_name_map(new_column_name_mapping);
  return Status::OK();
}

// Class functor operator () override.
// Most dataset ops operate by launching a thread (see ExecutionTree).
// However, the ProjectOp is defined as a inlined operator, so it is invalid to launch the
// functor since this op runs inlined inside another operator. The function is overloaded to
// ensure that it is not called by mistake (it will generate an error).
Status ProjectOp::operator()() { RETURN_STATUS_UNEXPECTED("Logic error. ProjectOp is an inlined operator."); }

int32_t ProjectOp::num_consumers() const {
  if (parent_.empty()) {
    MS_LOG(INFO) << "Project operator, no parent node, assuming it's the root and returning 1.";
    return 1;
  } else if (parent_[0] == nullptr) {
    MS_LOG(INFO) << "Project operator, pointer to the first parent is null. Returning 0.";
    return 0;
  } else {
    return parent_[0]->num_consumers();
  }
}

int32_t ProjectOp::num_producers() const {
  if (child_.empty() || child_[0] == nullptr) {
    MS_LOG(INFO) << "Project operator, pointer to child node is null. Returning 0.";
    return 0;
  } else {
    return child_[0]->num_producers();
  }
}

Status ProjectOp::EoeReceived(int32_t worker_id) {
  state_ = OpState::kDeOpIdle;
  return Status::OK();
}

Status ProjectOp::EofReceived(int32_t worker_id) { return Status::OK(); }
}  // namespace dataset
}  // namespace mindspore
