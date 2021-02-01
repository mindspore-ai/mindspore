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

#include "minddata/dataset/engine/ir/datasetops/project_node.h"

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/engine/datasetops/project_op.h"

#include "minddata/dataset/util/status.h"
namespace mindspore {
namespace dataset {

// Function to build ProjectOp
ProjectNode::ProjectNode(std::shared_ptr<DatasetNode> child, const std::vector<std::string> &columns)
    : columns_(columns) {
  this->AddChild(child);
}

std::shared_ptr<DatasetNode> ProjectNode::Copy() {
  auto node = std::make_shared<ProjectNode>(nullptr, this->columns_);
  return node;
}

void ProjectNode::Print(std::ostream &out) const { out << Name() + "(column: " + PrintColumns(columns_) + ")"; }

Status ProjectNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  if (columns_.empty()) {
    std::string err_msg = "ProjectNode: No columns are specified.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  RETURN_IF_NOT_OK(ValidateDatasetColumnParam("ProjectNode", "columns", columns_));

  return Status::OK();
}

Status ProjectNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  auto op = std::make_shared<ProjectOp>(columns_);
  op->set_total_repeats(GetTotalRepeats());
  op->set_num_repeats_per_epoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(op);
  return Status::OK();
}

Status ProjectNode::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["columns"] = columns_;
  *out_json = args;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
