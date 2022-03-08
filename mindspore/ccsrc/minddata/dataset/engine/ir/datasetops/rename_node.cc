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

#include "minddata/dataset/engine/ir/datasetops/rename_node.h"

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/engine/datasetops/rename_op.h"
#include "minddata/dataset/engine/opt/pass.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

// Function to build RenameOp
RenameNode::RenameNode(std::shared_ptr<DatasetNode> child, const std::vector<std::string> &input_columns,
                       const std::vector<std::string> &output_columns)
    : input_columns_(input_columns), output_columns_(output_columns) {
  this->AddChild(child);
}

std::shared_ptr<DatasetNode> RenameNode::Copy() {
  auto node = std::make_shared<RenameNode>(nullptr, input_columns_, output_columns_);
  return node;
}

void RenameNode::Print(std::ostream &out) const {
  out << (Name() + "(input:" + PrintColumns(input_columns_) + ",output:" + PrintColumns(output_columns_) + ")");
}

Status RenameNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  if (input_columns_.size() != output_columns_.size()) {
    std::string err_msg = "Rename: 'input columns' and 'output columns' must have the same size.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  RETURN_IF_NOT_OK(ValidateDatasetColumnParam("Rename", "input_columns", input_columns_));

  RETURN_IF_NOT_OK(ValidateDatasetColumnParam("Rename", "output_columns", output_columns_));

  return Status::OK();
}

Status RenameNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  auto op = std::make_shared<RenameOp>(input_columns_, output_columns_);
  op->SetTotalRepeats(GetTotalRepeats());
  op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(op);
  return Status::OK();
}

Status RenameNode::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["input_columns"] = input_columns_;
  args["output_columns"] = output_columns_;
  *out_json = args;
  return Status::OK();
}

Status RenameNode::from_json(nlohmann::json json_obj, std::shared_ptr<DatasetNode> ds,
                             std::shared_ptr<DatasetNode> *result) {
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "input_columns", kRenameNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "output_columns", kRenameNode));
  std::vector<std::string> input_columns = json_obj["input_columns"];
  std::vector<std::string> output_columns = json_obj["output_columns"];
  *result = std::make_shared<RenameNode>(ds, input_columns, output_columns);
  return Status::OK();
}

// Visitor accepting method for IRNodePass
Status RenameNode::Accept(IRNodePass *const p, bool *const modified) {
  // Downcast shared pointer then call visitor
  return p->Visit(shared_from_base<RenameNode>(), modified);
}

// Visitor accepting method for IRNodePass
Status RenameNode::AcceptAfter(IRNodePass *const p, bool *const modified) {
  // Downcast shared pointer then call visitor
  return p->VisitAfter(shared_from_base<RenameNode>(), modified);
}

}  // namespace dataset
}  // namespace mindspore
