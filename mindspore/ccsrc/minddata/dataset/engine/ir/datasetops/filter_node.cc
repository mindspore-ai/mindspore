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

#include "minddata/dataset/engine/ir/datasetops/filter_node.h"

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/engine/datasetops/filter_op.h"
#include "minddata/dataset/engine/opt/pass.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

// Constructor for FilterNode
FilterNode::FilterNode(std::shared_ptr<DatasetNode> child, std::shared_ptr<TensorOp> predicate,
                       std::vector<std::string> input_columns)
    : predicate_(predicate), input_columns_(input_columns) {
  this->AddChild(child);
}

std::shared_ptr<DatasetNode> FilterNode::Copy() {
  auto node = std::make_shared<FilterNode>(nullptr, predicate_, input_columns_);
  return node;
}

void FilterNode::Print(std::ostream &out) const {
  out << Name() + "(<predicate>," + "input_cols:" + PrintColumns(input_columns_) + ")";
}

Status FilterNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  auto op = std::make_shared<FilterOp>(input_columns_, num_workers_, connector_que_size_, predicate_);
  op->set_total_repeats(GetTotalRepeats());
  op->set_num_repeats_per_epoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(op);
  return Status::OK();
}

Status FilterNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  if (predicate_ == nullptr) {
    std::string err_msg = "FilterNode: predicate is not specified.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (!input_columns_.empty()) {
    RETURN_IF_NOT_OK(ValidateDatasetColumnParam("FilterNode", "input_columns", input_columns_));
  }
  return Status::OK();
}

// Visitor accepting method for IRNodePass
Status FilterNode::Accept(IRNodePass *const p, bool *const modified) {
  // Downcast shared pointer then call visitor
  return p->Visit(shared_from_base<FilterNode>(), modified);
}

// Visitor accepting method for IRNodePass
Status FilterNode::AcceptAfter(IRNodePass *const p, bool *const modified) {
  // Downcast shared pointer then call visitor
  return p->VisitAfter(shared_from_base<FilterNode>(), modified);
}

Status FilterNode::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["input_columns"] = input_columns_;
  args["num_parallel_workers"] = num_workers_;
  args["predicate"] = "pyfunc";
  *out_json = args;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
