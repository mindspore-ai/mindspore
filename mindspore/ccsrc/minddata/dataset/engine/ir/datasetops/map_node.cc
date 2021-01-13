/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/engine/ir/datasetops/map_node.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/map_op/map_op.h"
#include "minddata/dataset/engine/opt/pass.h"
#include "minddata/dataset/include/transforms.h"
#include "minddata/dataset/util/status.h"
namespace mindspore {
namespace dataset {

MapNode::MapNode(std::shared_ptr<DatasetNode> child, std::vector<std::shared_ptr<TensorOperation>> operations,
                 std::vector<std::string> input_columns, std::vector<std::string> output_columns,
                 const std::vector<std::string> &project_columns, std::shared_ptr<DatasetCache> cache,
                 std::vector<std::shared_ptr<DSCallback>> callbacks)
    : operations_(operations),
      input_columns_(input_columns),
      output_columns_(output_columns),
      project_columns_(project_columns),
      DatasetNode(std::move(cache)),
      callbacks_(callbacks) {
  this->AddChild(child);
}

std::shared_ptr<DatasetNode> MapNode::Copy() {
  std::vector<std::shared_ptr<TensorOperation>> operations = operations_;
  auto node = std::make_shared<MapNode>(nullptr, operations, input_columns_, output_columns_, project_columns_, cache_,
                                        callbacks_);
  return node;
}

void MapNode::Print(std::ostream &out) const {
  out << Name() + "(<ops>" + ",input:" + PrintColumns(input_columns_) + ",output:" + PrintColumns(output_columns_) +
           ",<project_cols>" + ",num_tensor_ops:"
      << operations_.size() << ",...)";
}

Status MapNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  std::vector<std::shared_ptr<TensorOp>> tensor_ops;

  // Build tensorOp from tensorOperation vector
  // This is to ensure each iterator hold its own copy of the tensorOp objects.
  (void)std::transform(
    operations_.begin(), operations_.end(), std::back_inserter(tensor_ops),
    [](std::shared_ptr<TensorOperation> operation) -> std::shared_ptr<TensorOp> { return operation->Build(); });

  // This parameter will be removed with next rebase
  std::vector<std::string> col_orders;
  auto map_op = std::make_shared<MapOp>(input_columns_, output_columns_, tensor_ops, num_workers_, connector_que_size_);

  if (!callbacks_.empty()) {
    map_op->AddCallbacks(callbacks_);
  }

  if (!project_columns_.empty()) {
    auto project_op = std::make_shared<ProjectOp>(project_columns_);
    node_ops->push_back(project_op);
  }

  node_ops->push_back(map_op);
  return Status::OK();
}

Status MapNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  if (operations_.empty()) {
    std::string err_msg = "MapNode: No operation is specified.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  for (const auto &op : operations_) {
    RETURN_IF_NOT_OK(op->ValidateParams());
  }
  if (!input_columns_.empty()) {
    RETURN_IF_NOT_OK(ValidateDatasetColumnParam("MapNode", "input_columns", input_columns_));
  }

  if (!output_columns_.empty()) {
    RETURN_IF_NOT_OK(ValidateDatasetColumnParam("MapNode", "output_columns", output_columns_));
  }

  if (!project_columns_.empty()) {
    RETURN_IF_NOT_OK(ValidateDatasetColumnParam("MapNode", "project_columns", project_columns_));
  }

  return Status::OK();
}

// Visitor accepting method for IRNodePass
Status MapNode::Accept(IRNodePass *const p, bool *const modified) {
  // Downcast shared pointer then call visitor
  return p->Visit(shared_from_base<MapNode>(), modified);
}

// Visitor accepting method for IRNodePass
Status MapNode::AcceptAfter(IRNodePass *const p, bool *const modified) {
  // Downcast shared pointer then call visitor
  return p->VisitAfter(shared_from_base<MapNode>(), modified);
}

void MapNode::setOperations(const std::vector<std::shared_ptr<TensorOperation>> &operations) {
  operations_ = operations;
}
std::vector<std::shared_ptr<TensorOperation>> MapNode::operations() { return operations_; }

Status MapNode::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["num_parallel_workers"] = num_workers_;
  args["input_columns"] = input_columns_;
  args["output_columns"] = output_columns_;
  if (!project_columns_.empty()) args["column_order"] = project_columns_;
  if (cache_ != nullptr) {
    nlohmann::json cache_args;
    RETURN_IF_NOT_OK(cache_->to_json(&cache_args));
    args["cache"] = cache_args;
  }

  std::vector<nlohmann::json> ops;
  std::vector<int32_t> cbs;
  for (auto op : operations_) {
    nlohmann::json op_args;
    RETURN_IF_NOT_OK(op->to_json(&op_args));
    op_args["tensor_op_name"] = op->Name();
    ops.push_back(op_args);
  }
  args["operations"] = ops;
  std::transform(callbacks_.begin(), callbacks_.end(), std::back_inserter(cbs),
                 [](std::shared_ptr<DSCallback> cb) -> int32_t { return cb->step_size(); });
  args["callback"] = cbs;
  *out_json = args;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
