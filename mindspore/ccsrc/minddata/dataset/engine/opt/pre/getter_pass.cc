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

#include "minddata/dataset/engine/opt/pre/getter_pass.h"
#include "minddata/dataset/engine/execution_tree.h"

namespace mindspore {
namespace dataset {
Status GetterPass::GetterNodes::RunOnNode(std::shared_ptr<ShuffleOp> node, bool *modified) {
  nodes_to_remove_.push_back(node);
  return Status::OK();
}

Status GetterPass::GetterNodes::RunOnNode(std::shared_ptr<RepeatOp> node, bool *modified) {
  if (type_ == kOutputShapeAndType) nodes_to_remove_.push_back(node);
  return Status::OK();
}

Status GetterPass::GetterNodes::RunOnNode(std::shared_ptr<SkipOp> node, bool *modified) {
  if (type_ == kOutputShapeAndType) nodes_to_remove_.push_back(node);
  return Status::OK();
}

Status GetterPass::GetterNodes::RunOnNode(std::shared_ptr<TakeOp> node, bool *modified) {
  if (type_ == kOutputShapeAndType) nodes_to_remove_.push_back(node);
  return Status::OK();
}

Status GetterPass::GetterNodes::RunOnNode(std::shared_ptr<MapOp> node, bool *modified) {
  nodes_to_clear_callback_.push_back(node);
  return Status::OK();
}

#ifdef ENABLE_PYTHON
Status GetterPass::GetterNodes::RunOnNode(std::shared_ptr<FilterOp> node, bool *modified) {
  if (type_ == kOutputShapeAndType) nodes_to_remove_.push_back(node);
  return Status::OK();
}
#endif

Status GetterPass::RunOnTree(ExecutionTree *tree, bool *modified) {
  RETURN_IF_NOT_OK(pass_.Run(tree, modified));

  // currently the getter pass only disables call_back from the execution tree

  // clear the callback for selected ops (map when its GetOutputType/Shape)
  for (auto node : pass_.nodes_to_clear_callback_) node->ClearCallbacks();

  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
