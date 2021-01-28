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

#include "minddata/dataset/engine/ir/datasetops/zip_node.h"

#include <memory>
#include <string>
#include <vector>
#include <algorithm>

#include "minddata/dataset/engine/datasetops/zip_op.h"
#include "minddata/dataset/engine/opt/pass.h"
#include "minddata/dataset/util/status.h"
namespace mindspore {
namespace dataset {

ZipNode::ZipNode(const std::vector<std::shared_ptr<DatasetNode>> &datasets) {
  nary_op_ = true;
  for (auto const &child : datasets) AddChild(child);
}

std::shared_ptr<DatasetNode> ZipNode::Copy() {
  std::vector<std::shared_ptr<DatasetNode>> empty_vector;
  empty_vector.clear();
  auto node = std::make_shared<ZipNode>(empty_vector);
  return node;
}

void ZipNode::Print(std::ostream &out) const { out << Name(); }

Status ZipNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  if (children_.size() < 2) {
    std::string err_msg = "ZipNode: input datasets are not specified.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (find(children_.begin(), children_.end(), nullptr) != children_.end()) {
    std::string err_msg = "ZipNode: input datasets should not be null.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

Status ZipNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  auto op = std::make_shared<ZipOp>(rows_per_buffer_, connector_que_size_);
  op->set_total_repeats(GetTotalRepeats());
  op->set_num_repeats_per_epoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(op);
  return Status::OK();
}

// Get Dataset size
Status ZipNode::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                               int64_t *dataset_size) {
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }
  std::vector<int32_t> dataset_sizes;
  int64_t child_dataset_size;
  for (auto child : children_) {
    RETURN_IF_NOT_OK(child->GetDatasetSize(size_getter, estimate, &child_dataset_size));
    dataset_sizes.push_back(child_dataset_size);
  }

  *dataset_size = *std::min_element(dataset_sizes.begin(), dataset_sizes.end());
  dataset_size_ = *dataset_size;
  return Status::OK();
}

// Visitor accepting method for IRNodePass
Status ZipNode::Accept(IRNodePass *const p, bool *const modified) {
  // Downcast shared pointer then call visitor
  return p->Visit(shared_from_base<ZipNode>(), modified);
}

// Visitor accepting method for IRNodePass
Status ZipNode::AcceptAfter(IRNodePass *const p, bool *const modified) {
  // Downcast shared pointer then call visitor
  return p->VisitAfter(shared_from_base<ZipNode>(), modified);
}

}  // namespace dataset
}  // namespace mindspore
