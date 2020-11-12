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

#include "minddata/dataset/engine/ir/datasetops/batch_node.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/batch_op.h"
#include "minddata/dataset/util/status.h"
namespace mindspore {
namespace dataset {

#ifdef ENABLE_PYTHON
// constructor #1, called by Pybind
BatchNode::BatchNode(std::shared_ptr<DatasetNode> child, int32_t batch_size, bool drop_remainder, bool pad,
                     const std::vector<std::string> &in_col_names, const std::vector<std::string> &out_col_names,
                     const std::vector<std::string> &col_order, py::function batch_size_func,
                     py::function batch_map_func,
                     std::map<std::string, std::pair<TensorShape, std::shared_ptr<Tensor>>> pad_map)
    : batch_size_(batch_size),
      drop_remainder_(drop_remainder),
      pad_(pad),
      in_col_names_(in_col_names),
      out_col_names_(out_col_names),
      col_order_(col_order),
      batch_size_func_(batch_size_func),
      batch_map_func_(batch_map_func),
      pad_map_(pad_map) {
  this->children.push_back(child);
}
#endif

// constructor #2, called by C++ API
BatchNode::BatchNode(std::shared_ptr<DatasetNode> child, int32_t batch_size, bool drop_remainder)
    : batch_size_(batch_size), drop_remainder_(drop_remainder), pad_(false) {
  this->children.push_back(child);
}

Status BatchNode::ValidateParams() {
  if (batch_size_ <= 0) {
    std::string err_msg = "BatchNode: batch_size should be positive integer, but got: " + std::to_string(batch_size_);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

#ifdef ENABLE_PYTHON
  if (batch_map_func_ && pad_) {
    std::string err_msg = "BatchNode: per_batch_map and pad should not be used at the same time.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (batch_map_func_ && in_col_names_.empty()) {
    std::string err_msg = "BatchNode: in_col_names cannot be empty when per_batch_map is used.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
#endif
  return Status::OK();
}

std::vector<std::shared_ptr<DatasetOp>> BatchNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

#ifdef ENABLE_PYTHON
  // if col_order_ isn't empty, then a project node needs to be attached after batch node. (same as map)
  // this means project_node needs to be the parent of batch_node. this means node_ops = [project_node, batch_node]
  if (!col_order_.empty()) {
    auto project_op = std::make_shared<ProjectOp>(col_order_);
    node_ops.push_back(project_op);
  }

  node_ops.push_back(std::make_shared<BatchOp>(batch_size_, drop_remainder_, pad_, connector_que_size_, num_workers_,
                                               in_col_names_, out_col_names_, batch_size_func_, batch_map_func_,
                                               pad_map_));
#else
  node_ops.push_back(std::make_shared<BatchOp>(batch_size_, drop_remainder_, pad_, connector_que_size_, num_workers_,
                                               in_col_names_, pad_map_));
#endif

  return node_ops;
}

}  // namespace dataset
}  // namespace mindspore
