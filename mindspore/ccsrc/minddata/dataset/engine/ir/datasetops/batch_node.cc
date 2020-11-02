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
namespace api {

BatchNode::BatchNode(std::shared_ptr<DatasetNode> child, int32_t batch_size, bool drop_remainder, bool pad,
                     std::vector<std::string> cols_to_map,
                     std::map<std::string, std::pair<TensorShape, std::shared_ptr<Tensor>>> pad_map)
    : batch_size_(batch_size),
      drop_remainder_(drop_remainder),
      pad_(pad),
      cols_to_map_(cols_to_map),
      pad_map_(pad_map) {
  this->children.push_back(child);
}

Status BatchNode::ValidateParams() {
  if (batch_size_ <= 0) {
    std::string err_msg = "BatchNode: batch_size should be positive integer, but got: " + std::to_string(batch_size_);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (!cols_to_map_.empty()) {
    std::string err_msg = "BatchNode: cols_to_map functionality is not implemented in C++; this should be left empty.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::vector<std::shared_ptr<DatasetOp>> BatchNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

#ifdef ENABLE_PYTHON
  py::function noop;
  node_ops.push_back(std::make_shared<BatchOp>(batch_size_, drop_remainder_, pad_, connector_que_size_, num_workers_,
                                               cols_to_map_, cols_to_map_, noop, noop, pad_map_));
#else
  node_ops.push_back(std::make_shared<BatchOp>(batch_size_, drop_remainder_, pad_, connector_que_size_, num_workers_,
                                               cols_to_map_, pad_map_));
#endif

  // Until py::function is implemented for C++ API, there is no need for a project op to be inserted after batch
  // because project is only needed when batch op performs per_batch_map. This per_batch_map is a pyfunc
  return node_ops;
}

}  // namespace api
}  // namespace dataset
}  // namespace mindspore
