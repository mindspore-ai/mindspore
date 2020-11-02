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

#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"

#include <memory>

namespace mindspore {
namespace dataset {
namespace api {

Status DatasetNode::AddCacheOp(std::vector<std::shared_ptr<DatasetOp>> *node_ops) {
  if (cache_ != nullptr) {
    RETURN_IF_NOT_OK(cache_->Build());
    std::shared_ptr<DatasetOp> cache_op;
    RETURN_IF_NOT_OK(cache_->CreateCacheOp(num_workers_, &cache_op));
    node_ops->push_back(cache_op);
  }
  return Status::OK();
}
// Constructor to initialize the cache
DatasetNode::DatasetNode(const std::shared_ptr<DatasetCache> &dataset_cache) : DatasetNode() { cache_ = dataset_cache; }

std::shared_ptr<DatasetNode> DatasetNode::SetNumWorkers(int32_t num_workers) {
#if !defined(_WIN32) && !defined(_WIN64)
#ifndef ENABLE_ANDROID
  int32_t cpu_count = sysconf(_SC_NPROCESSORS_CONF);
  if (cpu_count < 0 || cpu_count > INT32_MAX) {
    MS_LOG(ERROR) << "Error determining current CPU: " << cpu_count;
    return nullptr;
  }
  if (num_workers < 1 || num_workers > cpu_count) {
    MS_LOG(ERROR) << "num_workers exceeds the boundary between 1 and " << cpu_count;
    return nullptr;
  }
#endif
#endif
  num_workers_ = num_workers;
  return shared_from_this();
}
DatasetNode::DatasetNode() {
  // Fetch some default value from config manager
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  num_workers_ = cfg->num_parallel_workers();
  rows_per_buffer_ = cfg->rows_per_buffer();
  connector_que_size_ = cfg->op_connector_size();
  worker_connector_size_ = cfg->worker_connector_size();
}

}  // namespace api
}  // namespace dataset
}  // namespace mindspore
