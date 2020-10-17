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

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "minddata/dataset/engine/consumers/tree_consumer.h"
#include "minddata/dataset/engine/tree_adapter.h"

namespace mindspore::dataset {

Status IteratorConsumer::GetNextAsVector(std::vector<TensorPtr> *out) {
  RETURN_UNEXPECTED_IF_NULL(out);
  out->clear();

  TensorRow res;
  RETURN_IF_NOT_OK(tree_adapter_->GetNext(&res));

  // Return empty vector if there's no data
  RETURN_OK_IF_TRUE(res.empty());

  std::copy(res.begin(), res.end(), std::back_inserter(*out));
  return Status::OK();
}
Status IteratorConsumer::GetNextAsMap(std::unordered_map<std::string, TensorPtr> *out_map) {
  RETURN_UNEXPECTED_IF_NULL(out_map);
  out_map->clear();

  TensorRow res;
  RETURN_IF_NOT_OK(tree_adapter_->GetNext(&res));

  // Return empty map if there's no data
  RETURN_OK_IF_TRUE(res.empty());

  // Populate the out map from the row and return it
  for (const auto &colMap : tree_adapter_->GetColumnNameMap()) {
    (*out_map)[colMap.first] = std::move(res[colMap.second]);
  }
  return Status::OK();
}

TreeConsumer::TreeConsumer() { tree_adapter_ = std::make_unique<TreeAdapter>(); }

Status IteratorConsumer::Init(std::shared_ptr<api::Dataset> d) {
  return tree_adapter_->BuildAndPrepare(std::move(d), num_epochs_);
}
Status TreeConsumer::Init(std::shared_ptr<api::Dataset> d) { return tree_adapter_->BuildAndPrepare(std::move(d)); }

Status ToDevice::Init(std::shared_ptr<api::Dataset> d) {
  // TODO(CRC):
  // Get device ID from children look at get_distribution in python
  // Add DeviceQue IR on top of dataset d

  return tree_adapter_->BuildAndPrepare(std::move(d), num_epochs_);
}
}  // namespace mindspore::dataset
