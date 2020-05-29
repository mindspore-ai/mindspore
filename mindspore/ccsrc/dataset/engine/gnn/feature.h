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
#ifndef DATASET_ENGINE_GNN_FEATURE_H_
#define DATASET_ENGINE_GNN_FEATURE_H_

#include <memory>

#include "dataset/core/tensor.h"
#include "dataset/util/status.h"

namespace mindspore {
namespace dataset {
namespace gnn {
using FeatureType = int16_t;

class Feature {
 public:
  // Constructor
  // @param FeatureType type_name - feature type
  // @param std::shared_ptr<Tensor> value - feature value
  Feature(FeatureType type_name, std::shared_ptr<Tensor> value);

  // Get feature value
  // @return std::shared_ptr<Tensor> *out_value - feature value
  const std::shared_ptr<Tensor> Value() const { return value_; }

  // @return NodeIdType - Returned feature type
  FeatureType type() const { return type_name_; }

 private:
  FeatureType type_name_;
  std::shared_ptr<Tensor> value_;
};
}  // namespace gnn
}  // namespace dataset
}  // namespace mindspore
#endif  // DATASET_ENGINE_GNN_FEATURE_H_
