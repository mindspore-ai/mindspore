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

#include "plugin/device/gpu/kernel/data/dataset_utils.h"
#include <algorithm>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace kernel {
int ElementNums(const std::vector<int> &shape) {
  if (shape.size() == 0) {
    return 0;
  }

  int nums = 1;
  for (size_t i = 0; i < shape.size(); i++) {
    nums *= shape[i];
  }

  return nums;
}

void GetShapeAndType(const CNodePtr &kernel_node, std::vector<std::vector<int>> *shapes, std::vector<TypePtr> *types) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  MS_EXCEPTION_IF_NULL(shapes);
  MS_EXCEPTION_IF_NULL(types);
  std::vector<std::vector<int64_t>> shapes_me;
  shapes_me = common::AnfAlgo::GetNodeAttr<std::vector<std::vector<int64_t>>>(kernel_node, "shapes");
  (void)std::transform(shapes_me.begin(), shapes_me.end(), std::back_inserter(*shapes),
                       [](const std::vector<int64_t> &values) {
                         std::vector<int> shape;
                         (void)std::transform(values.begin(), values.end(), std::back_inserter(shape),
                                              [](const int64_t &value) { return static_cast<int>(value); });
                         // Empty means scalar. Push one elements for bytes calculation.
                         if (shape.empty()) {
                           shape.push_back(1);
                         }
                         return shape;
                       });

  *types = common::AnfAlgo::GetNodeAttr<std::vector<TypePtr>>(kernel_node, "types");
  if (shapes->size() != types->size()) {
    MS_LOG(EXCEPTION) << "Invalid shapes: " << *shapes << ", types: " << *types;
  }
}
}  // namespace kernel
}  // namespace mindspore
