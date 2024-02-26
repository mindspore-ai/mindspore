/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_SPLIT_RESHAPE_AND_CACHE
#define MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_SPLIT_RESHAPE_AND_CACHE

#include <memory>
#include "include/backend/optimizer/optimizer.h"
#include "backend/common/graph_kernel/core/graph_kernel_expander.h"
#include "backend/common/graph_kernel/core/split_umonad.h"

namespace mindspore::graphkernel {

class SplitReshapeAndCache : public SplitNode {
 public:
  /**
   * @brief This pass will split umonad from ReshapeAndCache inputs and add Depend node.
   */
  explicit SplitReshapeAndCache(bool multigraph = true) : SplitNode("split_reshape_and_cache", multigraph) {}
  ~SplitReshapeAndCache() override = default;
  const BaseRef DefinePattern() const override;
  const bool CanSplit(const AnfNodePtr &node) const override;
};

}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_SPLIT_RESHAPE_AND_CACHE
