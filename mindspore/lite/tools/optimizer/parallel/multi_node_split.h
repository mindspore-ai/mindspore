/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_PARALLEL_MULTI_NODE_SPLIT_H
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_PARALLEL_MULTI_NODE_SPLIT_H
#include <utility>
#include <memory>
#include "tools/optimizer/parallel/split_strategy.h"
#include "schema/model_generated.h"
#include "include/errorcode.h"
#include "base/base.h"

using mindspore::schema::PrimitiveType;
namespace mindspore {
namespace opt {
class MultiNodeSplit {
 public:
  MultiNodeSplit() = default;

  virtual ~MultiNodeSplit() = default;

  virtual AnfNodePtr DoSplit(const FuncGraphPtr &func_graph, const AnfNodePtr &node) = 0;
};

class MultiNodeSplitProxy : public MultiNodeSplit {
 public:
  explicit MultiNodeSplitProxy(const SplitStrategy &strategy, PrimitiveType primitive_type, int32_t fmk_type = -1,
                               int32_t num = 3)
      : MultiNodeSplit(), strategy_(strategy), primitive_type_(primitive_type), fmk_type_(fmk_type), num_(num) {}

  ~MultiNodeSplitProxy() override = default;

  AnfNodePtr DoSplit(const FuncGraphPtr &func_graph, const AnfNodePtr &node) override;

 private:
  int InitResource();
  void FreeResource();

 private:
  SplitMode split_mode_{NoSplit};
  SplitStrategy strategy_{};
  PrimitiveType primitive_type_{schema::PrimitiveType_NONE};
  int32_t fmk_type_{-1};
  int32_t num_{0};
  std::shared_ptr<MultiNodeSplit> multi_node_split_{nullptr};
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_PARALLEL_MULTI_NODE_SPLIT_H
