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
#ifndef MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_SPLIT_UMONAD_H
#define MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_SPLIT_UMONAD_H

#include <memory>
#include <string>
#include "include/backend/optimizer/optimizer.h"
#include "tools/optimizer/common/pattern_process_pass_extends.h"

namespace mindspore::graphkernel {

class SplitNodeLite : public opt::LitePatternProcessPass {
  /**
   * @brief This is An abstract class.
   *        This pass will split umonad from node inputs and add Depend node.
   * @example
   *  %1 = op1
   *  %2 = op2
   *  %3 = op(%1, %2, umonad)
   * -->
   *  %1 = op1
   *  %2 = op2
   *  %3 = Depend(%1,umonad)
   *  %4 = op(%1, %2, %3)
   */
 public:
  explicit SplitNodeLite(std::string pass_name, bool multigraph = true)
      : opt::LitePatternProcessPass(pass_name, multigraph) {}
  ~SplitNodeLite() override = default;
  virtual const BaseRef DefinePattern() const = 0;
  virtual const bool CanSplit(const AnfNodePtr &node) const = 0;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &node, const EquivPtr &) const override;
};

class SplitReshapeAndCacheLite : public SplitNodeLite {
 public:
  /**
   * @brief This pass will split umonad from ReshapeAndCache inputs and add Depend node.
   */
  explicit SplitReshapeAndCacheLite(bool multigraph = true) : SplitNodeLite("split_reshape_and_cache", multigraph) {}
  ~SplitReshapeAndCacheLite() override = default;
  const BaseRef DefinePattern() const override;
  const bool CanSplit(const AnfNodePtr &node) const override;
};

}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_SPLIT_UMONAD_H
