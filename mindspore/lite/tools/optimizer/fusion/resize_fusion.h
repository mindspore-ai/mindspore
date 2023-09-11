/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_RESIZE_FUSION_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_RESIZE_FUSION_H_

#include "tools/optimizer/common/pattern_process_pass_extends.h"
#include "tools/converter/converter_context.h"

namespace mindspore {
namespace opt {
class ResizeFusion : public LitePatternProcessPass {
 public:
  explicit ResizeFusion(bool multigraph = true) : LitePatternProcessPass("ResizeFusion", multigraph) {}
  virtual ~ResizeFusion() = default;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  virtual int DoFuison(const FuncGraphPtr &func_graph, const AnfNodePtr &node) const = 0;
};

class ResizeFusion1 : public ResizeFusion {
 public:
  explicit ResizeFusion1(bool multigraph = true) : ResizeFusion(multigraph) {}
  ~ResizeFusion1() override = default;

  const BaseRef DefinePattern() const override;

 private:
  int DoFuison(const FuncGraphPtr &func_graph, const AnfNodePtr &node) const override;
  mutable VarPtr input_ = nullptr;
};

class ResizeFusion2 : public ResizeFusion {
 public:
  explicit ResizeFusion2(bool multigraph = true) : ResizeFusion(multigraph) {}
  ~ResizeFusion2() override = default;

  const BaseRef DefinePattern() const override;

 private:
  int DoFuison(const FuncGraphPtr &func_graph, const AnfNodePtr &node) const override;
  mutable VarPtr input_ = nullptr;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_RESIZE_FUSION_H_
