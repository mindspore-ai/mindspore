/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_CONV2D_UNIFY_MINDIR_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_CONV2D_UNIFY_MINDIR_H_

#include <memory>
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class Conv2DUnifyMindIR : public PatternProcessPass {
 public:
  explicit Conv2DUnifyMindIR(bool multigraph = true) : PatternProcessPass("conv2d_unify_mindir", multigraph) {}
  ~Conv2DUnifyMindIR() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  CNodePtr CreateDepthwiseConv2D(const FuncGraphPtr &graph, const CNodePtr &conv2d, const CNodePtr &transpose) const;
};

class Conv2DBackpropInputUnifyMindIR : public PatternProcessPass {
 public:
  explicit Conv2DBackpropInputUnifyMindIR(bool multigraph = true)
      : PatternProcessPass("conv2d_backprop_input_unify_mindir", multigraph) {}
  ~Conv2DBackpropInputUnifyMindIR() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  CNodePtr CreateDepthwiseConv2DBackpropInput(const FuncGraphPtr &graph, const CNodePtr &conv2d_backin,
                                              const CNodePtr &transpose) const;
};

class Conv2DBackpropFilterUnifyMindIR : public PatternProcessPass {
 public:
  explicit Conv2DBackpropFilterUnifyMindIR(bool multigraph = true)
      : PatternProcessPass("conv2d_backprop_filter_unify_mindir", multigraph) {}
  ~Conv2DBackpropFilterUnifyMindIR() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  CNodePtr CreateDepthwiseConv2DBackpropFilter(const FuncGraphPtr &graph, const CNodePtr &conv2d_backfil) const;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_CONV2D_UNIFY_MINDIR_H_
