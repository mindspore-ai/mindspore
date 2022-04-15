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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_IR_FUSION_TENSOR_SCATTER_FUSION_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_IR_FUSION_TENSOR_SCATTER_FUSION_H_
#include <string>

#include "backend/common/optimizer/pass.h"
#include "ir/func_graph.h"
#include "ir/anf.h"
#include "backend/common/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class TensorScatterFusion : public PatternProcessPass {
 public:
  explicit TensorScatterFusion(bool multigraph = true, const string &name = "tensor_scatter_fusion")
      : PatternProcessPass(name, multigraph) {}
  ~TensorScatterFusion() override = default;
  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const override;

 protected:
  virtual ValueNodePtr GetScatterNdPrimNode() const = 0;
};

class TensorScatterAddFusion : public TensorScatterFusion {
 public:
  explicit TensorScatterAddFusion(bool multigraph = true, const string &name = "tensor_scatter_add_fusion")
      : TensorScatterFusion(multigraph, name) {}
  ~TensorScatterAddFusion() override = default;
  const BaseRef DefinePattern() const override;

 protected:
  ValueNodePtr GetScatterNdPrimNode() const override;
};

class TensorScatterSubFusion : public TensorScatterFusion {
 public:
  explicit TensorScatterSubFusion(bool multigraph = true, const string &name = "tensor_scatter_sub_fusion")
      : TensorScatterFusion(multigraph, name) {}
  ~TensorScatterSubFusion() override = default;
  const BaseRef DefinePattern() const override;

 protected:
  ValueNodePtr GetScatterNdPrimNode() const override;
};

class TensorScatterMaxFusion : public TensorScatterFusion {
 public:
  explicit TensorScatterMaxFusion(bool multigraph = true, const string &name = "tensor_scatter_max_fusion")
      : TensorScatterFusion(multigraph, name) {}
  ~TensorScatterMaxFusion() override = default;
  const BaseRef DefinePattern() const override;

 protected:
  ValueNodePtr GetScatterNdPrimNode() const override;
};

class TensorScatterMinFusion : public TensorScatterFusion {
 public:
  explicit TensorScatterMinFusion(bool multigraph = true, const string &name = "tensor_scatter_min_fusion")
      : TensorScatterFusion(multigraph, name) {}
  ~TensorScatterMinFusion() override = default;
  const BaseRef DefinePattern() const override;

 protected:
  ValueNodePtr GetScatterNdPrimNode() const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_IR_FUSION_TENSOR_SCATTER_FUSION_H_
