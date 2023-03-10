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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_IR_FUSION_TENSOR_SCATTER_FISSION_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_IR_FUSION_TENSOR_SCATTER_FISSION_H_
#include <string>

#include "include/backend/optimizer/pass.h"
#include "ir/func_graph.h"
#include "ir/anf.h"
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class TensorScatterFission : public PatternProcessPass {
 public:
  explicit TensorScatterFission(bool multigraph = true, const string &name = "tensor_scatter_fission")
      : PatternProcessPass(name, multigraph) {}
  ~TensorScatterFission() override = default;
  const AnfNodePtr Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const override;

 protected:
  virtual ValueNodePtr GetScatterNdPrimNode() const = 0;
};

class TensorScatterUpdateFission : public TensorScatterFission {
 public:
  explicit TensorScatterUpdateFission(bool multigraph = true, const string &name = "tensor_scatter_update_fission")
      : TensorScatterFission(multigraph, name) {}
  ~TensorScatterUpdateFission() override = default;
  const BaseRef DefinePattern() const override;

 protected:
  ValueNodePtr GetScatterNdPrimNode() const override;
};

class TensorScatterAddFission : public TensorScatterFission {
 public:
  explicit TensorScatterAddFission(bool multigraph = true, const string &name = "tensor_scatter_add_fission")
      : TensorScatterFission(multigraph, name) {}
  ~TensorScatterAddFission() override = default;
  const BaseRef DefinePattern() const override;

 protected:
  ValueNodePtr GetScatterNdPrimNode() const override;
};

class TensorScatterSubFission : public TensorScatterFission {
 public:
  explicit TensorScatterSubFission(bool multigraph = true, const string &name = "tensor_scatter_sub_fission")
      : TensorScatterFission(multigraph, name) {}
  ~TensorScatterSubFission() override = default;
  const BaseRef DefinePattern() const override;

 protected:
  ValueNodePtr GetScatterNdPrimNode() const override;
};

class TensorScatterMaxFission : public TensorScatterFission {
 public:
  explicit TensorScatterMaxFission(bool multigraph = true, const string &name = "tensor_scatter_max_fission")
      : TensorScatterFission(multigraph, name) {}
  ~TensorScatterMaxFission() override = default;
  const BaseRef DefinePattern() const override;

 protected:
  ValueNodePtr GetScatterNdPrimNode() const override;
};

class TensorScatterMinFission : public TensorScatterFission {
 public:
  explicit TensorScatterMinFission(bool multigraph = true, const string &name = "tensor_scatter_min_fission")
      : TensorScatterFission(multigraph, name) {}
  ~TensorScatterMinFission() override = default;
  const BaseRef DefinePattern() const override;

 protected:
  ValueNodePtr GetScatterNdPrimNode() const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_IR_FUSION_TENSOR_SCATTER_FISSION_H_
