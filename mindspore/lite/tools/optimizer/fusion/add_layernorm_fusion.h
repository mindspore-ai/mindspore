/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_ADD_LAYERNORM_FUSION_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_ADD_LAYERNORM_FUSION_H_

#include <memory>
#include <string>
#include <unordered_map>
#include "include/common/utils/utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/optimizer/common/multiple_pattern_process_pass.h"

namespace mindspore {
namespace opt {
class AddLayerNormFusion : public MultiplePatternProcessPass {
 public:
  explicit AddLayerNormFusion(const std::string &name = "AddLayerNormFusion", bool multigraph = true)
      : MultiplePatternProcessPass(name, multigraph) {}

  ~AddLayerNormFusion() override = default;

  std::unordered_map<std::string, VectorRef> DefinePatterns() const override;

  AnfNodePtr Process(const std::string &pattern_name, const FuncGraphPtr &, const AnfNodePtr &,
                     const EquivPtr &) const override;

 private:
  bool Init() const;
  bool CheckPattern(const EquivPtr &equiv) const;

  CNodePtr CreateAddLayerNormNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &equiv) const;

  CNodePtr CreateLayerNormV3Node(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &equiv) const;

  const VectorRef DefineAddlayerNormPattern() const;

  const VectorRef DefineLayerNormV3Pattern() const;

  // Add
  mutable VarPtr add_1_a_{nullptr};  // input 1
  mutable VarPtr add_1_b_{nullptr};  // input 2

  // LayerNormV3
  mutable VarPtr reduce_1_x_{nullptr};
  mutable VarPtr reduce_1_axis_{nullptr};  // -1
  mutable VarPtr sub_a_{nullptr};
  mutable VarPtr pow_y_{nullptr};
  mutable VarPtr reduce_2_axis_{nullptr};  // -1
  mutable VarPtr add_2_b_{nullptr};        // -0.00001
  mutable VarPtr mul_b_{nullptr};
  mutable VarPtr add_3_b_{nullptr};
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_ADD_LAYERNORM_FUSION_H_
