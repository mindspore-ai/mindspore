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
#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_MULTI_HEAD_ATTENTION_FUSION_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_MULTI_HEAD_ATTENTION_FUSION_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <tuple>
#include "tools/optimizer/common/multiple_pattern_process_pass.h"
#include "include/common/utils/utils.h"
#include "include/errorcode.h"
#include "ops/attention.h"

namespace mindspore {
namespace opt {
class MultiHeadAttentionFusion : public MultiplePatternProcessPass {
 public:
  explicit MultiHeadAttentionFusion(const std::string &name = "MultiHeadAttentionFusion", bool multigraph = true)
      : MultiplePatternProcessPass(name, multigraph) {}

  ~MultiHeadAttentionFusion() override = default;

  AnfNodePtr Process(const std::string &pattern_name, const FuncGraphPtr &, const AnfNodePtr &,
                     const EquivPtr &) const override;
  std::unordered_map<std::string, VectorRef> DefinePatterns() const override;

 protected:
  virtual bool Init() const;

  // create multi-head-attention without mask
  virtual std::shared_ptr<ops::Attention> BuildAttentionPrim(const EquivPtr &equiv) const;

 private:
  // define patterns
  VectorRef DefineMPWithMaskPattern(bool mask = true) const;
  VectorRef DefineMPWithMaskPatternPA() const;
  VectorRef DefineMPWithMaskPatternT5() const;
  VectorRef DefineMPWithMaskPatternT5New(bool transpose = true, bool no_div_flag = false) const;
  VectorRef DefineMPPatternSwin(bool flag = true) const;
  VectorRef DefineEmbedding(const BaseRef &input, const BaseRef &weight, const BaseRef &bias, const BaseRef &axis,
                            const BaseRef &transpose_var, bool test_div = false, bool transpose = true,
                            bool mul = false) const;
  VectorRef DefineEmbedding(const BaseRef &input, const BaseRef &weight, const BaseRef &axis,
                            const BaseRef &transpose_var, bool test_div, bool transpose) const;

  // create masked-multi-head-attention
  CNodePtr CreateMaskedMultiHeadAttentionNode(const FuncGraphPtr &func_graph, const EquivPtr &equiv,
                                              const std::string &base_name, bool mask = true) const;
  // check pattern
  bool CheckPattern(const EquivPtr &equiv, int *head_num, int *head_size) const;
  CNodePtr CreateOutputGetItem(const FuncGraphPtr &func_graph, const CNodePtr &node, const int item_index) const;
  lite::STATUS SetAbstractTuple(const CNodePtr &cnode, const int output_num) const;
  lite::STATUS AdjustOtherGetItems(const FuncGraphPtr &func_graph, const CNodePtr &attention, int index,
                                   const AnfNodePtr &node) const;
  lite::STATUS RemoveRedundantInput(const FuncGraphPtr &func_graph, const std::vector<AnfNodePtr> &redundant) const;
  std::shared_ptr<ops::Attention> CreatePrim() const;
  CNodePtr MakeGetTuple(const FuncGraphPtr &func_graph, const CNodePtr &new_node, const AnfNodePtr &knode,
                        const AnfNodePtr &vnode) const;
  std::shared_ptr<ops::Attention> CreatePrim(const EquivPtr &equiv, bool cross) const;
  bool IsCross(const EquivPtr &equiv) const;
  std::vector<AnfNodePtr> GetNewNodeInputs(const EquivPtr &equiv, ParameterPtr q_weight_param,
                                           ParameterPtr c_weight_param, AnfNodePtr weight_o, ParameterPtr c_bias_param,
                                           AnfNodePtr bias_o, bool mask, bool cross) const;
  std::tuple<AnfNodePtr, std::shared_ptr<tensor::Tensor>, std::shared_ptr<tensor::Tensor>,
             std::shared_ptr<tensor::Tensor> >
  GetAttentionNodeWeights(const EquivPtr &equiv, std::vector<AnfNodePtr> *redundant) const;
  mutable int match_count_ = 0;

 protected:
  const std::string kMPAWithMaskPatternName = "MPAWithMaskPattern";
  const std::string kMPAWithMaskPatternNamePA = "MPAWithMaskPatternPA";
  const std::string kMPAPatternName = "MPAPattern";
  const std::string kMPAWithMaskPatternNameT5 = "MPAWithMaskPatternT5";
  const std::string kMPAWithMaskPatternNameT5New = "MPAWithMaskPatternT5New";
  const std::string kMPAWithMaskPatternNameT5New2 = "MPAWithMaskPatternT5New2";
  const std::string kMPAWithMaskTransposePatternNameT5New = "MPAWithMaskTransposePatternT5New";
  const std::string kMPAPatternNameSwin1 = "MPAPatternNameSwin1";
  const std::string kMPAPatternNameSwin2 = "MPAPatternNameSwin2";

  mutable VarPtr input_q_{nullptr};
  mutable VarPtr input_k_{nullptr};
  mutable VarPtr input_v_{nullptr};
  mutable VarPtr input1_{nullptr};
  mutable VarPtr input2_{nullptr};
  mutable VarPtr position_bias_{nullptr};

  mutable VarPtr weight_q_{nullptr};
  mutable VarPtr weight_k_{nullptr};
  mutable VarPtr weight_v_{nullptr};
  mutable VarPtr weight_o_{nullptr};
  mutable VarPtr weight_o2_{nullptr};
  mutable VarPtr bias_q_{nullptr};
  mutable VarPtr bias_k_{nullptr};
  mutable VarPtr bias_v_{nullptr};
  mutable VarPtr bias_o_{nullptr};
  mutable VarPtr bias_o2_{nullptr};

  mutable VarPtr mask_{nullptr};

  mutable VarPtr reshape_k_{nullptr};
  mutable VarPtr reshape_v_{nullptr};

  mutable VarPtr reshape_axis_{nullptr};
  mutable VarPtr v_transpose_{nullptr};
  mutable VarPtr k_transpose_{nullptr};

  mutable bool t5_x_{false};
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_MULTI_HEAD_ATTENTION_FUSION_H_
