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
#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_TF_MULTI_HEAD_ATTENTION_FUSION_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_TF_MULTI_HEAD_ATTENTION_FUSION_H_

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include "tools/optimizer/common/multiple_pattern_process_pass.h"
#include "utils/utils.h"
#include "include/errorcode.h"
#include "ops/attention.h"

namespace mindspore {
namespace opt {
class MultiHeadAttentionFusion : public MultiplePatternProcessPass {
 public:
  explicit MultiHeadAttentionFusion(const std::string &name = "MultiHeadAttentionFusion", bool multigraph = true);

  ~MultiHeadAttentionFusion() override = default;

 protected:
  std::unordered_map<std::string, VectorRef> DefinePatterns() const override;

  AnfNodePtr Process(const std::string &pattern_name, const FuncGraphPtr &, const AnfNodePtr &,
                     const EquivPtr &) const override;
  // define patterns
  VectorRef DefineMPWithMaskPattern() const;
  VectorRef DefineMPWithoutMaskPattern() const;
  // create multi-head-attention without mask
  virtual std::shared_ptr<ops::Attention> BuildAttentionPrim(const EquivPtr &equiv) const;
  CNodePtr CreateMultiHeadAttentionNode(const FuncGraphPtr &func_graph, const EquivPtr &equiv,
                                        const std::string &base_name, int var_offset) const;
  // create masked-multi-head-attention
  CNodePtr CreateMaskedMultiHeadAttentionNode(const FuncGraphPtr &func_graph, const EquivPtr &equiv,
                                              const std::string &base_name, int var_offset) const;

 protected:
  const std::string kMPAWithoutMaskPatternName = "MPAWithoutMaskPattern";
  const std::string kMPAWithMaskPatternName = "MPAWithMaskPattern";

  VarPtr input_q_{nullptr};
  VarPtr input_k_{nullptr};
  VarPtr input_v_{nullptr};

  VarPtr weight_q_{nullptr};
  VarPtr weight_k_{nullptr};
  VarPtr weight_v_{nullptr};
  VarPtr weight_o_{nullptr};
  VarPtr bias_q_{nullptr};
  VarPtr bias_k_{nullptr};
  VarPtr bias_v_{nullptr};
  VarPtr bias_o_{nullptr};

  VarPtr mask_{nullptr};

  VarPtr reshape_k_{nullptr};
  VarPtr reshape_v_{nullptr};
};

}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_TF_MULTI_HEAD_ATTENTION_FUSION_H_
