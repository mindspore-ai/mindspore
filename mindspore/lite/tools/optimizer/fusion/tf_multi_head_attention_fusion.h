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
#include "backend/optimizer/common/optimizer.h"
#include "utils/utils.h"
#include "include/errorcode.h"
#include "tools/converter/ops/attention.h"

namespace mindspore {
namespace opt {
class TfMultiHeadAttentionFusion : public PatternProcessPass {
 public:
  explicit TfMultiHeadAttentionFusion(const std::string &name = "tflite_multi_head_attention_fusion",
                                      bool multigraph = true);
  ~TfMultiHeadAttentionFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 protected:
  const VectorRef DefineDensePattern(const BaseRef &input, const BaseRef &weight, const BaseRef &bias) const;
  virtual const VectorRef DefineProcessInputPattern(const BaseRef &input, const BaseRef &weight, const BaseRef &bias,
                                                    const BaseRef &reshape_shape, bool transpose = false) const;
  virtual const VectorRef DefineProcessOutputPattern(const BaseRef &input, const BaseRef &weight,
                                                     const BaseRef &bias) const;

  CNodePtr CreateMultiHeadAttentionNode(const FuncGraphPtr &func_graph, const EquivPtr &equiv,
                                        const std::string &base_name, int var_offset) const;
  virtual std::shared_ptr<ops::Attention> BuildAttentionPrim(const EquivPtr &equiv) const;

  VarPtr input_q_;
  VarPtr input_k_;
  VarPtr input_v_;

  VarPtr weight_q_;
  VarPtr weight_k_;
  VarPtr weight_v_;
  VarPtr weight_o_;
  VarPtr bias_q_;
  VarPtr bias_k_;
  VarPtr bias_v_;
  VarPtr bias_o_;

  VarPtr reshape_k_;
  VarPtr reshape_v_;
};

}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_TF_MULTI_HEAD_ATTENTION_FUSION_H_
