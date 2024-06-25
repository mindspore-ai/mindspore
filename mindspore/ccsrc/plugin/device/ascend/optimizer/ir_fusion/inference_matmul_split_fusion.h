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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_INFERENCE_MATMUL_SPLIT_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_INFERENCE_MATMUL_SPLIT_FUSION_H_

#include <string>
#include <memory>
#include <map>
#include <set>

#include "plugin/device/ascend/optimizer/ir_fusion/inference_weight_preprocess_utils.h"
#include "include/backend/optimizer/pass.h"
#include "ir/func_graph.h"
#include "ir/anf.h"
#include "include/backend/optimizer/helper.h"
#include "include/backend/optimizer/optimizer.h"
#include "mindspore/core/ops/nn_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/framework_ops.h"

namespace mindspore {
namespace opt {
constexpr auto kMatmulQkvSplitSizeLen = 3;
constexpr auto kMatmulFfnSplitSizeLen = 2;

class InferenceMatmulSplitFusion : public Pass {
 public:
  InferenceMatmulSplitFusion() : Pass("inference_matmul_split_fusion") {}
  ~InferenceMatmulSplitFusion() override = default;
  bool Run(const FuncGraphPtr &graph) override;

 private:
  std::string GetFusionPatternName(const CNodePtr &cnode) const;
  std::string GetSplitFusionPatternName(const CNodePtr &cnode) const;
  bool CheckMatMulDataFormat(const CNodePtr &matmul_cnode) const;
  size_t GetSplitSizeLen(const CNodePtr &split_cnode) const;
  PrimitivePtr CreateMatmulSplitPrim(const CNodePtr &split_cnode, size_t split_size_len, const std::string &) const;
  CNodePtr CreateGetItemNode(const FuncGraphPtr &func_graph, const CNodePtr &split_cnode,
                             const CNodePtr &matmul_split_cnode, const CNodePtr &silu_cnode,
                             const size_t output_index) const;
  CNodePtr CreateMatmulSplitNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const std::string &) const;
  CNodePtr CreateMatmulBiasAddSplitNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                        const std::string &) const;
  CNodePtr CreateQuantbatchmatmulSplitNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                           const std::string &) const;
  CNodePtr CreateMatmulSplitSiluNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const std::string &) const;
  CNodePtr CreateMatmulBiasAddSplitSiluNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                            const std::string &) const;
  CNodePtr CreateQuantbatchmatmulSplitSiluNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                               const std::string &) const;
  bool enable_fusion_silu = false;
  mutable std::set<CNodePtr> visited_cnodes;

 protected:
  AnfNodePtr Process(const std::string &pattern_name, const FuncGraphPtr &func_graph, const AnfNodePtr &node) const;
  const std::string kPrimNameMatmulSplitOut2 = "MatmulSplitOut2";
  const std::string kPrimNameMatmulSplitOut3 = "MatmulSplitOut3";
  const std::string kPrimNameMatmulSplitSiluOut2 = "MatmulSplitSiluOut2";
  const std::string kPrimNameMatmulBiasSplitOut2 = "MatmulBiasSplitOut2";
  const std::string kPrimNameMatmulBiasSplitOut3 = "MatmulBiasSplitOut3";
  const std::string kPrimNameMatmulBiasSplitSiluOut2 = "MatmulBiasSplitSiluOut2";
  const std::string kPrimNameQuantbatchmatmulSplitOut2 = "QuantbatchmatmulSplitOut2";
  const std::string kPrimNameQuantbatchmatmulSplitOut3 = "QuantbatchmatmulSplitOut3";
  const std::string kPrimNameQuantbatchmatmulSplitSiluOut2 = "QuantbatchmatmulSplitSiluOut2";

  const std::string kPatternNameMatMulSplit = "MatmulSplit";
  const std::string kPatternNameMatMulSplitSilu = "MatmulSplitSilu";
  const std::string kPatternNameMatMulBiasAddSplit = "MatmulBiasAddSplit";
  const std::string kPatternNameMatMulBiasAddSplitSilu = "MatmulBiasAddSplitSilu";
  const std::string kPatternNameQuantbatchmatmulSplit = "QuantbatchmatmulSplit";
  const std::string kPatternNameQuantbatchmatmulSplitSilu = "QuantbatchmatmulSplitSilu";

  std::map<size_t, std::map<std::string, std::string>> PatternPrimMap = {
    {
      kMatmulQkvSplitSizeLen,
      {{kPatternNameMatMulSplit, kPrimNameMatmulSplitOut3},
       {kPatternNameMatMulBiasAddSplit, kPrimNameMatmulBiasSplitOut3},
       {kPatternNameQuantbatchmatmulSplit, kPrimNameQuantbatchmatmulSplitOut3}},
    },

    {kMatmulFfnSplitSizeLen,
     {{kPatternNameMatMulSplit, kPrimNameMatmulSplitOut2},
      {kPatternNameMatMulSplitSilu, kPrimNameMatmulSplitSiluOut2},
      {kPatternNameMatMulBiasAddSplit, kPrimNameMatmulBiasSplitOut2},
      {kPatternNameMatMulBiasAddSplitSilu, kPrimNameMatmulBiasSplitSiluOut2},
      {kPatternNameQuantbatchmatmulSplit, kPrimNameQuantbatchmatmulSplitOut2},
      {kPatternNameQuantbatchmatmulSplitSilu, kPrimNameQuantbatchmatmulSplitSiluOut2}}}};
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_INFERENCE_MATMUL_SPLIT_FUSION_H_
