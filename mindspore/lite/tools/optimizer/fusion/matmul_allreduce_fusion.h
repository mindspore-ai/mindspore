/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_MATMUL_ALLREDUCE_FUSION_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_MATMUL_ALLREDUCE_FUSION_H_

#include <string>
#include <unordered_map>
#include "tools/optimizer/common/pattern_process_pass_extends.h"
#include "tools/optimizer/common/multiple_pattern_process_pass.h"

namespace mindspore::opt {

class MatMulAllReduceFusion : public MultiplePatternProcessPass {
 public:
  explicit MatMulAllReduceFusion(bool multigraph = true, const std::string &name = "MatMulAllReduceFusion")
      : MultiplePatternProcessPass(name, multigraph) {}
  ~MatMulAllReduceFusion() override = default;
  std::unordered_map<std::string, VectorRef> DefinePatterns() const override;
  AnfNodePtr Process(const std::string &pattern_name, const FuncGraphPtr &, const AnfNodePtr &,
                     const EquivPtr &) const override;

 private:
  VectorRef DefineMatMulAllReducePattern() const;
  VectorRef DefineMatMulBiasAddAllReducePattern() const;
  VectorRef DefineMatMulDequantAllReducePattern() const;
  VectorRef DefineQuantBatchMatmulAllReducePattern() const;
  CNodePtr CreateMatMulAllReduceNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node) const;
  CNodePtr CreateMatMulBiasAddAllReduceNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node) const;
  CNodePtr CreateMatMulDequantAllReduceNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node) const;
  CNodePtr CreateQuantBatchMatmulAllReduceNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node) const;
  PrimitivePtr CreateMatMulAllReducePrim(const PrimitivePtr &allreduce_prim, const PrimitivePtr &matmul_prim) const;

 protected:
  const std::string kPatternNameMatMulAllReduce = "MatMulAllReduce";
  const std::string kPatternNameMatMulBiasAddAllReduce = "MatMulBiasAddAllReduce";
  const std::string kPatternNameMatMulDequantAllReduce = "MatMulDequantAllReduce";
  const std::string kPatternNameQuantBatchMatmulAllReduce = "QuantBatchMatmulAllReduce";
  const std::string kAttrNameGroup = "group";
  const std::string kAttrNameFusion = "fusion";
  const std::string kAttrNameOp = "op";
  const std::string kAttrNameTransposeA = "transpose_a";
  const std::string kAttrNameTransposeB = "transpose_b";
  const std::string kAttrNameNeedFusedXoffsetToBias = "need_fused_x_offset_to_bias";
};
}  // namespace mindspore::opt
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_MATMUL_ALLREDUCE_FUSION_H_
