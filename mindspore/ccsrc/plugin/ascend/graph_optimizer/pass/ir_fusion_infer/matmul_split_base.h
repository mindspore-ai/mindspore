/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_INFER_MATMUL_SPLIT_BASE_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_INFER_MATMUL_SPLIT_BASE_H_

#include <memory>
#include <vector>
#include <string>
#include <set>
#include <tuple>
#include "include/backend/common/pass_manager/optimizer.h"

namespace mindspore {
namespace opt {
class MatmulSplitBase : public PatternProcessPass {
 public:
  explicit MatmulSplitBase(const std::string &pass_name, bool multigraph = true)
      : PatternProcessPass(pass_name, multigraph) {}

 protected:
  BaseRef GetMatmulPattern() const;
  BaseRef GetSplitWithSizePattern(const BaseRef &pre_pattern_ref) const;
  bool IsEnableMatmulSplit() const;
  bool CheckMatmulSplit(const AnfNodePtr &input_x, const AnfNodePtr &input_w, const ValueNodePtr &input_trans_a,
                        const ValueNodePtr &input_trans_b, const ValueNodePtr &split_size_node) const;
  bool CheckMatmulDataFormat(const ValueNodePtr &input_trans_a, const ValueNodePtr &input_trans_b) const;
  bool CheckSplitSize(const AnfNodePtr &input_w, const ValueNodePtr &split_size_node) const;
  std::vector<int64_t> GetSplitSizeShape(const ValueNodePtr &split_size_node) const;
  PrimitivePtr GetMatmulSplitPrimitive(const ValueNodePtr &split_size_node) const;
  std::string GetMatmulSplitPrimName(const ValueNodePtr &split_size_node) const;
  std::tuple<CNodePtr, ValueNodePtr> GetSplitSizeNode(const AnfNodePtr &node) const;
  std::tuple<CNodePtr, AnfNodePtr, AnfNodePtr, ValueNodePtr, ValueNodePtr> GetMatmulNode(
    const CNodePtr &pre_cnode) const;
  ValueNodePtr GetReshapeTupleNode(const FuncGraphPtr &graph) const;
  CNodePtr GetMatmulSplitCNode(const PrimitivePtr &matmul_split_prim, const AnfNodePtrList &matmul_split_inputs,
                               const FuncGraphPtr &graph, const CNodePtr &matmul_cnode,
                               const CNodePtr &split_cnode) const;

  virtual std::string GetFfnSplitPriName() const = 0;
  virtual std::string GetQkvSplitPriName() const = 0;
  virtual void SetMatmulSplitPrimitiveAttr(const PrimitivePtr &matmul_split_prim,
                                           const ValueNodePtr &split_size_node) const = 0;

  static constexpr auto kInferenceMatmulSplitName = "InferenceMatmulSplit";
  const std::set<TypeId> kSupportDataType = {kNumberTypeFloat16, kNumberTypeBFloat16};
  static constexpr auto kNLength = "n_lens";
  static constexpr auto kTuplePlaceHolderNum = 0;
  static constexpr auto kMatmulFfnSplitSizeLen = 2;
  static constexpr auto kMatmulQkvSplitSizeLen = 3;
  static constexpr auto kValidShape = 16;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_INFER_MATMUL_SPLIT_BASE_H_
