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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_MATMUL_ALLREDUCE_ADD_RMSNORM_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_MATMUL_ALLREDUCE_ADD_RMSNORM_FUSION_H_
#include <string>
#include <memory>
#include <vector>
#include "include/backend/common/pass_manager/optimizer.h"

namespace mindspore {
namespace opt {
class MatMulAllReduceAddRmsNormBaseFusion : public PatternProcessPass {
 public:
  explicit MatMulAllReduceAddRmsNormBaseFusion(const string &name, bool multigraph = true)
      : PatternProcessPass(name, multigraph) {
    x1_ = std::make_shared<Var>();
    x2_ = std::make_shared<Var>();
    residual_ = std::make_shared<Var>();
    gamma_ = std::make_shared<Var>();
    eps_ = std::make_shared<Var>();
  }
  ~MatMulAllReduceAddRmsNormBaseFusion() override = default;

 protected:
  VarPtr x1_;
  VarPtr x2_;
  VarPtr residual_;
  VarPtr gamma_;
  VarPtr eps_;
  const std::string kAttrNameGroup = "group";
  const std::string kAttrNameOp = "op";
};

class MatMulAllReduceAddRmsNormFusion : public MatMulAllReduceAddRmsNormBaseFusion {
 public:
  MatMulAllReduceAddRmsNormFusion() : MatMulAllReduceAddRmsNormBaseFusion("MatMulAllReduceAddRmsNorm", true) {}
  ~MatMulAllReduceAddRmsNormFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &equiv) const override;
  std::vector<std::string> MustExistPrimitiveName() const override;

 private:
  virtual CNodePtr CreateMatMulAllReduceAddRmsNormNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                       const EquivPtr &equiv, const TypeId &add_result_type) const;
  AnfNodePtr NewTransposeNode(const FuncGraphPtr &func_graph, const AnfNodePtr &x2, const AnfNodePtr &node,
                              const TypeId &add_result_type) const;
  bool IsSupport(const AnfNodePtr &node, const FuncGraphPtr &graph) const;

  // currently, reduction only support "sum"
  const std::vector<std::string> support_reduce_op_list_ = {"sum"};
};

}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_MATMUL_ALLREDUCE_ADD_RMSNORM_FUSION_H_
