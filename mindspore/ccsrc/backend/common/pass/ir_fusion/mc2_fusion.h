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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_MC2_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_MC2_FUSION_H_

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include "include/backend/common/pass_manager/optimizer.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace opt {
enum MC2FusionLevel { kMC2NotFusion = 0, kMC2FusionForward = 1, kMC2FusionBackward = 2, kMC2FusionFull = 3 };
class BACKEND_COMMON_EXPORT MC2FusionBase : public PatternProcessPass {
 public:
  explicit MC2FusionBase(const std::string &name = "", bool multigraph = true) : PatternProcessPass(name, multigraph) {}
  ~MC2FusionBase() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                           const EquivPtr &equiv) const override;

 protected:
  virtual const VectorRef DefineFusionPattern() const = 0;
  virtual CNodePtr CreateFusionCNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                     const EquivPtr &equiv) const = 0;
};

class BACKEND_COMMON_EXPORT MatmulReduceScatterFusion : public MC2FusionBase {
 public:
  explicit MatmulReduceScatterFusion(const std::string &name = "matmul_reduce_scatter_fusion", bool multigraph = true)
      : MC2FusionBase(name, multigraph) {}
  ~MatmulReduceScatterFusion() override = default;

 private:
  const VectorRef DefineFusionPattern() const override;
  CNodePtr CreateFusionCNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                             const EquivPtr &equiv) const override;
  std::vector<std::string> MustExistPrimitiveName() const override;
};

class BACKEND_COMMON_EXPORT AllGatherMatmulFusion : public MC2FusionBase {
 public:
  explicit AllGatherMatmulFusion(const std::string &name = "allgather_matmul_fusion", bool multigraph = true)
      : MC2FusionBase(name, multigraph) {}
  ~AllGatherMatmulFusion() override = default;

 private:
  const VectorRef DefineFusionPattern() const override;
  CNodePtr CreateFusionCNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                             const EquivPtr &equiv) const override;
  std::vector<std::string> MustExistPrimitiveName() const override;
};

class BACKEND_COMMON_EXPORT QuantBatchMatmulAllReduceFusion : public MC2FusionBase {
 public:
  explicit QuantBatchMatmulAllReduceFusion(const std::string &name = "quantbatchmatmul_reduce_fusion",
                                           bool multigraph = true)
      : MC2FusionBase(name, multigraph) {}
  ~QuantBatchMatmulAllReduceFusion() override = default;

 private:
  const VectorRef DefineFusionPattern() const override;
  CNodePtr CreateFusionCNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                             const EquivPtr &equiv) const override;
  std::vector<std::string> MustExistPrimitiveName() const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_MC2_FUSION_H_
