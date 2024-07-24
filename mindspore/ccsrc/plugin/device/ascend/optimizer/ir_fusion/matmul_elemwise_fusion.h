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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_MATMUL_ELEMWISE_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_MATMUL_ELEMWISE_FUSION_H_

#include <memory>
#include <string>
#include "include/backend/optimizer/optimizer.h"
#include "mindspore/core/ops/math_ops.h"

namespace mindspore {
namespace opt {
constexpr auto kUnaryInputNum = 1;
constexpr auto kBinaryInputNum = 2;
class MatmulElemFusionBase : public PatternProcessPass {
 public:
  explicit MatmulElemFusionBase(bool multigraph = true, const string &pass_name = "", int64_t elewise_input_num = 0)
      : PatternProcessPass(pass_name, multigraph) {
    elewise_input_num_ = elewise_input_num;
  }
  ~MatmulElemFusionBase() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &equiv) const override;

 protected:
  virtual const VectorRef DefineMatmulFusionPattern(const VectorRef &predecessor) const = 0;
  virtual const std::string GetElemwiseType() const = 0;
  int64_t elewise_input_num_{0};
};

class MatmulElemBiasaddFusion : public MatmulElemFusionBase {
 public:
  explicit MatmulElemBiasaddFusion(bool multigraph = true)
      : MatmulElemFusionBase(multigraph, "matmul_elem_biasadd_fusion", kBinaryInputNum) {}
  ~MatmulElemBiasaddFusion() override = default;

 protected:
  const VectorRef DefineMatmulFusionPattern(const VectorRef &predecessor) const override;
  const std::string GetElemwiseType() const override { return "bias_add"; };
};

class MatmulElemAddFusion : public MatmulElemFusionBase {
 public:
  explicit MatmulElemAddFusion(bool multigraph = true)
      : MatmulElemFusionBase(multigraph, "matmul_elem_add_fusion", kBinaryInputNum) {}
  ~MatmulElemAddFusion() override = default;

 protected:
  const VectorRef DefineMatmulFusionPattern(const VectorRef &predecessor) const override;
  const std::string GetElemwiseType() const override { return "bias_add"; };
};

class MatmulElemReluFusion : public MatmulElemFusionBase {
 public:
  explicit MatmulElemReluFusion(bool multigraph = true)
      : MatmulElemFusionBase(multigraph, "matmul_elem_relu_fusion", kUnaryInputNum) {}
  ~MatmulElemReluFusion() override = default;

 protected:
  const VectorRef DefineMatmulFusionPattern(const VectorRef &predecessor) const override;
  const std::string GetElemwiseType() const override { return "relu"; }
};

class MatmulElemGeluFusion : public MatmulElemFusionBase {
 public:
  explicit MatmulElemGeluFusion(bool multigraph = true)
      : MatmulElemFusionBase(multigraph, "matmul_elem_gelu_fusion", kUnaryInputNum) {}
  ~MatmulElemGeluFusion() override = default;

 protected:
  const VectorRef DefineMatmulFusionPattern(const VectorRef &predecessor) const override;
  const std::string GetElemwiseType() const override { return "gelu"; }
};

}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_MATMUL_ELEMWISE_FUSION_H_
