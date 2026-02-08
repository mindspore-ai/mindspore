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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_ASCEND_IR_FUSION_INFER_INFERENCE_QBMM_ELEMWISE_FUSION_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_ASCEND_IR_FUSION_INFER_INFERENCE_QBMM_ELEMWISE_FUSION_H_

#include <memory>
#include <vector>
#include <string>
#include <map>
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/inference_qbmm_fusion_base.h"
#include "include/backend/common/pass_manager/optimizer.h"
#include "primitive/math_ops.h"

namespace mindspore {
namespace opt {
class InferenceQbmmElemwiseFusion : public QbmmFusionBase {
 public:
  explicit InferenceQbmmElemwiseFusion(bool multigraph = true, const string &pass_name = "quant_matmul_elemwise_fusion")
      : QbmmFusionBase(pass_name, multigraph) {}
  ~InferenceQbmmElemwiseFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &equiv) const override;

 private:
  void SetInternalNodes(const EquivPtr &equiv) const;
  bool CheckIOValid() const;
  CNodePtr CreateQbmmElemNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &equiv) const;
  std::vector<std::string> MustExistPrimitiveName() const override;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_ASCEND_IR_FUSION_INFER_INFERENCE_QBMM_ELEMWISE_FUSION_H_
