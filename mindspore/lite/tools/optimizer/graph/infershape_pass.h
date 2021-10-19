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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_INFERSHAPE_PASS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_INFERSHAPE_PASS_H_

#include <map>
#include <memory>
#include <vector>
#include "backend/optimizer/common/pass.h"
#include "tools/optimizer/graph/node_infershape.h"

namespace mindspore {
namespace opt {
class InferShapePass : public Pass {
 public:
  explicit InferShapePass(FmkType fmk_type = converter::kFmkTypeMs, bool train_flag = false)
      : Pass("infer_shape"), fmk_type_(fmk_type), train_flag_(train_flag) {}
  ~InferShapePass() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  bool JudgeAllOpsCanInfer(const FuncGraphPtr &func_graph);
  STATUS InferProcess(const FuncGraphPtr &func_graph);
  STATUS SetSubGraphInput(const CNodePtr &cnode, const FuncGraphPtr &sub_graph);
  STATUS SetSubGraphOutput(const FuncGraphPtr &sub_graph);
  STATUS SetSubGraphAbstract(const CNodePtr &cnode, const FuncGraphPtr &sub_graph);
  int ResetSubGraphInput();

  FmkType fmk_type_{converter::kFmkTypeMs};
  bool train_flag_{false};
  std::shared_ptr<NodeInferShape> node_infer_shape_{nullptr};
  std::map<FuncGraphPtr, std::vector<AnfNodePtr>> sub_inputs_map_{};
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_INFERSHAPE_PASS_H_
