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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_CONST_FOLD_FOLD_ALONG_INFERSHAPE_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_CONST_FOLD_FOLD_ALONG_INFERSHAPE_H_

#include <memory>
#include "tools/optimizer/graph/infershape_pass.h"
#include "tools/optimizer/const_fold/fold_utils.h"

namespace mindspore {
namespace opt {
class ConstFoldAlongInferShape : public InferShapePass {
 public:
  explicit ConstFoldAlongInferShape(FmkType fmk_type = converter::kFmkTypeMs, bool train_flag = false)
      : InferShapePass(fmk_type, train_flag, "ConstFoldAlongInferShape") {}
  ~ConstFoldAlongInferShape() override = default;

 private:
  STATUS PostProcess(const FuncGraphPtr &func_graph, const CNodePtr &cnode) override;
  bool CheckCanFold(const FuncGraphPtr &func_graph, const CNodePtr &cnode);
  std::shared_ptr<ConstFoldProcessor> const_fold_processor_{nullptr};
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_CONST_FOLD_FOLD_ALONG_INFERSHAPE_H_
