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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_CONST_FOLD_CONSTANT_FOLDING_FUSION_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_CONST_FOLD_CONSTANT_FOLDING_FUSION_H_

#include "include/backend/optimizer/pass.h"
#include "include/registry/converter_context.h"
#include "tools/optimizer/const_fold/fold_along_infershape.h"
#include "tools/optimizer/const_fold/fold_with_infershape.h"
#include "tools/optimizer/graph/update_conv2d_param_pass.h"

namespace mindspore {
namespace opt {
class ConstFoldPass : public Pass {
 public:
  explicit ConstFoldPass(converter::FmkType fmk_type = converter::kFmkTypeMs, bool train_flag = false)
      : Pass("ConstFoldPass"), fmk_type_(fmk_type), train_flag_(train_flag) {}
  ~ConstFoldPass() override = default;
  bool Run(const FuncGraphPtr &func_graph) override {
    if (func_graph == nullptr) {
      MS_LOG(ERROR) << "func_graph is nullptr, do constant fold failed.";
      return false;
    }
    // current infer-shape cannot support the control-flow model of Mindir.
    if (fmk_type_ == converter::kFmkTypeMs) {
      auto fold_schedule = ConstFoldWithInferShape(fmk_type_, train_flag_);
      if (!fold_schedule.Run(func_graph)) {
        MS_LOG(WARNING) << "Do constant fold failed.";
        return false;
      }
    } else {
      auto fold_schedule = ConstFoldAlongInferShape(fmk_type_, train_flag_);
      if (!fold_schedule.Run(func_graph)) {
        MS_LOG(WARNING) << "Do constant fold failed.";
        return false;
      }
    }

    // the attrs of convolution only can be update after constant fold.
    auto update_attrs = UpdateConv2DParamPass();
    if (!update_attrs.Run(func_graph)) {
      MS_LOG(ERROR) << "update attrs failed.";
      return false;
    }
    return true;
  }

 private:
  FmkType fmk_type_{converter::kFmkTypeMs};
  bool train_flag_{false};
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_CONST_FOLD_CONSTANT_FOLDING_FUSION_H_
