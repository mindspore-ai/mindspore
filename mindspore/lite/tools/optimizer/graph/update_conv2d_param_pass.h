/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_UPDATE_CONV2D_PARAM_PASS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_UPDATE_CONV2D_PARAM_PASS_H_

#include "include/backend/optimizer/pass.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore::opt {
class UpdateConv2DParamPass : public Pass {
 public:
  UpdateConv2DParamPass() : Pass("UpdateConv2DParamPass") {}
  ~UpdateConv2DParamPass() override = default;
  bool Run(const FuncGraphPtr &graph) override;

 private:
  STATUS UpdateConv2DAttr(const CNodePtr &cnode);
};
}  // namespace mindspore::opt
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_UPDATE_CONV2D_PARAM_PASS_H_
