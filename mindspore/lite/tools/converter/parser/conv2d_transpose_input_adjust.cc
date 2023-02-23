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

#include "tools/converter/parser/conv2d_transpose_input_adjust.h"
#include "tools/converter/parser/parser_utils.h"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "ops/fusion/conv2d_transpose_fusion.h"
#include "ops/op_name.h"

namespace mindspore::lite {
namespace {
constexpr size_t kInputSizeFour = 4;
}  // namespace

bool Conv2DTransposeInputAdjust::Run(const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "func graph is nullptr";
    return false;
  }
  auto node_list = TopoSort(func_graph->get_return());
  for (auto node : node_list) {
    if (utils::isa<CNodePtr>(node)) {
      auto cnode = node->cast<CNodePtr>();
      if (cnode == nullptr) {
        MS_LOG(ERROR) << "cnode is nullptr";
        return false;
      }
      if (opt::CheckPrimitiveType(cnode, prim::kPrimConv2dTransposeFusion)) {
        auto inputs = cnode->inputs();
        if (inputs.size() == kInputSizeFour) {
          auto prim = GetValuePtr<Primitive>(inputs[0]);
          if (prim == nullptr) {
            MS_LOG(ERROR) << "prim is nullptr";
            return false;
          }
          auto value = prim->GetAttr("has_bias");
          if (value != nullptr && GetValue<bool>(value)) {
            MS_LOG(INFO) << "The value of has_bias is true";
            continue;
          } else {
            MS_LOG(INFO) << "node do not have has_bias attr";
            (void)prim->AddAttr(ops::kOriginalOpName, MakeValue(ops::kNameConv2dTransposeFusion));
            inputs.pop_back();
            cnode->set_inputs(inputs);
          }
        }
      }
    }
  }
  return true;
}
}  // namespace mindspore::lite
