/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "tools/optimizer/graph/update_conv2d_param_pass.h"
#include <memory>
#include "ops/fusion/conv2d_fusion.h"
#include "mindspore/lite/include/errorcode.h"

namespace mindspore::opt {
namespace {
constexpr int kAnfPopulaterInputNumTwo = 2;
}
bool UpdateConv2DParamPass::Run(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    if (CheckPrimitiveType(node, prim::kPrimConv2DFusion)) {
      auto dwconv2d_cnode = node->cast<CNodePtr>();
      auto conv = GetValueNode<std::shared_ptr<mindspore::ops::Conv2DFusion>>(dwconv2d_cnode->input(0));
      if (conv == nullptr) {
        MS_LOG(ERROR) << "Depthwise conv2D node has no primitiveC.";
        return RET_ERROR;
      }
      if (conv->GetAttr(ops::kIsDepthWise) == nullptr || !GetValue<bool>(conv->GetAttr(ops::kIsDepthWise))) {
        continue;
      }
      int64_t channel_in = conv->GetAttr(ops::kInChannel) != nullptr ? conv->get_in_channel() : -1;
      if (channel_in == -1) {
        auto input_node = node->cast<CNodePtr>()->input(kAnfPopulaterInputNumTwo);
        MS_ASSERT(input_node != nullptr);
        if (input_node->isa<Parameter>()) {
          auto param_node = input_node->cast<ParameterPtr>();
          auto param = param_node->default_param();
          auto weight = std::dynamic_pointer_cast<ParamValueLite>(param);
          conv->set_in_channel(static_cast<int64_t>(weight->tensor_shape().at(0)));
        }
      }
    }
  }
  return true;
}
}  // namespace mindspore::opt
