/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "tools/converter/parser/tf/tf_fake_quant_adjust.h"
#include <utility>
#include <memory>
#include <algorithm>
#include "ops/primitive_c.h"
#include "tools/converter/ops/ops_def.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "src/common/utils.h"
#include "tools/converter/quantizer/quantize_util.h"

namespace mindspore {
namespace lite {
bool TFFakeQuantAdjust::SetInputQuantParam(const CNodePtr &cnode, const QuantParamHolderPtr &quant_param_holder,
                                           size_t index) {
  MS_CHECK_TRUE_MSG(quant_param_holder != nullptr, false, "Primitive quant param holder nullptr.");
  auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  auto min_value = primitive->GetAttr("min");
  MS_CHECK_FALSE(min_value == nullptr, false);
  auto max_value = primitive->GetAttr("max");
  MS_CHECK_FALSE(max_value == nullptr, false);
  MS_LOG(INFO) << "min: " << GetValue<float>(min_value) << " max: " << GetValue<float>(max_value);

  std::vector<schema::QuantParamT> quant_params;
  auto quant_param = std::make_unique<QuantParamT>();
  quant_param->min = GetValue<float>(min_value);
  quant_param->max = GetValue<float>(max_value);
  quant_param->scale = (std::max(abs(quant_param->min), abs(quant_param->max))) / quant::kQuantRange;
  quant_param->zeroPoint = 0;
  quant_param->inited = true;
  quant_params.push_back(*std::move(quant_param));
  quant_param_holder->set_input_quant_param(index, quant_params);
  return true;
}

bool TFFakeQuantAdjust::Adjust(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, false);
  for (auto &cnode : func_graph->GetOrderedCnodes()) {
    if (!opt::CheckPrimitiveType(cnode, std::make_unique<Primitive>(lite::kNameFakeQuantWithMinMaxVars))) {
      continue;
    }
    MS_CHECK_GE(cnode->inputs().size(), kInputSize1, false);
    auto manager = func_graph->manager();
    if (manager == nullptr) {
      manager = Manage(func_graph, true);
    }
    MS_CHECK_TRUE_RET(manager != nullptr, true);
    auto node_users = manager->node_users()[cnode];
    for (auto &node_user : node_users) {
      auto next_quant_holder = quant::GetCNodeQuantHolder(node_user.first->cast<CNodePtr>());
      auto ret = SetInputQuantParam(cnode, next_quant_holder, node_user.second - quant::kPrimOffset);
      if (!ret) {
        MS_LOG(ERROR) << "Set quant param failed.";
        return false;
      }
      manager->SetEdge(node_user.first, node_user.second, cnode->inputs()[kIndex1]);
    }
  }
  return true;
}
}  // namespace lite
}  // namespace mindspore
