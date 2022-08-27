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
#include <set>
#include "ops/primitive_c.h"
#include "tools/converter/ops/ops_def.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "src/common/utils.h"
#include "tools/converter/quantizer/quantize_util.h"

namespace mindspore {
namespace lite {
bool TFFakeQuantAdjust::SetQuantParam(const CNodePtr &cnode, const CNodePtr &post_cnode, size_t index) {
  MS_CHECK_TRUE_RET(post_cnode != nullptr, false);
  auto quant_param_holder = quant::GetCNodeQuantHolder(post_cnode);
  MS_CHECK_TRUE_MSG(quant_param_holder != nullptr, false, "Primitive quant param holder nullptr.");

  auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  auto min_value = primitive->GetAttr("min");
  MS_CHECK_FALSE(min_value == nullptr, false);
  auto max_value = primitive->GetAttr("max");
  MS_CHECK_FALSE(max_value == nullptr, false);
  auto num_bits_value = primitive->GetAttr("num_bits");
  MS_CHECK_FALSE(num_bits_value == nullptr, false);
  auto narrow_range_value = primitive->GetAttr("narrow_range");
  MS_CHECK_FALSE(narrow_range_value == nullptr, false);

  std::vector<schema::QuantParamT> quant_params;
  schema::QuantParamT quant_param;
  auto real_min = GetValue<float>(min_value);
  auto real_max = GetValue<float>(max_value);
  auto bit_num = GetValue<int>(num_bits_value);
  bool narrow_range = GetValue<bool>(narrow_range_value);
  MS_LOG(DEBUG) << "min: " << real_min << " max: " << real_max << " bit_num: " << bit_num << " narrow_range"
                << narrow_range;
  auto ret = CalQuantizationParams(&quant_param, real_min, real_max, bit_num, narrow_range);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Failed to calculate quant params, post node name: " << post_cnode->fullname_with_scope();
    return false;
  }

  quant_params.push_back(quant_param);
  quant_param_holder->set_input_quant_param(index, quant_params);
  return true;
}

bool TFFakeQuantAdjust::RemoveFakeQuant(const FuncGraphPtr &func_graph) {
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
      auto status = SetQuantParam(cnode, node_user.first->cast<CNodePtr>(), node_user.second - quant::kPrimOffset);
      if (!status) {
        MS_LOG(ERROR) << "Set quant param failed.";
        return false;
      }
      manager->SetEdge(node_user.first, node_user.second, cnode->inputs()[kIndex1]);
    }
  }
  return true;
}

bool TFFakeQuantAdjust::Adjust(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, false);
  if (!RemoveFakeQuant(func_graph)) {
    MS_LOG(ERROR) << "RemoveFakeQuant failed.";
    return false;
  }
  return true;
}
}  // namespace lite
}  // namespace mindspore
