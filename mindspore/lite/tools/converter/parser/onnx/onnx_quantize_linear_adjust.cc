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

#include "tools/converter/parser/onnx/onnx_quantize_linear_adjust.h"
#include <memory>
#include <utility>
#include "ops/primitive_c.h"
#include "ops/reshape.h"
#include "ops/fusion/scale_fusion.h"
#include "ops/fusion/mat_mul_fusion.h"
#include "tools/converter/ops/ops_def.h"
#include "src/common/utils.h"
#include "nnacl/op_base.h"
#include "tools/converter/quantizer/quantize_util.h"

namespace mindspore::lite {
void OnnxQuantizeLinearAdjust::RemoveDequantizeLinear(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  auto inputs = cnode->inputs();
  MS_CHECK_TRUE_RET_VOID(inputs.size() >= kIndex2);
  auto pre_cnode = inputs[1];
  MS_CHECK_PTR_IF_NULL(pre_cnode);
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    manager = Manage(func_graph, true);
  }
  MS_CHECK_PTR_IF_NULL(manager);
  auto node_users = manager->node_users()[cnode];
  if (node_users.empty()) {
    MS_LOG(WARNING) << cnode->fullname_with_scope() << " cnode is isolated.";
    return;
  }
  // remove fake quant node
  for (auto &node_user : node_users) {
    manager->SetEdge(node_user.first, node_user.second, pre_cnode);
  }
}

QuantParamHolderPtr OnnxQuantizeLinearAdjust::GetCNodeQuantHolder(const PrimitivePtr &primitive) {
  MS_CHECK_TRUE_RET(primitive != nullptr, nullptr);
  QuantParamHolderPtr quant_params_holder = nullptr;
  auto quant_params_valueptr = primitive->GetAttr("quant_params");
  if (quant_params_valueptr == nullptr) {
    quant_params_holder = std::make_shared<QuantParamHolder>(0, 0);
    MS_CHECK_TRUE_MSG(quant_params_holder != nullptr, nullptr, "quant_params_holder is nullptr.");
    primitive->AddAttr("quant_params", quant_params_holder);
  } else {
    quant_params_holder = quant_params_valueptr->cast<QuantParamHolderPtr>();
    if (quant_params_holder == nullptr) {
      quant_params_holder = std::make_shared<QuantParamHolder>(0, 0);
      MS_CHECK_TRUE_MSG(quant_params_holder != nullptr, nullptr, "quant_params_holder is nullptr.");
      primitive->AddAttr("quant_params", quant_params_holder);
    }
  }
  return quant_params_holder;
}

bool OnnxQuantizeLinearAdjust::SetInputQuantParam(const CNodePtr &cnode, const QuantParamHolderPtr &quant_param_holder,
                                                  size_t index) {
  MS_CHECK_TRUE_MSG(quant_param_holder != nullptr, false, "Primitive quant params holder nullptr.");
  auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  auto scale_value = primitive->GetAttr("scale");
  MS_CHECK_FALSE_MSG(scale_value == nullptr, false, "Scale value nullptr.");
  auto zero_point_value = primitive->GetAttr("zero_point");
  MS_CHECK_FALSE_MSG(zero_point_value == nullptr, false, "Zero point value nullptr.");
  auto scale = GetValue<float>(scale_value);
  auto zero_point = GetValue<int>(zero_point_value);

  std::vector<schema::QuantParamT> quant_params;
  QuantParamT quant_param;
  quant_param.scale = scale;
  quant_param.zeroPoint = zero_point;
  quant_param.inited = true;
  quant_params.push_back(quant_param);
  quant_param_holder->set_input_quant_param(index, quant_params);
  return true;
}

bool OnnxQuantizeLinearAdjust::Adjust(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, false);

  // remove dequantizer
  for (auto &cnode : func_graph->GetOrderedCnodes()) {
    if (!opt::CheckPrimitiveType(cnode, std::make_shared<Primitive>(lite::kNameDequantizeLinear))) {
      continue;
    }
    RemoveDequantizeLinear(func_graph, cnode);
  }

  for (auto &cnode : func_graph->GetOrderedCnodes()) {
    if (!opt::CheckPrimitiveType(cnode, std::make_shared<Primitive>(lite::kNameQuantizeLinear))) {
      continue;
    }
    auto manager = func_graph->manager();
    if (manager == nullptr) {
      manager = Manage(func_graph, true);
    }
    MS_CHECK_TRUE_RET(manager != nullptr, false);
    auto node_users = manager->node_users()[cnode];
    for (auto &node_user : node_users) {
      auto next_quant_holder = quant::GetCNodeQuantHolder(node_user.first->cast<CNodePtr>());
      auto ret = SetInputQuantParam(cnode, next_quant_holder, (node_user.second - kPrimOffset));
      if (!ret) {
        MS_LOG(ERROR) << "Set quant param failed.";
        return false;
      }
      manager->SetEdge(node_user.first, node_user.second, cnode->inputs()[kIndex1]);
    }
  }

  // check quant param
  for (auto &cnode : func_graph->GetOrderedCnodes()) {
    MS_LOG(DEBUG) << "check cnode name: " << cnode->fullname_with_scope();
    auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
    auto primitive_quant_holder = quant::GetCNodeQuantHolder(primitive);
    auto input_quant_params = primitive_quant_holder->get_input_quant_params();
    for (size_t i = 0; i < input_quant_params.size(); i++) {
      auto quant_params = input_quant_params.at(i);
      if (!quant_params.empty()) {
        auto quant_param = quant_params.front();
        MS_LOG(DEBUG) << "scale: " << quant_param.scale << " zp: " << quant_param.zeroPoint << " index: " << i;
      }
    }
  }
  return true;
}
}  // namespace mindspore::lite
