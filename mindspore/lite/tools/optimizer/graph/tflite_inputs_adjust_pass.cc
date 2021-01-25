/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "tools/optimizer/graph/tflite_inputs_adjust_pass.h"
#include <vector>
#include <memory>
#include "ops/fusion/pad_fusion.h"
#include "ops/op_utils.h"
#include "ops/resize.h"
#include "tools/converter/quant_param_holder.h"
#include "tools/converter/quantizer/quant_cast.h"

namespace mindspore::opt {
namespace {
bool CheckResize(const CNodePtr &c_node) {
  if (!CheckPrimitiveType(c_node, prim::kPrimResize)) {
    return false;
  }
  auto prim_resize = GetValueNode<std::shared_ptr<ops::Resize>>(c_node->input(0));
  if (prim_resize == nullptr || prim_resize->GetAttr(ops::kNewHeight) == nullptr ||
      prim_resize->GetAttr(ops::kNewWidth) == nullptr) {
    return false;
  }
  int64_t new_height = prim_resize->get_new_height();
  int64_t new_width = prim_resize->get_new_width();
  return new_height != 0 && new_width != 0;
}
}  // namespace

STATUS TfliteInputsAdjustPass::ReplaceInt64ParameterNode(const FuncGraphPtr &func_graph,
                                                         const ParameterPtr &param_node) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(param_node != nullptr);
  if (param_node->abstract() == nullptr) {
    MS_LOG(ERROR) << "parameter node abstract is invalid.";
    return lite::RET_NULL_PTR;
  }
  auto abstract_tensor = param_node->abstract()->cast<abstract::AbstractTensorPtr>();
  if (abstract_tensor == nullptr) {
    MS_LOG(ERROR) << "param node has no abstract tensor.";
    return lite::RET_NULL_PTR;
  }
  if (abstract_tensor->element() == nullptr || abstract_tensor->element()->GetTypeTrack() == nullptr) {
    MS_LOG(ERROR) << "get typePtr failed.";
    return lite::RET_NULL_PTR;
  }
  if (abstract_tensor->element()->GetTypeTrack()->type_id() != kNumberTypeInt64) {
    MS_LOG(DEBUG) << "don't need to convert to int32.";
    return lite::RET_OK;
  }
  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  if (param_node->has_default()) {
    auto default_value = param_node->default_param();
    if (default_value == nullptr) {
      MS_LOG(ERROR) << "default data is nullptr.";
      return lite::RET_NULL_PTR;
    }
    auto param_value = default_value->cast<ParamValueLitePtr>();
    if (param_value == nullptr) {
      MS_LOG(ERROR) << "default data is not paramvaluelite.";
      return lite::RET_NULL_PTR;
    }
    auto param_node_new = BuildParameterNode(func_graph, param_node, param_value);
    manager->Replace(param_node, param_node_new);
  } else {
    // set graph input
    param_node->abstract()->set_type(TypeIdToType(kNumberTypeInt32));
  }
  return lite::RET_OK;
}

STATUS TfliteInputsAdjustPass::AdjustSlice(const AnfNodePtr &node, const FuncGraphPtr &graph) {
  auto cnode = node->cast<CNodePtr>();
  if (cnode->inputs().size() < 4) {
    MS_LOG(ERROR) << "Slice should own 3 inputs";
    return RET_ERROR;
  }

  auto begin_param_node = cnode->input(2)->cast<ParameterPtr>();
  auto size_param_node = cnode->input(3)->cast<ParameterPtr>();
  if (ReplaceInt64ParameterNode(graph, begin_param_node) == RET_OK &&
      ReplaceInt64ParameterNode(graph, size_param_node) == RET_OK) {
    return RET_OK;
  } else {
    MS_LOG(ERROR) << "Adjust inputs for Slice failed";
    return RET_ERROR;
  }
}

bool TfliteInputsAdjustPass::Run(const FuncGraphPtr &graph) {
  auto node_list = TopoSort(graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNode>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto primitive_c = GetValueNode<PrimitiveCPtr>(cnode->input(0));
    if (CheckPrimitiveType(node, prim::kPrimConv2dTransposeFusion)) {
      cnode->set_input(1, cnode->input(3));
      auto inputs = cnode->inputs();
      inputs.pop_back();
      cnode->set_inputs(inputs);

      auto input_quant_params_ptr = primitive_c->GetAttr("quant_params");
      if (input_quant_params_ptr == nullptr) {
        continue;
      }
      auto input_quant_params_holder = input_quant_params_ptr->cast<lite::QuantParamHolderPtr>();
      if (input_quant_params_holder == nullptr) {
        MS_LOG(ERROR) << "quant param is invalid.";
        return false;
      }
      auto input_quant_params = input_quant_params_holder->input_quant_params();
      input_quant_params[0] = input_quant_params.at(2);
      input_quant_params.pop_back();
      input_quant_params_holder->set_input_quant_params(input_quant_params);
      continue;
    }

    if (CheckPrimitiveType(node, prim::kPrimSplit) && cnode->inputs().size() == 3) {
      cnode->set_input(1, cnode->input(2));
      auto inputs = cnode->inputs();
      inputs.pop_back();
      cnode->set_inputs(inputs);

      auto input_quant_params_ptr = primitive_c->GetAttr("quant_params");
      if (input_quant_params_ptr == nullptr) {
        continue;
      }
      auto input_quant_params_holder = input_quant_params_ptr->cast<lite::QuantParamHolderPtr>();
      if (input_quant_params_holder == nullptr) {
        MS_LOG(ERROR) << "quant param is invalid.";
        return false;
      }
      auto input_quant_params = input_quant_params_holder->input_quant_params();
      input_quant_params[0] = input_quant_params.at(1);
      input_quant_params.pop_back();
      input_quant_params_holder->set_input_quant_params(input_quant_params);
      continue;
    }

    if (CheckPrimitiveType(node, prim::kPrimArgMinFusion) || CheckPrimitiveType(node, prim::kPrimArgMaxFusion) ||
        CheckPrimitiveType(node, prim::kPrimSpaceToBatch) || CheckPrimitiveType(node, prim::kPrimBatchToSpace) ||
        CheckPrimitiveType(node, prim::kPrimSpaceToBatchND) || CheckPrimitiveType(node, prim::kPrimBatchToSpaceND) ||
        CheckPrimitiveType(node, prim::kPrimSpaceToDepth) ||
        (CheckPrimitiveType(node, prim::kPrimResize) && CheckResize(cnode))) {
      std::vector<AnfNodePtr> new_inputs;
      new_inputs.emplace_back(cnode->input(0));
      new_inputs.emplace_back(cnode->input(1));
      cnode->set_inputs(new_inputs);
      continue;
    }

    if (CheckPrimitiveType(node, prim::kPrimSliceFusion)) {
      if (AdjustSlice(node, graph) != RET_OK) {
        return false;
      } else {
        continue;
      }
    }
  }
  return true;
}
}  // namespace mindspore::opt
