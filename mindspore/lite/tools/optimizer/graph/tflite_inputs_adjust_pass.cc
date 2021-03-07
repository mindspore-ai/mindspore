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
#include "tools/optimizer/graph/tflite_inputs_adjust_pass.h"
#include <vector>
#include <memory>
#include "ops/batch_to_space.h"
#include "ops/batch_to_space_nd.h"
#include "ops/fusion/arg_max_fusion.h"
#include "ops/fusion/arg_min_fusion.h"
#include "ops/fusion/pad_fusion.h"
#include "ops/fusion/reduce_fusion.h"
#include "ops/op_utils.h"
#include "ops/resize.h"
#include "ops/space_to_batch.h"
#include "ops/space_to_batch_nd.h"
#include "ops/space_to_depth.h"
#include "tools/converter/quant_param_holder.h"
#include "tools/converter/quantizer/quant_cast.h"

namespace mindspore::opt {
namespace {
constexpr size_t split_inputs_size = 3;
const std::vector<std::string> single_input_ops = {
  ops::kNameArgMaxFusion, ops::kNameArgMinFusion,   ops::kNameBatchToSpace, ops::kNameBatchToSpaceND,
  ops::kNameSpaceToBatch, ops::kNameSpaceToBatchND, ops::kNameSpaceToDepth};

bool CheckResize(const CNodePtr &cnode) {
  if (!CheckPrimitiveType(cnode, prim::kPrimResize)) {
    return false;
  }
  auto prim_resize = GetValueNode<std::shared_ptr<ops::Resize>>(cnode->input(0));
  if (prim_resize == nullptr || prim_resize->GetAttr(ops::kNewHeight) == nullptr ||
      prim_resize->GetAttr(ops::kNewWidth) == nullptr) {
    return false;
  }
  int64_t new_height = prim_resize->get_new_height();
  int64_t new_width = prim_resize->get_new_width();
  return new_height != 0 && new_width != 0;
}

lite::STATUS ReorderCnodeInputs(CNode *cnode, const std::vector<size_t> &perm) {
  // add primitive first
  std::vector<AnfNodePtr> new_inputs = {cnode->input(0)};
  auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  auto input_quant_params = primitive->GetAttr("quant_params");
  auto input_quant_params_holder = input_quant_params == nullptr
                                     ? std::make_shared<lite::QuantParamHolder>()
                                     : input_quant_params->cast<lite::QuantParamHolderPtr>();
  auto old_quant_params = input_quant_params_holder->input_quant_params();
  auto new_input_quant_holder = std::make_shared<lite::QuantParamHolder>();
  // add inputs as perm order
  for (size_t idx : perm) {
    if (idx > cnode->inputs().size() - 1) {
      MS_LOG(ERROR) << "Idx " << idx << " is larger than inputs size: " << cnode->inputs().size() - 1;
      return lite::RET_ERROR;
    }
    new_inputs.emplace_back(cnode->input(idx));
    auto quant_param = idx < old_quant_params.size() ? old_quant_params.at(idx) : std::vector<schema::QuantParamT>();
    new_input_quant_holder->AddInputQuantParam(quant_param);
  }
  cnode->set_inputs(new_inputs);
  primitive->set_attr("quant_params", new_input_quant_holder);
  return lite::RET_OK;
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
    if (CheckPrimitiveType(cnode, prim::kPrimFill)) {
      // dims, value => value, dims
      if (RET_OK != ReorderCnodeInputs(cnode.get(), {2, 1})) {
        MS_LOG(ERROR) << "Reorder fill inputs failed";
        return false;
      }
      continue;
    }

    if (CheckPrimitiveType(cnode, prim::kPrimConv2dTransposeFusion)) {
      // output_shape, weights, input => input, weight
      if (RET_OK != ReorderCnodeInputs(cnode.get(), {3, 2})) {
        MS_LOG(ERROR) << "Reorder deconv inputs failed";
        return false;
      }
      continue;
    }

    if (CheckPrimitiveType(cnode, prim::kPrimSplit) && cnode->inputs().size() == split_inputs_size) {
      // axis, input, ??? => input, axis
      if (RET_OK != ReorderCnodeInputs(cnode.get(), {2, 1})) {
        MS_LOG(ERROR) << "Reorder split inputs failed";
        return false;
      }
      continue;
    }
    auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (lite::IsContain(single_input_ops, primitive->name()) || CheckResize(cnode)) {
      if (ReorderCnodeInputs(cnode.get(), {1}) != lite::RET_OK) {
        MS_LOG(ERROR) << "Reorder single input failed";
        return false;
      }
    }
    if (CheckPrimitiveType(node, prim::kPrimSliceFusion)) {
      if (AdjustSlice(node, graph) == RET_OK) {
        continue;
      }
      return false;
    }
  }
  return true;
}
}  // namespace mindspore::opt
