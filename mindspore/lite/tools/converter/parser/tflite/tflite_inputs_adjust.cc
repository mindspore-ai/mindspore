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
#include "tools/converter/parser/tflite/tflite_inputs_adjust.h"
#include <vector>
#include <memory>
#include "ops/batch_to_space.h"
#include "ops/batch_to_space_nd.h"
#include "ops/fusion/arg_max_fusion.h"
#include "ops/fusion/arg_min_fusion.h"
#include "ops/op_utils.h"
#include "ops/resize.h"
#include "ops/space_to_batch.h"
#include "ops/space_to_batch_nd.h"
#include "ops/space_to_depth.h"
#include "tools/converter/quant_param_holder.h"
#include "tools/converter/quantizer/quant_cast.h"
#include "nnacl/op_base.h"
#include "src/common/log_util.h"

namespace mindspore::lite {
namespace {
constexpr size_t split_inputs_size = 3;
const std::vector<std::string> single_input_ops = {
  ops::kNameArgMaxFusion, ops::kNameArgMinFusion,   ops::kNameBatchToSpace, ops::kNameBatchToSpaceND,
  ops::kNameSpaceToBatch, ops::kNameSpaceToBatchND, ops::kNameSpaceToDepth};

bool CheckResize(const CNodePtr &cnode) {
  if (!opt::CheckPrimitiveType(cnode, prim::kPrimResize)) {
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
  MSLITE_CHECK_PTR(cnode);
  std::vector<AnfNodePtr> new_inputs = {cnode->input(0)};
  auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  auto input_quant_params = primitive->GetAttr("quant_params");
  if (input_quant_params == nullptr) {
    MS_LOG(ERROR) << "quant params holder is null";
    return RET_ERROR;
  }
  auto input_quant_params_holder = input_quant_params->cast<lite::QuantParamHolderPtr>();
  MS_ASSERT(input_quant_params_holder != nullptr);
  auto old_quant_params = input_quant_params_holder->get_input_quant_params();
  auto new_input_quant_holder =
    std::make_shared<lite::QuantParamHolder>(perm.size(), input_quant_params_holder->get_output_quant_params().size());
  MSLITE_CHECK_PTR(new_input_quant_holder);
  // add inputs as perm order
  size_t new_idx = 0;
  for (size_t idx : perm) {
    if (idx > cnode->inputs().size() - 1) {
      MS_LOG(ERROR) << "Idx " << idx << " is larger than inputs size: " << (cnode->inputs().size() - 1);
      return lite::RET_ERROR;
    }
    new_inputs.emplace_back(cnode->input(idx));
    auto quant_param = idx < old_quant_params.size() ? old_quant_params.at(idx) : std::vector<schema::QuantParamT>();
    new_input_quant_holder->set_input_quant_param(new_idx, quant_param);
    new_idx++;
  }

  for (size_t i = 0; i < input_quant_params_holder->get_output_quant_params().size(); i++) {
    new_input_quant_holder->set_output_quant_param(i, input_quant_params_holder->get_output_quant_params().at(i));
  }
  cnode->set_inputs(new_inputs);
  primitive->set_attr("quant_params", new_input_quant_holder);
  return lite::RET_OK;
}
}  // namespace

STATUS TfliteInputsAdjust::ReplaceInt64ParameterNode(const FuncGraphPtr &func_graph, const ParameterPtr &param_node) {
  MSLITE_CHECK_PTR(func_graph);
  MSLITE_CHECK_PTR(param_node);
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
  auto manager = Manage(func_graph, true);
  MSLITE_CHECK_PTR(manager);
  if (param_node->has_default()) {
    auto default_value = param_node->default_param();
    if (default_value == nullptr) {
      MS_LOG(ERROR) << "default data is nullptr.";
      return lite::RET_NULL_PTR;
    }
    auto tensor_info = default_value->cast<tensor::TensorPtr>();
    if (tensor_info == nullptr) {
      MS_LOG(ERROR) << "default data is not tensor::Tensor.";
      return lite::RET_NULL_PTR;
    }
    auto param_node_new = opt::BuildParameterNode(func_graph, param_node, tensor_info);
    manager->Replace(param_node, param_node_new);
  } else {
    // set graph input
    param_node->abstract()->set_type(TypeIdToType(kNumberTypeInt32));
  }
  return lite::RET_OK;
}

STATUS TfliteInputsAdjust::AdjustSlice(const AnfNodePtr &node, const FuncGraphPtr &graph) {
  MSLITE_CHECK_PTR(node);
  MSLITE_CHECK_PTR(graph);
  auto cnode = node->cast<CNodePtr>();
  MS_ASSERT(cnode != nullptr);
  if (cnode->inputs().size() < opt::kInputSizeFour) {
    MS_LOG(ERROR) << "Slice should own 3 inputs";
    return RET_ERROR;
  }

  auto begin_param_node = cnode->input(opt::kInputIndexTwo)->cast<ParameterPtr>();
  auto size_param_node = cnode->input(opt::kInputIndexThree)->cast<ParameterPtr>();
  // slice's begin and size could be variable
  if (begin_param_node != nullptr && ReplaceInt64ParameterNode(graph, begin_param_node) != RET_OK) {
    MS_LOG(ERROR) << "Adjust begin for Slice failed";
    return RET_ERROR;
  }
  if (size_param_node != nullptr && ReplaceInt64ParameterNode(graph, size_param_node) != RET_OK) {
    MS_LOG(ERROR) << "Adjust size for Slice failed";
    return RET_ERROR;
  }
  return RET_OK;
}

bool TfliteInputsAdjust::Run(const FuncGraphPtr &graph) {
  auto node_list = TopoSort(graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNode>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (opt::CheckPrimitiveType(cnode, prim::kPrimFill)) {
      // dims, value => value, dims
      if (RET_OK != ReorderCnodeInputs(cnode.get(), {2, 1})) {
        MS_LOG(ERROR) << "Reorder fill inputs failed";
        return false;
      }
      continue;
    }

    if (opt::CheckPrimitiveType(cnode, prim::kPrimConv2dTransposeFusion)) {
      // output_shape, weights, input => input, weight
      if (RET_OK != ReorderCnodeInputs(cnode.get(), {3, 2})) {
        MS_LOG(ERROR) << "Reorder deconv inputs failed";
        return false;
      }
      continue;
    }

    if (opt::CheckPrimitiveType(cnode, prim::kPrimSplit) && cnode->inputs().size() == split_inputs_size) {
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
    if (opt::CheckPrimitiveType(node, prim::kPrimSliceFusion)) {
      if (AdjustSlice(node, graph) == RET_OK) {
        continue;
      }
      return false;
    }
  }
  return true;
}
}  // namespace mindspore::lite
