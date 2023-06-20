/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "tools/converter/import/mindir_adjust.h"
#include <vector>
#include <memory>
#include <set>
#include <map>
#include <algorithm>
#include <utility>
#include "mindspore/core/ops/array_ops.h"
#include "tools/converter/converter_context.h"
#include "tools/converter/quantizer/quant_param_holder.h"
#include "src/common/log_adapter.h"
#include "src/common/quant_utils.h"
#include "tools/converter/parser/parser_utils.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"
#include "ops/fake_quant_param.h"

namespace mindspore {
namespace lite {
namespace {
int ConvertQuantParam(const api::SharedPtr<mindspore::ops::FakeQuantParam> &fake_quant_prim,
                      std::vector<schema::QuantParamT> *quant_params) {
  MS_CHECK_TRUE_MSG(fake_quant_prim != nullptr, RET_NULL_PTR, "fake_quant_prim is nullptr.");
  MS_CHECK_TRUE_MSG(quant_params != nullptr, RET_NULL_PTR, "quant_params is nullptr.");
  schema::QuantParamT quant_param;
  auto scale = fake_quant_prim->get_scales();
  auto zp = fake_quant_prim->get_zero_points();
  if (scale.size() != zp.size()) {
    MS_LOG(ERROR) << "The number of quant params scale and zero_points should be same.";
    return RET_ERROR;
  }
  quant_params->resize(scale.size());
  for (size_t i = 0; i < scale.size(); i++) {
    quant_param.inited = True;
    quant_param.scale = scale[i];
    quant_param.zeroPoint = zp[i];
    (*quant_params)[i] = quant_param;
  }
  return lite::RET_OK;
}

int ConvertNodesQuantParam(const std::vector<std::shared_ptr<AnfNode>> &nodes,
                           std::map<int, std::vector<schema::QuantParamT>> *quant_params) {
  std::vector<schema::QuantParamT> quants;
  for (size_t i = 0; i < nodes.size(); i++) {
    quants.clear();
    if (IsPrimitiveCNode(nodes[i], prim::kPrimFakeQuantParam)) {
      auto fake_quant_prim =
        ops::GetOperator<mindspore::ops::FakeQuantParam>(nodes[i]->cast<CNodePtr>()->input(0)->cast<ValueNodePtr>());
      auto status = ConvertQuantParam(fake_quant_prim, &quants);
      if (status != lite::RET_OK) {
        MS_LOG(ERROR) << "Convert quant param from FakeQuantParam operation failed.";
        return lite::RET_ERROR;
      }
    }
    if (!quants.empty()) {
      quant_params->insert({i, quants});
    }
  }
  return lite::RET_OK;
}

int RemoveFakeQuantParam(const FuncGraphPtr &fg) {
  MS_CHECK_TRUE_MSG(fg != nullptr, RET_NULL_PTR, "fg is nullptr.");
  auto manager = fg->manager();
  auto node_list = TopoSort(fg->get_return());
  for (auto &node : node_list) {
    if (IsPrimitiveCNode(node, prim::kPrimFakeQuantParam)) {
      auto inputs = node->cast<CNodePtr>()->inputs();
      if (std::any_of(inputs.begin(), inputs.end(), [](const std::shared_ptr<AnfNode> &input) {
            return IsPrimitiveCNode(input, prim::kPrimFakeQuantParam);
          })) {
        MS_LOG(ERROR) << "Two FakeQuantParam operators can't be joined together in mindir origin model";
        return RET_ERROR;
      }

      auto iter = manager->node_users().find(node);
      if (iter != manager->node_users().end()) {
        auto outputs_set = manager->node_users()[node];
        if (std::any_of(outputs_set.begin(), outputs_set.end(),
                        [](const std::pair<std::shared_ptr<AnfNode>, int> &output) {
                          return IsPrimitiveCNode(output.first, prim::kPrimFakeQuantParam);
                        })) {
          MS_LOG(ERROR) << "Two FakeQuantParam operators can't be joined together in mindir origin model";
          return RET_ERROR;
        }
      }
      auto pre_node = node->cast<CNodePtr>()->input(1);
      (void)manager->Replace(node, pre_node);
    }
  }
  return RET_OK;
}

int GetNodeQuantParam(std::shared_ptr<AnfNode> anf_node, const PrimitivePtr &primitive,
                      const FuncGraphManagerPtr &manager) {
  if (!utils::isa<CNodePtr>(anf_node)) {
    MS_LOG(INFO) << "Only cnode need to convert primitive.";
    return RET_NO_CHANGE;
  }

  std::map<int, std::vector<schema::QuantParamT>> input_quant_params;
  std::map<int, std::vector<schema::QuantParamT>> output_quant_params;
  auto cnode = anf_node->cast<CNodePtr>();
  auto inputs = cnode->inputs();
  inputs.erase(inputs.begin());
  auto status = ConvertNodesQuantParam(inputs, &input_quant_params);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Convert input quant param failed.";
    return RET_ERROR;
  }

  auto iter = manager->node_users().find(anf_node);
  std::vector<AnfNodePtr> outputs;
  if (iter != manager->node_users().end()) {
    auto outputs_set = manager->node_users()[anf_node];
    std::transform(outputs_set.begin(), outputs_set.end(), std::back_inserter(outputs),
                   [](const std::pair<std::shared_ptr<AnfNode>, int> &output) { return output.first; });
    status = ConvertNodesQuantParam(outputs, &output_quant_params);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Convert output quant param failed.";
      return RET_ERROR;
    }
    if (output_quant_params.size() > 1 || (output_quant_params.size() == 1 && outputs.size() != 1)) {
      MS_LOG(ERROR) << "There can only be one FakeQuantParam as the output of " << anf_node->fullname_with_scope();
      return RET_ERROR;
    }
  }

  if (!input_quant_params.empty() || !output_quant_params.empty()) {
    auto quant_params_holder = std::make_shared<QuantParamHolder>(inputs.size(), outputs.size());
    MSLITE_CHECK_PTR(quant_params_holder);
    for (auto &input : input_quant_params) {
      quant_params_holder->set_input_quant_param(input.first, input.second);
    }
    for (auto &output : output_quant_params) {
      quant_params_holder->set_output_quant_param(output.first, output.second);
    }
    primitive->AddAttr("quant_params", quant_params_holder);
  }
  return RET_OK;
}
}  // namespace

int MindirAdjust::AdjustInputDataType(AnfNodePtr anf_node) {
  MS_CHECK_TRUE_MSG(anf_node != nullptr, RET_ERROR, "anf_node is nullptr");
  auto param_node = anf_node->cast<ParameterPtr>();
  MS_CHECK_TRUE_MSG(param_node != nullptr, RET_ERROR, "param_node is nullptr");
  auto abstract = param_node->abstract();
  MS_CHECK_TRUE_MSG(abstract != nullptr, RET_ERROR, "abstract is nullptr");
  auto abstract_tensor = abstract->cast<abstract::AbstractTensorPtr>();
  MS_CHECK_TRUE_MSG(abstract_tensor != nullptr, RET_ERROR, "param node has no abstract tensor.");
  auto tensor_element = abstract_tensor->element();
  MS_CHECK_TRUE_MSG(tensor_element != nullptr, RET_ERROR, "abstract tensor's element is null.");
  auto type_ptr = tensor_element->GetTypeTrack();
  MS_CHECK_TRUE_MSG(type_ptr != nullptr, RET_ERROR, "Type pointer is null.");
  auto org_type = type_ptr->type_id();
  if (!param_node->has_default() && (org_type == kNumberTypeInt64 || org_type == kNumberTypeFloat64)) {
    TypeId dst_type = org_type == kNumberTypeInt64 ? kNumberTypeInt32 : kNumberTypeFloat32;
    tensor_element->set_type(TypeIdToType(dst_type));
  }
  return RET_OK;
}

int MindirAdjust::ValueNodeInt64Convert(AnfNodePtr anf_node) {
  MS_CHECK_TRUE_MSG(anf_node != nullptr, RET_ERROR, "anf_node is nullptr");
  if (!utils::isa<ValueNodePtr>(anf_node)) {
    return lite::RET_NO_CHANGE;
  }
  auto value_node = anf_node->cast<ValueNodePtr>();
  MS_CHECK_TRUE_MSG(value_node != nullptr, RET_ERROR, "value_node is nullptr");
  if (value_node->abstract() == nullptr) {
    return lite::RET_NO_CHANGE;
  }
  auto abstract_tensor = value_node->abstract()->cast<abstract::AbstractTensorPtr>();
  if (abstract_tensor == nullptr) {
    return lite::RET_NO_CHANGE;
  }
  auto value = value_node->value();
  if (value != nullptr && value->isa<tensor::Tensor>()) {
    if (abstract_tensor->element() == nullptr) {
      MS_LOG(ERROR) << "abstractTensor->element() is nullptr.";
      return RET_ERROR;
    }
    auto type_ptr = abstract_tensor->element()->GetTypeTrack();
    if (type_ptr->type_id() == kNumberTypeInt64) {
      MS_CHECK_TRUE_MSG(utils::cast<abstract::ShapePtr>(abstract_tensor->BuildShape()) != nullptr, RET_NULL_PTR,
                        "Failed to cast pointer.");
      auto shape_vector = utils::cast<abstract::ShapePtr>(abstract_tensor->BuildShape())->shape();
      auto dest_tensor_info = std::make_shared<tensor::Tensor>(kNumberTypeInt32, shape_vector);
      MS_CHECK_TRUE_MSG(dest_tensor_info != nullptr, RET_NULL_PTR, "dest_tensor_info is nullptr.");
      MS_CHECK_TRUE_MSG(dest_tensor_info->data_c() != nullptr, RET_ERROR, "dest_tensor_info->data_c() is nullptr");
      MS_CHECK_TRUE_MSG(dest_tensor_info->data().nbytes() >= static_cast<int>(sizeof(int32_t)), RET_ERROR,
                        "num_bits_tensor->data_c() is not longer enough for int32_t");
      auto *dest_data_buf = reinterpret_cast<int32_t *>(dest_tensor_info->data_c());
      MS_CHECK_TRUE_MSG(dest_data_buf != nullptr, RET_NULL_PTR, "dest_data_buf is nullptr.");
      auto src_tensor_info = value->cast<tensor::TensorPtr>();
      MS_CHECK_TRUE_MSG(src_tensor_info != nullptr, RET_NULL_PTR, "src_tensor_info is nullptr.");
      MS_CHECK_TRUE_MSG(src_tensor_info->data_c() != nullptr, RET_ERROR, "src_tensor_info->data_c() is nullptr");
      MS_CHECK_TRUE_MSG(src_tensor_info->data().nbytes() >= static_cast<int>(sizeof(int64_t)), RET_ERROR,
                        "num_bits_tensor->data_c() is not longer enough for int64_t");
      auto *src_data_buf = reinterpret_cast<int64_t *>(src_tensor_info->data_c());
      MS_CHECK_TRUE_MSG(dest_tensor_info->ElementsNum() == src_tensor_info->ElementsNum(), RET_ERROR,
                        "Sizes don't match.");
      for (int i = 0; i < dest_tensor_info->ElementsNum(); i++) {
        dest_data_buf[i] = src_data_buf[i];
      }
      abstract_tensor->element()->set_type(TypeIdToType(kNumberTypeInt32));
      value_node->set_value(dest_tensor_info);
    }
  }
  return lite::RET_NO_CHANGE;
}

int MindirAdjust::ConvertQuantParams(std::shared_ptr<AnfNode> anf_node, const FuncGraphManagerPtr &manager) {
  MS_CHECK_TRUE_MSG(anf_node != nullptr, RET_ERROR, "anf_node is nullptr");
  if (!utils::isa<CNodePtr>(anf_node)) {
    MS_LOG(INFO) << "only cnode need to convert primitive.";
    return lite::RET_NO_CHANGE;
  }
  auto cnode = anf_node->cast<CNodePtr>();
  if (cnode->inputs().empty() || cnode->input(0) == nullptr) {
    MS_LOG(ERROR) << "the cnode is invalid.";
    return lite::RET_NULL_PTR;
  }
  if (utils::isa<CNodePtr>(cnode->input(0))) {
    MS_LOG(DEBUG) << "call cnode no need to convert primitive.";
    return lite::RET_NO_CHANGE;
  }
  auto value_node = cnode->input(0)->cast<ValueNodePtr>();
  if (value_node == nullptr || value_node->value() == nullptr) {
    MS_LOG(ERROR) << "value node is invalid.";
    return lite::RET_NULL_PTR;
  }
  auto primitive = value_node->value()->cast<PrimitivePtr>();
  if (primitive == nullptr) {
    if (utils::isa<FuncGraphPtr>(value_node->value())) {
      MS_LOG(DEBUG) << "is a funcgraph.";
      return lite::RET_NO_CHANGE;
    } else {
      MS_LOG(ERROR) << "the value is not primitive.";
      return lite::RET_ERROR;
    }
  }
  return GetNodeQuantParam(anf_node, primitive, manager);
}

int MindirAdjust::ResetFuncGraph(const FuncGraphPtr &fg, std::set<FuncGraphPtr> all_func_graphs) {
  MS_CHECK_TRUE_MSG(fg != nullptr, RET_NULL_PTR, "fg is nullptr.");
  auto manager = fg->manager();
  MS_CHECK_TRUE_MSG(manager != nullptr, RET_NULL_PTR, "manager is nullptr.");
  manager->Clear();
  manager->AddFuncGraph(fg, true);
  auto status = RemoveFakeQuantParam(fg);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Remove FakeQuantParam operators failed.";
    return RET_ERROR;
  }
  for (auto &item : all_func_graphs) {
    if (item == fg) {
      continue;
    }
    manager->AddFuncGraph(item);
    status = RemoveFakeQuantParam(item);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Remove FakeQuantParam operators failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

bool MindirAdjust::Run(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_MSG(func_graph != nullptr, false, "func_graph is nullptr.");
  if (this->fmk_type_ != converter::kFmkTypeMs) {
    MS_LOG(INFO) << "The framework type of model should be mindir.";
    return true;
  }
  std::set<FuncGraphPtr> all_func_graphs = {};
  GetAllFuncGraph(func_graph, &all_func_graphs);
  for (auto &graph : all_func_graphs) {
    auto manager = graph->manager();
    MS_CHECK_TRUE_MSG(manager != nullptr, RET_NULL_PTR, "manager is nullptr.");
    auto node_list = TopoSort(graph->get_return());
    int status = lite::RET_OK;
    bool success_flag = true;
    for (auto &node : node_list) {
      if (utils::isa<CNodePtr>(node)) {
        status = ConvertQuantParams(node, manager);
      } else if (utils::isa<ParameterPtr>(node)) {
        status = AdjustInputDataType(node);
      } else if (utils::isa<ValueNodePtr>(node)) {
        status = ValueNodeInt64Convert(node);
      }
      if (status != lite::RET_OK && status != lite::RET_NO_CHANGE) {
        lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
        success_flag = false;
      }
    }
    if (!success_flag) {
      MS_LOG(ERROR) << "Adjust mindir failed.";
      return false;
    }
  }
  if (ResetFuncGraph(func_graph, all_func_graphs) != RET_OK) {
    MS_LOG(ERROR) << "ResetFuncGraph failed.";
    return false;
  }
  return true;
}
}  // namespace lite
}  // namespace mindspore
