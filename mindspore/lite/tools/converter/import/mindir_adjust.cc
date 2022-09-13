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
#include "tools/converter/converter_context.h"
#include "tools/converter/quantizer/quant_param_holder.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "src/common/log_adapter.h"
#include "src/common/quant_utils.h"
#include "tools/common/node_util.h"
#include "tools/converter/parser/parser_utils.h"
#include "nnacl/op_base.h"
#include "ops/core_ops.h"

namespace mindspore {
namespace lite {
namespace {
int ConvertInputQuantParam(const PrimitivePtr &prim, bool input_narrow_range, bool weight_narrow_range,
                           int32_t act_numbits, int32_t weight_numbits,
                           std::map<int, std::vector<schema::QuantParamT>> *input_quant_params) {
  std::vector<schema::QuantParamT> quants;
  schema::QuantParamT quant_param;
  auto input_min = prim->GetAttr("input_minq");
  auto input_max = prim->GetAttr("input_maxq");
  if (input_min != nullptr && input_max != nullptr) {
    auto input_min_ptr = input_min->cast<tensor::TensorPtr>();
    MS_ASSERT(input_min_ptr != nullptr);
    auto input_max_ptr = input_max->cast<tensor::TensorPtr>();
    MS_ASSERT(input_max_ptr != nullptr);
    MS_CHECK_TRUE_MSG(input_min_ptr->data_c() != nullptr, RET_ERROR, "input_min_ptr->data_c() is nullptr");
    MS_CHECK_TRUE_MSG(input_max_ptr->data_c() != nullptr, RET_ERROR, "input_max_ptr->data_c() is nullptr");
    auto *min_buf = static_cast<float *>(input_min_ptr->data_c());
    auto *max_buf = static_cast<float *>(input_max_ptr->data_c());
    quant_param.min = *min_buf;
    quant_param.max = *max_buf;
    auto ret = CalQuantizationParams(&quant_param, quant_param.min, quant_param.max, act_numbits, input_narrow_range);
    MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "Failed to calculate quant parameters.");
    quants.emplace_back(quant_param);
    input_quant_params->insert({0, quants});
  }

  quants.clear();
  auto filter_min = prim->GetAttr("filter_minq");
  auto filter_max = prim->GetAttr("filter_maxq");
  if (filter_min != nullptr && filter_max != nullptr) {
    auto filter_min_ptr = filter_min->cast<tensor::TensorPtr>();
    MS_ASSERT(filter_min_ptr != nullptr);
    auto filter_max_ptr = filter_max->cast<tensor::TensorPtr>();
    MS_ASSERT(filter_max_ptr != nullptr);
    MS_CHECK_TRUE_MSG(filter_min_ptr->data_c() != nullptr, RET_ERROR, "filter_min_ptr->data_c() is nullptr");
    MS_CHECK_TRUE_MSG(filter_max_ptr->data_c() != nullptr, RET_ERROR, "filter_max_ptr->data_c() is nullptr");
    auto *min_buf = static_cast<float *>(filter_min_ptr->data_c());
    auto *max_buf = static_cast<float *>(filter_max_ptr->data_c());
    quant_param.min = FLT_MAX;
    quant_param.max = FLT_MIN;
    for (int i = 0; i < filter_min_ptr->ElementsNum(); ++i) {
      schema::QuantParamT tmp_quant_param;
      tmp_quant_param.min = *min_buf;
      tmp_quant_param.max = *max_buf;
      auto ret = CalQuantizationParams(&tmp_quant_param, tmp_quant_param.min, tmp_quant_param.max, weight_numbits,
                                       weight_narrow_range);
      MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "Failed to calculate quant parameters.");
      quants.emplace_back(tmp_quant_param);
      min_buf++;
      max_buf++;
    }
    input_quant_params->insert({1, quants});
  }
  return lite::RET_OK;
}

int ConvertOutputQuantParam(const PrimitivePtr &prim, bool narrow_range, int32_t numbits,
                            std::map<int, std::vector<schema::QuantParamT>> *output_quant_params) {
  std::vector<schema::QuantParamT> quants;
  schema::QuantParamT quant_param;
  auto outputMin = prim->GetAttr("output_minq");
  auto outputMax = prim->GetAttr("output_maxq");
  if (outputMin != nullptr && outputMax != nullptr) {
    auto outputMinPtr = outputMin->cast<tensor::TensorPtr>();
    auto outputMaxPtr = outputMax->cast<tensor::TensorPtr>();
    MS_ASSERT(outputMinPtr != nullptr);
    MS_ASSERT(outputMaxPtr != nullptr);
    MS_CHECK_TRUE_MSG(outputMinPtr->data_c() != nullptr, RET_ERROR, "outputMinPtr->data_c() is nullptr");
    MS_CHECK_TRUE_MSG(outputMaxPtr->data_c() != nullptr, RET_ERROR, "outputMaxPtr->data_c() is nullptr");
    auto *minBuf = static_cast<float *>(outputMinPtr->data_c());
    auto *maxBuf = static_cast<float *>(outputMaxPtr->data_c());
    quant_param.min = *minBuf;
    quant_param.max = *maxBuf;
    auto ret = CalQuantizationParams(&quant_param, quant_param.min, quant_param.max, numbits, narrow_range);
    MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "Failed to calculate quant parameters.");
    quants.emplace_back(quant_param);
    output_quant_params->insert({0, quants});
  }
  return lite::RET_OK;
}

int GetNarrowRange(const PrimitivePtr &prim, const std::string &narrow_range_str, bool *const narrow_range_param) {
  MS_CHECK_TRUE_MSG(narrow_range_param != nullptr, RET_ERROR, "narrow_range_param is nullptr");
  auto narrow_range = prim->GetAttr(narrow_range_str);
  if (narrow_range != nullptr) {
    if (utils::isa<tensor::TensorPtr>(narrow_range)) {
      auto narrow_range_tensor = narrow_range->cast<tensor::TensorPtr>();
      MS_ASSERT(narrow_range_tensor != nullptr);
      MS_CHECK_TRUE_MSG(narrow_range_tensor->data_c() != nullptr, RET_ERROR,
                        "narrow_range_tensor->data_c() is nullptr");
      *narrow_range_param = *reinterpret_cast<bool *>(narrow_range_tensor->data_c());
    } else if (utils::isa<ImmTraits<bool>::type>(narrow_range)) {
      *narrow_range_param = GetValue<bool>(narrow_range);
    } else {
      MS_LOG(ERROR) << "valueptr is invalid.";
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}

int GetNumBits(const PrimitivePtr &prim, const std::string &num_bits_str, int *const num_bits_param) {
  MS_CHECK_TRUE_MSG(num_bits_param != nullptr, RET_ERROR, "num_bits_param is nullptr");
  auto num_bits = prim->GetAttr(num_bits_str);
  if (num_bits != nullptr) {
    if (utils::isa<tensor::TensorPtr>(num_bits)) {
      auto num_bits_tensor = num_bits->cast<tensor::TensorPtr>();
      MS_ASSERT(num_bits_tensor != nullptr);
      MS_CHECK_TRUE_MSG(num_bits_tensor->data_c() != nullptr, RET_ERROR, "num_bits_tensor->data_c() is nullptr");
      MS_CHECK_TRUE_MSG(num_bits_tensor->data().nbytes() >= static_cast<int>(sizeof(int64_t)), RET_ERROR,
                        "num_bits_tensor->data_c() is not longer enough for int64_t");
      *num_bits_param = *reinterpret_cast<int64_t *>(num_bits_tensor->data_c());
    } else if (utils::isa<ImmTraits<int64_t>::type>(num_bits)) {
      *num_bits_param = GetValue<int64_t>(num_bits);
    } else {
      MS_LOG(ERROR) << "valueptr is invalid.";
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}

int ConvertQuantParam(const PrimitivePtr &prim, const std::vector<AnfNodePtr> &inputs) {
  bool input_narrow_range_param = false;
  auto status = GetNarrowRange(prim, "input_narrow_range", &input_narrow_range_param);
  MS_CHECK_TRUE_MSG(status == RET_OK, RET_ERROR, "Get input narrow range failed.");
  bool weight_narrow_range_param = true;
  status = GetNarrowRange(prim, "weight_narrow_range", &weight_narrow_range_param);
  MS_CHECK_TRUE_MSG(status == RET_OK, RET_ERROR, "Get weight narrow range failed.");
  bool output_narrow_range_param = false;
  status = GetNarrowRange(prim, "output_narrow_range", &output_narrow_range_param);
  MS_CHECK_TRUE_MSG(status == RET_OK, RET_ERROR, "Get output narrow range failed.");
  int32_t act_num_bits_param = 8;
  status = GetNumBits(prim, "act_num_bits", &act_num_bits_param);
  MS_CHECK_TRUE_MSG(status == RET_OK, RET_ERROR, "Get activation num_bits failed.");
  int32_t weight_num_bits_param = 8;
  status = GetNumBits(prim, "weight_num_bits", &weight_num_bits_param);
  MS_CHECK_TRUE_MSG(status == RET_OK, RET_ERROR, "Get weight num_bits failed.");

  std::map<int, std::vector<schema::QuantParamT>> input_quant_params;
  std::map<int, std::vector<schema::QuantParamT>> output_quant_params;
  status = ConvertInputQuantParam(prim, input_narrow_range_param, weight_narrow_range_param, act_num_bits_param,
                                  weight_num_bits_param, &input_quant_params);
  MS_CHECK_TRUE_MSG(status == RET_OK, RET_ERROR, "Compute input quant param failed.");
  status = ConvertOutputQuantParam(prim, output_narrow_range_param, act_num_bits_param, &output_quant_params);
  MS_CHECK_TRUE_MSG(status == RET_OK, RET_ERROR, "Compute output quant param failed.");

  if (!input_quant_params.empty() || !output_quant_params.empty()) {
    auto quant_params_holder = std::make_shared<QuantParamHolder>(0, 0);
    MSLITE_CHECK_PTR(quant_params_holder);
    for (auto &iter : input_quant_params) {
      quant_params_holder->set_input_quant_param(iter.first, iter.second);
    }
    for (auto &iter : output_quant_params) {
      quant_params_holder->set_output_quant_param(iter.first, iter.second);
    }
    prim->AddAttr("quant_params", quant_params_holder);
  }
  return lite::RET_OK;
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
  MS_ASSERT(value_node != nullptr);
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

int MindirAdjust::ComputeQuantParams(std::shared_ptr<AnfNode> anf_node) {
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
  auto inputs = cnode->inputs();
  inputs.erase(inputs.begin());

  if (ConvertQuantParam(primitive, inputs) != lite::RET_OK) {
    MS_LOG(ERROR) << "compute quant param failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

int MindirAdjust::UpdateConv2DTransposeInput(const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  if (!opt::CheckPrimitiveType(cnode, prim::kPrimConv2dTransposeFusion)) {
    return RET_OK;
  }
  auto inputs = cnode->inputs();
  if (inputs.size() != opt::kInputSizeFour) {
    MS_LOG(ERROR) << "the input size of mindir conv2dtranspose should be 4, but now is " << inputs.size()
                  << ", please check.";
    return RET_ERROR;
  }
  inputs.pop_back();
  cnode->set_inputs(inputs);
  return RET_OK;
}

int MindirAdjust::ResetFuncGraph(const FuncGraphPtr &fg, std::set<FuncGraphPtr> all_func_graphs) {
  MS_CHECK_TRUE_MSG(fg != nullptr, RET_NULL_PTR, "fg is nullptr.");
  auto manager = fg->manager();
  MS_CHECK_TRUE_MSG(manager != nullptr, RET_NULL_PTR, "manager is nullptr.");
  manager->Clear();
  manager->AddFuncGraph(fg, true);
  for (auto &item : all_func_graphs) {
    if (item == fg) {
      continue;
    }
    manager->AddFuncGraph(item);
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
    auto node_list = TopoSort(graph->get_return());
    int status = lite::RET_OK;
    bool success_flag = true;
    for (auto &node : node_list) {
      if (utils::isa<CNodePtr>(node)) {
        status = ComputeQuantParams(node);
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
