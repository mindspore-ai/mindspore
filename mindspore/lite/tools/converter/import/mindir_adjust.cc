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
#include "tools/converter/import/mindir_adjust.h"
#include <vector>
#include <memory>
#include <set>
#include "tools/converter/converter_context.h"
#include "tools/converter/quant_param_holder.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "src/common/log_adapter.h"
#include "tools/common/node_util.h"
#include "tools/converter/parser/parser_utils.h"

namespace mindspore {
namespace lite {
namespace {
int ConvertInputQuantParam(const PrimitivePtr &prim, bool input_narrow_range, bool weight_narrow_range,
                           int32_t act_numbits, int32_t weight_numbits) {
  auto quant_param_holder = prim->GetAttr("quant_params")->cast<lite::QuantParamHolderPtr>();
  std::vector<schema::QuantParamT> quants;
  schema::QuantParamT quant_param;
  auto input_min = prim->GetAttr("input_minq");
  auto input_max = prim->GetAttr("input_maxq");
  if (input_min != nullptr && input_max != nullptr) {
    auto input_min_ptr = input_min->cast<tensor::TensorPtr>();
    auto input_max_ptr = input_max->cast<tensor::TensorPtr>();
    auto *min_buf = static_cast<float *>(input_min_ptr->data_c());
    auto *max_buf = static_cast<float *>(input_max_ptr->data_c());
    quant_param.min = *min_buf;
    quant_param.max = *max_buf;
    auto ret = lite::quant::CalQuantizationParams(&quant_param, quant_param.min, quant_param.max, input_narrow_range,
                                                  act_numbits);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Can't calculate quant parameters";
      return ret;
    }
    quants.emplace_back(quant_param);
    quant_param_holder->set_input_quant_param(0, quants);
  }

  quants.clear();
  auto filter_min = prim->GetAttr("filter_minq");
  auto filter_max = prim->GetAttr("filter_maxq");
  if (filter_min != nullptr && filter_max != nullptr) {
    auto filter_min_ptr = filter_min->cast<tensor::TensorPtr>();
    auto filter_max_ptr = filter_max->cast<tensor::TensorPtr>();
    auto *min_buf = static_cast<float *>(filter_min_ptr->data_c());
    auto *max_buf = static_cast<float *>(filter_max_ptr->data_c());
    quant_param.min = FLT_MAX;
    quant_param.max = FLT_MIN;
    for (int i = 0; i < filter_min_ptr->ElementsNum(); ++i) {
      schema::QuantParamT tmp_quant_param;
      tmp_quant_param.min = *min_buf;
      tmp_quant_param.max = *max_buf;
      auto ret = lite::quant::CalQuantizationParams(&tmp_quant_param, tmp_quant_param.min, tmp_quant_param.max,
                                                    weight_narrow_range, weight_numbits);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Can't calculate quant parameters";
        return ret;
      }
      quants.emplace_back(tmp_quant_param);
      min_buf++;
      max_buf++;
    }
    quant_param_holder->set_input_quant_param(1, quants);
  }
  return lite::RET_OK;
}

int ConvertOutputQuantParam(const PrimitivePtr &prim, bool narrow_range, int32_t numbits) {
  auto quant_param_holder = prim->GetAttr("quant_params")->cast<lite::QuantParamHolderPtr>();
  std::vector<schema::QuantParamT> quants;
  schema::QuantParamT quant_param;
  auto outputMin = prim->GetAttr("output_minq");
  auto outputMax = prim->GetAttr("output_maxq");
  if (outputMin != nullptr && outputMax != nullptr) {
    auto outputMinPtr = outputMin->cast<tensor::TensorPtr>();
    auto outputMaxPtr = outputMax->cast<tensor::TensorPtr>();
    auto *minBuf = static_cast<float *>(outputMinPtr->data_c());
    auto *maxBuf = static_cast<float *>(outputMaxPtr->data_c());
    quant_param.min = *minBuf;
    quant_param.max = *maxBuf;
    auto ret =
      lite::quant::CalQuantizationParams(&quant_param, quant_param.min, quant_param.max, narrow_range, numbits);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Can't calculate quant parameters";
      return ret;
    }
    quants.emplace_back(quant_param);
    quant_param_holder->set_output_quant_param(0, quants);
  }
  return lite::RET_OK;
}

int GetNarrowRange(const PrimitivePtr &prim, const std::string &narrow_range_str, bool *narrow_range_param) {
  auto narrow_range = prim->GetAttr(narrow_range_str);
  if (narrow_range != nullptr) {
    if (utils::isa<tensor::TensorPtr>(narrow_range)) {
      auto narrow_range_tensor = narrow_range->cast<tensor::TensorPtr>();
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

int GetNumBits(const PrimitivePtr &prim, const std::string &num_bits_str, int *num_bits_param) {
  auto num_bits = prim->GetAttr(num_bits_str);
  if (num_bits != nullptr) {
    if (utils::isa<tensor::TensorPtr>(num_bits)) {
      auto num_bits_tensor = num_bits->cast<tensor::TensorPtr>();
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
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "get input narrow range failed.";
    return status;
  }
  bool weight_narrow_range_param = true;
  status = GetNarrowRange(prim, "weight_narrow_range", &weight_narrow_range_param);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "get weight narrow range failed.";
    return status;
  }
  bool output_narrow_range_param = false;
  status = GetNarrowRange(prim, "output_narrow_range", &output_narrow_range_param);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "get output narrow range failed.";
    return status;
  }

  int32_t act_num_bits_param = 8;
  status = GetNumBits(prim, "act_num_bits", &act_num_bits_param);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "get activation num_bits failed.";
    return status;
  }
  int32_t weight_num_bits_param = 8;
  status = GetNumBits(prim, "weight_num_bits", &weight_num_bits_param);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "get weight num_bits failed.";
    return status;
  }
  status = ConvertInputQuantParam(prim, input_narrow_range_param, weight_narrow_range_param, act_num_bits_param,
                                  weight_num_bits_param);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "compute int quant param failed.";
    return status;
  }
  status = ConvertOutputQuantParam(prim, output_narrow_range_param, act_num_bits_param);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "compute output quant param failed.";
    return status;
  }
  return lite::RET_OK;
}
}  // namespace

int MindirAdjust::ValueNodeInt64Convert(AnfNodePtr anf_node) {
  if (!utils::isa<ValueNodePtr>(anf_node)) {
    return lite::RET_NO_CHANGE;
  }
  auto value_node = anf_node->cast<ValueNodePtr>();
  if (value_node->abstract() == nullptr) {
    return lite::RET_NO_CHANGE;
  }
  auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(value_node->abstract());
  if (abstract_tensor == nullptr) {
    return lite::RET_NO_CHANGE;
  }
  auto value = abstract_tensor->GetValueTrack();
  if (value != nullptr && value->isa<tensor::Tensor>()) {
    if (abstract_tensor->element() == nullptr) {
      MS_LOG(ERROR) << "abstractTensor->element() is nullptr.";
      return RET_ERROR;
    }
    auto type_ptr = abstract_tensor->element()->GetTypeTrack();
    if (type_ptr->type_id() == kNumberTypeInt64) {
      auto shape_vector = utils::cast<abstract::ShapePtr>(abstract_tensor->BuildShape())->shape();
      auto dest_tensor_info = std::make_shared<tensor::Tensor>(kNumberTypeInt32, shape_vector);
      auto *dest_data_buf = reinterpret_cast<int32_t *>(dest_tensor_info->data_c());
      auto src_tensor_info = value->cast<tensor::TensorPtr>();
      auto *src_data_buf = reinterpret_cast<int64_t *>(src_tensor_info->data_c());
      MS_ASSERT(dest_tensor_info->ElementsNum() == src_tensor_info->ElementsNum());
      for (int i = 0; i < dest_tensor_info->ElementsNum(); i++) {
        dest_data_buf[i] = src_data_buf[i];
      }
      abstract_tensor->set_value(dest_tensor_info);
      abstract_tensor->set_type(TypeIdToType(kNumberTypeInt32));
      abstract_tensor->element()->set_type(TypeIdToType(kNumberTypeInt32));
      value_node->set_value(dest_tensor_info);
    }
  }
  return lite::RET_NO_CHANGE;
}

int MindirAdjust::ComputeQuantParams(std::shared_ptr<AnfNode> anf_node) {
  if (!utils::isa<CNodePtr>(anf_node)) {
    MS_LOG(INFO) << "only cnode need to convert primitive.";
    return lite::RET_NO_CHANGE;
  }
  auto cnode = anf_node->cast<CNodePtr>();
  if (cnode->inputs().empty() || cnode->input(0) == nullptr) {
    MS_LOG(ERROR) << "the cnode is invalid.";
    return lite::RET_NULL_PTR;
  }
  auto value_node = cnode->input(0)->cast<ValueNodePtr>();
  if (value_node == nullptr || value_node->value() == nullptr) {
    MS_LOG(ERROR) << "value node is invalid.";
    return lite::RET_NULL_PTR;
  }
  auto primitive = value_node->value()->cast<PrimitivePtr>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "the value is not primitive.";
    return lite::RET_ERROR;
  }
  auto inputs = cnode->inputs();
  inputs.erase(inputs.begin());

  auto quant_param_holder =
    std::make_shared<lite::QuantParamHolder>(inputs.size(), lite::GetCNodeOutputsSize(anf_node, train_flag_));
  primitive->AddAttr("quant_params", quant_param_holder);

  if (ConvertQuantParam(primitive, inputs) != lite::RET_OK) {
    MS_LOG(ERROR) << "compute quant param failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

bool MindirAdjust::Run(const FuncGraphPtr &func_graph) {
  if (this->fmk_type_ != converter::kFmkTypeMs) {
    MS_LOG(INFO) << "The framework type of model should be mindir.";
    return lite::RET_OK;
  }
  MS_ASSERT(graph != nullptr);
  std::set<FuncGraphPtr> all_func_graphs = {};
  GetAllFuncGraph(func_graph, &all_func_graphs);
  for (auto &graph : all_func_graphs) {
    auto node_list = TopoSort(graph->get_return());
    int status = lite::RET_OK;
    bool success_flag = true;
    for (auto &node : node_list) {
      if (utils::isa<CNodePtr>(node)) {
        status = ComputeQuantParams(node);
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
  return true;
}
}  // namespace lite
}  // namespace mindspore
