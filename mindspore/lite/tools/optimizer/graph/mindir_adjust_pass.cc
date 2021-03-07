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
#include "tools/optimizer/graph/mindir_adjust_pass.h"
#include <algorithm>
#include <vector>
#include <memory>

#include "tools/converter/converter_context.h"
#include "tools/converter/quant_param_holder.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "src/common/log_adapter.h"
#include "src/tensor.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kDoubleNum = 2;
void FillDefaultInputQuantParamIfNeed(const PrimitivePtr &prim, const size_t &input_size) {
  auto quant_param_valueptr = prim->GetAttr("quant_params");
  if (quant_param_valueptr == nullptr) {
    prim->AddAttr("quant_params", std::make_shared<lite::QuantParamHolder>());
  }
  auto quant_param_holder = prim->GetAttr("quant_params")->cast<lite::QuantParamHolderPtr>();
  std::vector<schema::QuantParamT> quants;
  schema::QuantParamT quant_param;
  auto input_quant_params = quant_param_holder->input_quant_params();
  if (input_quant_params.size() == kDoubleNum) {
    quants.clear();
    quant_param.min = 0.0;
    quant_param.max = 0.0;
    quant_param.dstDtype = kNumberTypeInt32;
    quant_param.inited = input_quant_params.at(0).at(0).inited && input_quant_params.at(1).at(0).inited;
    quant_param.inited = false;
    quant_param.zeroPoint = 0;
    if (quant_param.inited) {
      quant_param.scale = input_quant_params.at(0).at(0).scale * input_quant_params.at(1).at(0).scale;
    }
    quant_param.roundType = 1;
    quant_param.multiplier = 1;
    quants.emplace_back(quant_param);
    input_quant_params.emplace_back(quants);
  }
  // fill input_quant_param_ by not inited quant_parm
  if (input_quant_params.size() < input_size) {
    schema::QuantParamT tmpQuantParam;
    quants.emplace_back(tmpQuantParam);
    input_quant_params.insert(input_quant_params.end(), input_size - input_quant_params.size(), quants);
  }
  quant_param_holder->set_input_quant_params(input_quant_params);
}

int ConvertInputQuantParam(const PrimitivePtr &prim, bool narrow_range, int32_t numbits) {
  auto quant_param_valueptr = prim->GetAttr("quant_params");
  if (quant_param_valueptr == nullptr) {
    prim->AddAttr("quant_params", std::make_shared<lite::QuantParamHolder>());
  }
  auto quant_param_holder = prim->GetAttr("quant_params")->cast<lite::QuantParamHolderPtr>();
  std::vector<schema::QuantParamT> quants;
  schema::QuantParamT quant_param;
  auto inputMin = prim->GetAttr("input_minq");
  auto inputMax = prim->GetAttr("input_maxq");
  if (inputMin != nullptr && inputMax != nullptr) {
    auto inputMinPtr = inputMin->cast<tensor::TensorPtr>();
    auto inputMaxPtr = inputMax->cast<tensor::TensorPtr>();
    auto *minBuf = static_cast<float *>(inputMinPtr->data_c());
    auto *maxBuf = static_cast<float *>(inputMaxPtr->data_c());
    quant_param.min = *minBuf;
    quant_param.max = *maxBuf;
    auto ret =
      lite::quant::CalQuantizationParams(&quant_param, quant_param.min, quant_param.max, narrow_range, numbits);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Can't calculate quant parameters";
      return ret;
    }
    quants.emplace_back(quant_param);
    quant_param_holder->AddInputQuantParam(quants);
  } else {
    std::vector<schema::QuantParamT> notinited_quant_params(1);
    quant_param_holder->AddInputQuantParam(notinited_quant_params);
  }

  quants.clear();
  auto filterMin = prim->GetAttr("filter_minq");
  auto filterMax = prim->GetAttr("filter_maxq");
  if (filterMin != nullptr && filterMax != nullptr) {
    auto filterMinPtr = filterMin->cast<tensor::TensorPtr>();
    auto filterMaxPtr = filterMax->cast<tensor::TensorPtr>();
    auto *minBuf = static_cast<float *>(filterMinPtr->data_c());
    auto *maxBuf = static_cast<float *>(filterMaxPtr->data_c());
    quant_param.min = FLT_MAX;
    quant_param.max = FLT_MIN;
    for (int i = 0; i < filterMinPtr->ElementsNum(); ++i) {
      quant_param.min = (*(minBuf) < quant_param.min) ? (*minBuf) : quant_param.min;
      quant_param.max = (*(maxBuf) > quant_param.max) ? (*maxBuf) : quant_param.max;
      minBuf++;
      maxBuf++;
    }
    auto ret = lite::quant::CalQuantizationParams(&quant_param, quant_param.min, quant_param.max, true, numbits);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Can't calculate quant parameters";
      return ret;
    }
    quants.emplace_back(quant_param);
    quant_param_holder->AddInputQuantParam(quants);
  } else {
    std::vector<schema::QuantParamT> notinited_quant_params(1);
    quant_param_holder->AddInputQuantParam(notinited_quant_params);
  }
  return lite::RET_OK;
}

int ConvertOutputQuantParam(const PrimitivePtr &prim, bool narrow_range, int32_t numbits) {
  auto quant_param_valueptr = prim->GetAttr("quant_params");
  if (quant_param_valueptr == nullptr) {
    prim->AddAttr("quant_params", std::make_shared<lite::QuantParamHolder>());
  }
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
    quant_param_holder->AddOutputQuantParam(quants);
  } else {
    schema::QuantParamT tmpQuantParam;
    quants.emplace_back(tmpQuantParam);
    quant_param_holder->AddOutputQuantParam(quants);
  }
  return lite::RET_OK;
}

void CheckQuantParams(const PrimitivePtr &prim) {
  auto quant_param_valueptr = prim->GetAttr("quant_params");
  if (quant_param_valueptr == nullptr) {
    prim->AddAttr("quant_params", std::make_shared<lite::QuantParamHolder>());
  }
  auto quant_param_holder = prim->GetAttr("quant_params")->cast<lite::QuantParamHolderPtr>();
  auto input_quant_params = quant_param_holder->input_quant_params();
  bool is_quant = false;
  for (size_t i = 0; i < input_quant_params.size(); ++i) {
    if (!input_quant_params.at(i).empty() && input_quant_params.at(i).at(0).inited) {
      is_quant = true;
      break;
    }
  }
  auto output_quant_params = quant_param_holder->output_quant_params();
  for (size_t i = 0; i < output_quant_params.size(); ++i) {
    if (!output_quant_params.at(i).empty() && output_quant_params.at(i).at(0).inited) {
      is_quant = true;
    }
  }
  if (!is_quant) {
    prim->EraseAttr("quant_params");
  }
}

int ConvertQuantParam(const PrimitivePtr &prim, const std::vector<AnfNodePtr> &inputs) {
  auto quant_param_holder = std::make_shared<lite::QuantParamHolder>();
  prim->AddAttr("quant_params", quant_param_holder);
  auto narrow_range = prim->GetAttr("narrow_range");
  bool narrow_range_param = false;
  if (narrow_range != nullptr) {
    if (utils::isa<tensor::TensorPtr>(narrow_range)) {
      auto narrow_range_tensor = narrow_range->cast<tensor::TensorPtr>();
      narrow_range_param = *reinterpret_cast<bool *>(narrow_range_tensor->data_c());
    } else if (utils::isa<ImmTraits<bool>::type>(narrow_range)) {
      narrow_range_param = GetValue<bool>(narrow_range);
    } else {
      MS_LOG(ERROR) << "valueptr is invalid.";
      return lite::RET_ERROR;
    }
  }
  auto num_bits = prim->GetAttr("num_bits");
  int32_t num_bits_param = 8;
  if (num_bits != nullptr) {
    if (utils::isa<tensor::TensorPtr>(num_bits)) {
      auto num_bits_tensor = num_bits->cast<tensor::TensorPtr>();
      num_bits_param = *reinterpret_cast<int64_t *>(num_bits_tensor->data_c());
    } else if (utils::isa<ImmTraits<int64_t>::type>(num_bits)) {
      num_bits_param = GetValue<int64_t>(num_bits);
    } else {
      MS_LOG(ERROR) << "valueptr is invalid.";
      return lite::RET_ERROR;
    }
  }
  auto status = ConvertInputQuantParam(prim, narrow_range_param, num_bits_param);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "compute int quant param failed.";
    return status;
  }
  FillDefaultInputQuantParamIfNeed(prim, inputs.size());
  status = ConvertOutputQuantParam(prim, narrow_range_param, num_bits_param);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "compute output quant param failed.";
    return status;
  }
  CheckQuantParams(prim);
  return lite::RET_OK;
}
}  // namespace

int MindirAdjustPass::ValueNodeInt64Convert(AnfNodePtr anf_node) {
  if (!utils::isa<ValueNodePtr>(anf_node)) {
    return lite::RET_NO_CHANGE;
  }
  auto valueNode = anf_node->cast<ValueNodePtr>();
  if (valueNode->abstract() == nullptr) {
    return lite::RET_NO_CHANGE;
  }
  auto abstractTensor = utils::cast<abstract::AbstractTensorPtr>(valueNode->abstract());
  if (abstractTensor == nullptr) {
    return lite::RET_NO_CHANGE;
  }
  auto value = abstractTensor->GetValueTrack();
  if (value != nullptr && value->isa<tensor::Tensor>()) {
    if (abstractTensor->element() == nullptr) {
      MS_LOG(ERROR) << "abstractTensor->element() is nullptr.";
      return RET_ERROR;
    }
    auto typePtr = abstractTensor->element()->GetTypeTrack();
    if (typePtr->type_id() == kNumberTypeInt64) {
      auto shape_vector = utils::cast<abstract::ShapePtr>(abstractTensor->BuildShape())->shape();
      auto dest_tensor_info = std::make_shared<tensor::Tensor>(kNumberTypeInt32, shape_vector);
      auto *dest_data_buf = reinterpret_cast<int32_t *>(dest_tensor_info->data_c());
      auto src_tensor_info = value->cast<tensor::TensorPtr>();
      auto *src_data_buf = reinterpret_cast<int64_t *>(src_tensor_info->data_c());
      MS_ASSERT(dest_tensor_info->ElementsNum() == src_tensor_info->ElementsNum());
      for (int i = 0; i < dest_tensor_info->ElementsNum(); i++) {
        dest_data_buf[i] = src_data_buf[i];
      }
      abstractTensor->set_value(dest_tensor_info);
      abstractTensor->set_type(TypeIdToType(kNumberTypeInt32));
      abstractTensor->element()->set_type(TypeIdToType(kNumberTypeInt32));
      valueNode->set_value(dest_tensor_info);
    }
  }
  return lite::RET_NO_CHANGE;
}

int MindirAdjustPass::ParameterNodeConvert(AnfNodePtr anf_node) {
  if (!utils::isa<ParameterPtr>(anf_node)) {
    MS_LOG(INFO) << "only parameter node need to convert tensor.";
    return lite::RET_NO_CHANGE;
  }
  auto param_node = anf_node->cast<ParameterPtr>();
  if (!param_node->has_default()) {
    MS_LOG(INFO) << "this is graph input, don't need to convert.";
    return lite::RET_NO_CHANGE;
  }
  if (utils::isa<ParamValueLitePtr>(param_node->default_param())) {
    MS_LOG(INFO) << "the tensor has been a paramvalueLite.";
    return lite::RET_NO_CHANGE;
  }
  ParamValueLitePtr param_value = std::make_shared<ParamValueLite>();
  if (param_value == nullptr) {
    MS_LOG(ERROR) << "fail to new a ParamValueLite.";
    return lite::RET_ERROR;
  }
  param_node->set_name(param_node->debug_info()->name());
  auto tensor_info = param_node->default_param()->cast<tensor::TensorPtr>();
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "the node is not a tensor::TensorPtr.";
    return lite::RET_ERROR;
  }
  param_value->set_tensor_size(tensor_info->Size());
  param_value->set_tensor_type(tensor_info->data_type());
  auto tensor_shape = tensor_info->shape();
  std::vector<int> shape;
  std::transform(tensor_shape.begin(), tensor_shape.end(), std::back_inserter(shape),
                 [](int64_t value) { return static_cast<int>(value); });
  param_value->set_tensor_shape(shape);
  auto *tensor = new (std::nothrow) lite::Tensor(tensor_info->data_type(), shape);
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "new a lite::tensor failed, get a nullptr.";
    return lite::RET_MEMORY_FAILED;
  }
  auto *tensor_data_buf = tensor->MutableData();
  if (tensor_data_buf == nullptr) {
    MS_LOG(ERROR) << "malloc tensor data failed.";
    delete tensor;
    return lite::RET_MEMORY_FAILED;
  }
  if (memcpy_s(tensor_data_buf, tensor_info->Size(), tensor_info->data_c(), tensor_info->Size()) != EOK) {
    MS_LOG(ERROR) << "memcpy_s error.";
    delete tensor;
    return lite::RET_MEMORY_FAILED;
  }
  tensor->set_data(nullptr);
  param_value->set_tensor_addr(tensor_data_buf);
  param_node->set_default_param(param_value);
  delete tensor;
  return lite::RET_OK;
}

int MindirAdjustPass::ComputeQuantParams(std::shared_ptr<AnfNode> anf_node) {
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
  if (ConvertQuantParam(primitive, inputs) != lite::RET_OK) {
    MS_LOG(ERROR) << "compute quant param failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

bool MindirAdjustPass::Run(const FuncGraphPtr &graph) {
  if (this->fmk_type_ != lite::converter::FmkType_MS) {
    MS_LOG(INFO) << "The framework type of model should be mindir.";
    return lite::RET_OK;
  }
  MS_ASSERT(graph != nullptr);
  auto node_list = TopoSort(graph->get_return());
  int status = lite::RET_OK;
  bool success_flag = true;
  for (auto &node : node_list) {
    if (utils::isa<ParameterPtr>(node)) {
      status = ParameterNodeConvert(node);
    } else if (utils::isa<CNodePtr>(node)) {
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
  return true;
}
}  // namespace opt
}  // namespace mindspore
