/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/tbe/tbe_json/single_tbe_json_creator.h"
#include <algorithm>
#include <string>
#include "frontend/parallel/ops_info/ops_utils.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/device/ascend/kernel/tbe/tbe_adapter.h"
#include "plugin/device/ascend/kernel/tbe/tbe_convert_utils.h"
#include "plugin/device/ascend/kernel/tbe/tbe_dynamic_shape_util.h"
#include "plugin/device/ascend/kernel/tbe/tbe_utils.h"
#include "runtime/dev.h"
#include "utils/ms_utils.h"
#include "include/common/utils/json_operation_utils.h"
#include "plugin/device/ascend/kernel/tbe/tbe_json/tbe_json_utils.h"
#include "runtime/device/ms_device_shape_transfer.h"

namespace mindspore::kernel {
using mindspore::kernel::tbe::TbeAdapter;
bool SingleTbeJsonCreator::GenJson(const AnfNodePtr &anf_node, nlohmann::json *kernel_json) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(kernel_json);
  auto op_name = common::AnfAlgo::GetCNodeName(anf_node);
  MS_LOG(DEBUG) << "Start, node [ " << op_name << " ].";
  nlohmann::json soc_info_json = kernel::tbe::TbeUtils::GenSocInfo();
  std::vector<nlohmann::json> op_list;
  if (!GenOpListJson(anf_node, &op_list)) {
    MS_LOG(ERROR) << "Anf Node [" << op_name << "] generate op_list json failed";
    return false;
  }
  auto core_type = GetCoreType(anf_node);
  soc_info_json[kJCoreType] = core_type;
  soc_info_json[kJOpDebugConfig] = tbe::TbeUtils::GetOpDebugConfig();
  (*kernel_json)[kJSocInfo] = soc_info_json;
  (*kernel_json)[kJOpList] = op_list;
  GenFusionOpName(kernel_json);
  AddOpNameForComputeNode(kernel_json);
  (*kernel_json)[kJFullName] = anf_node->fullname_with_scope();
  (*kernel_json)[kJGraphName] = "";
  (*kernel_json)[kJScopeID] = -1;
  (*kernel_json)[kJL1Size] = -1;
  MS_LOG(DEBUG) << "Json info name is : " << GetJsonName() << ", kernel json:" << kernel_json->dump();
  return true;
}

void NpuClearV2PostProcessing(const AnfNodePtr &anf_node, std::vector<nlohmann::json> *op_list_json) {
  if (op_list_json->size() != 2) {
    MS_LOG(ERROR) << "Op list json's size is not equal to 2, abort post processing.";
  }

  auto compute_json = (*op_list_json)[1];
  std::vector<nlohmann::json> empty_vector_json;
  compute_json[kJInputDesc] = empty_vector_json;
  compute_json[kJOutputDataDesc] = empty_vector_json;
  compute_json[kJOutputDesc] = empty_vector_json;
  op_list_json->clear();
  (*op_list_json).emplace_back(compute_json);
  MS_LOG(DEBUG) << "Op list json after post processing:" << compute_json.dump();
}

void NpuGetV2PostProcessing(const AnfNodePtr &anf_node, std::vector<nlohmann::json> *op_list_json) {
  if (op_list_json->size() != 2) {
    MS_LOG(ERROR) << "Op list json's size is not equal to 2, abort post processing.";
  }

  auto compute_json = (*op_list_json)[1];
  std::vector<nlohmann::json> empty_vector_json;
  compute_json[kJInputDesc] = empty_vector_json;
  op_list_json->clear();
  (*op_list_json).emplace_back(compute_json);
  MS_LOG(DEBUG) << "Op list json after post processing:" << compute_json.dump();
}

void SingleTbeJsonCreator::OpListPostProcessing(const AnfNodePtr &anf_node, std::vector<nlohmann::json> *op_list_json) {
  auto kernel_name = common::AnfAlgo::GetCNodeName(anf_node);
  if (kernel_name == kNPUClearFloatStatusV2OpName) {
    NpuClearV2PostProcessing(anf_node, op_list_json);
  } else if (kernel_name == kNPUGetFloatStatusV2OpName) {
    NpuGetV2PostProcessing(anf_node, op_list_json);
  }
}

bool SingleTbeJsonCreator::GenOpListJson(const AnfNodePtr &anf_node, std::vector<nlohmann::json> *op_list_json) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(op_list_json);
  MS_LOG(DEBUG) << "Start.";
  nlohmann::json compute_json;
  if (!GenComputeJson(anf_node, &compute_json)) {
    MS_LOG(ERROR) << "Anf Node [" << common::AnfAlgo::GetCNodeName(anf_node) << "] generate compute json failed";
    return false;
  }
  GenDataJson(anf_node, compute_json, op_list_json);
  (*op_list_json).push_back(compute_json);
  OpListPostProcessing(anf_node, op_list_json);
  MS_LOG(DEBUG) << "End.";
  return true;
}

void SingleTbeJsonCreator::GenDataJson(const AnfNodePtr &anf_node, const nlohmann::json &compute_json,
                                       std::vector<nlohmann::json> *op_list_json) const {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(op_list_json);
  MS_LOG(DEBUG) << "Start";
  auto op_name = common::AnfAlgo::GetCNodeName(anf_node);
  auto op_info_ptr = mindspore::kernel::tbe::TbeDynamicShapeUtil::FindOp(op_name, anf_node);
  auto inputs_ptr = op_info_ptr->inputs_ptr();
  auto inputs_json = GetJsonValue<std::vector<nlohmann::json>>(compute_json, kJInputDesc);
  for (size_t i = 0; i < inputs_ptr.size(); i++) {
    auto input_json = inputs_json.at(i);
    auto input_ptr = inputs_ptr[i];
    MS_EXCEPTION_IF_NULL(input_ptr);
    nlohmann::json data_json;
    std::vector<nlohmann::json> output_desc;
    if (input_json.is_array()) {
      data_json[kJName] = input_ptr->name() + "_dynamic";
      auto tmp_inputs = input_json.get<std::vector<nlohmann::json>>();
      std::copy(tmp_inputs.begin(), tmp_inputs.end(), std::back_inserter(output_desc));
    } else {
      data_json[kJName] = GetJsonValue<std::string>(input_json, kJName);
      output_desc.push_back(input_json);
    }
    data_json[kJOutputDesc] = output_desc;
    data_json[kJType] = kJData;
    (*op_list_json).push_back(data_json);
  }
  MS_LOG(DEBUG) << "End";
}

bool SingleTbeJsonCreator::GenInputsJson(const AnfNodePtr &anf_node, nlohmann::json *compute_json) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(compute_json);
  MS_LOG(DEBUG) << "Start.";
  auto op_name = common::AnfAlgo::GetCNodeName(anf_node);
  auto op_info_ptr = mindspore::kernel::tbe::TbeDynamicShapeUtil::FindOp(op_name, anf_node);
  MS_EXCEPTION_IF_NULL(op_info_ptr);
  std::vector<OpIOInfoPtr> inputs_ptr = op_info_ptr->inputs_ptr();
  std::vector<nlohmann::json> inputs_json;
  if (inputs_ptr.empty()) {
    MS_LOG(DEBUG) << op_name << " registration info has no input info.";
    (*compute_json)[kJInputDesc] = inputs_json;
    return true;
  }
  std::vector<size_t> inputs_tensor_num;
  if (!TbeJsonUtils::GetInputsRealNum(anf_node, inputs_ptr, &inputs_tensor_num)) {
    return false;
  }

  std::vector<nlohmann::json> inputs_desc;
  size_t real_input_num = common::AnfAlgo::GetInputTensorNum(anf_node);
  for (size_t i = 0; i < real_input_num; i++) {
    nlohmann::json input_desc;
    GenInputDescJson(anf_node, i, &input_desc);
    inputs_desc.emplace_back(input_desc);
  }

  size_t need_input_num = std::accumulate(inputs_tensor_num.begin(), inputs_tensor_num.end(), static_cast<size_t>(0));
  // gen optional desc
  for (size_t i = common::AnfAlgo::GetInputTensorNum(anf_node); i < need_input_num; i++) {
    nlohmann::json input_desc;
    input_desc[kJValid] = false;
    input_desc[kJShape] = kJNull;
    input_desc[kJDataType] = 0;
    inputs_desc.emplace_back(input_desc);
  }

  std::vector<nlohmann::json> inputs_list;
  if (!AssignInputsJson(anf_node, inputs_desc, inputs_tensor_num, inputs_ptr, &inputs_list)) {
    return false;
  }

  TbeAdapter::InputOrderPass<nlohmann::json>(anf_node, inputs_list, &inputs_json);
  (*compute_json)[kJInputDesc] = inputs_json;
  MS_LOG(DEBUG) << "End.";
  return true;
}

void SingleTbeJsonCreator::GenInputDescJson(const AnfNodePtr &anf_node, size_t real_input_index,
                                            nlohmann::json *input_desc) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(input_desc);
  GenDesJsonCommon(input_desc);
  auto shape = TbeJsonUtils::GetInputDeviceShapeForTbeBuild(anf_node, real_input_index);
  auto ori_shape = TbeJsonUtils::GetInputOriShapeForTbeBuild(anf_node, real_input_index);
  if (shape.empty()) {
    shape.emplace_back(1);
  }
  if (ori_shape.empty()) {
    ori_shape.emplace_back(1);
  }

  std::string def_format = TbeJsonUtils::IsNeedChangeDefaultFormat(anf_node) ? kOpFormat_NCDHW : kOpFormat_NCHW;
  def_format = AnfAlgo::HasOriginFormat(anf_node) ? AnfAlgo::GetOriginFormat(anf_node) : def_format;
  auto format = AnfAlgo::GetInputFormat(anf_node, real_input_index);
  format = TbeAdapter::FormatPass(format, ori_shape.size());
  format = (def_format == kOpFormat_NCDHW && !IsOneOf3DFormat(format)) ? kOpFormat_NCDHW : format;
  auto d_type = AnfAlgo::GetInputDeviceDataType(anf_node, real_input_index);
  (*input_desc)[kJDtype] = tbe::TypeIdToString(d_type);
  (*input_desc)[kJDataType] = GetJsonValue<std::string>(*input_desc, kJDtype);
  (*input_desc)[kJOriShape] = ori_shape;
  (*input_desc)[kJOriFormat] = def_format;
  if (common::AnfAlgo::GetCNodeName(anf_node) == kMaxPool3DGradGradDOpName) {
    (*input_desc)[kJOriFormat] = kOpFormat_NDHWC;
  }
  if (format == kOpFormat_NC1HWC0) {
    auto infer_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(anf_node, real_input_index);
    infer_shape = trans::PaddingShape(infer_shape, format, AnfAlgo::GetInputReshapeType(anf_node, real_input_index));
    (*input_desc)[kJCValue] = infer_shape[1];
  }
  (*input_desc)[kJShape] = shape;
  (*input_desc)[kJFormat] = format;
  (*input_desc)[kJValid] = true;
  (*input_desc)[kJRange] = tbe::TbeDynamicShapeUtil::GetInputDynamicRange(anf_node, real_input_index, format, d_type);
  GenInputConstValue(anf_node, real_input_index, input_desc);
}

void SingleTbeJsonCreator::GenOutputDescJson(const AnfNodePtr &anf_node, size_t node_out_idx,
                                             nlohmann::json *output_desc) {
  MS_EXCEPTION_IF_NULL(anf_node);
  GenDescJson(anf_node, node_out_idx, node_out_idx, output_desc);
  output_desc->erase(kJOutputIndex);
  auto type_str = GetJsonValue<std::string>(*output_desc, kJDtype);
  auto d_type = tbe::DtypeToTypeId(type_str);
  (*output_desc)[kJValid] = true;
  (*output_desc)[kJRange] =
    tbe::TbeDynamicShapeUtil::GetOutputDynamicRange(anf_node, node_out_idx, (*output_desc)[kJFormat], d_type);
}

bool SingleTbeJsonCreator::AssignInputsJson(const AnfNodePtr &anf_node, const std::vector<nlohmann::json> &inputs_desc,
                                            const std::vector<size_t> &inputs_tensor_num,
                                            const std::vector<OpIOInfoPtr> &inputs_ptr,
                                            std::vector<nlohmann::json> *inputs_json) const {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(inputs_json);
  MS_LOG(DEBUG) << "Start.";
  size_t inputs_desc_index = 0;
  for (size_t i = 0; i < inputs_tensor_num.size(); i++) {
    auto input_ptr = inputs_ptr[i];
    MS_EXCEPTION_IF_NULL(input_ptr);
    auto param_type = input_ptr->param_type();
    if (param_type == kJParamOptional) {
      nlohmann::json current_input_desc;
      // DynamicRnn and DynamicGRUV2 exist placeholder input
      current_input_desc = inputs_desc.at(inputs_desc_index);
      if (TbeAdapter::IsPlaceHolderInput(anf_node, input_ptr)) {
        current_input_desc[kJValid] = false;
      }
      current_input_desc[kJName] = input_ptr->name() + "_optional_";
      current_input_desc[kJParamType] = input_ptr->param_type();
      (*inputs_json).emplace_back(current_input_desc);
      inputs_desc_index++;
    } else if (param_type == kJParamDynamic) {
      std::vector<nlohmann::json> dynamic_inputs_desc;
      for (size_t j = 0; j < inputs_tensor_num[i]; j++) {
        auto current_input_desc = inputs_desc.at(inputs_desc_index);
        current_input_desc[kJName] = input_ptr->name() + "_dynamic_" + std::to_string(j);
        current_input_desc[kJParamType] = input_ptr->param_type();
        dynamic_inputs_desc.emplace_back(current_input_desc);
        inputs_desc_index++;
      }
      (*inputs_json).emplace_back(dynamic_inputs_desc);
    } else if (param_type == kJParamRequred) {
      auto current_input_desc = inputs_desc.at(inputs_desc_index);
      current_input_desc[kJName] = input_ptr->name() + "_0";
      current_input_desc[kJParamType] = input_ptr->param_type();
      (*inputs_json).emplace_back(current_input_desc);
      inputs_desc_index++;
    } else {
      MS_LOG(ERROR) << "Unsupported input param type:[" << param_type
                    << "], supported list: {optional, dynamic, required}.";
      return false;
    }
  }
  MS_LOG(DEBUG) << "End.";
  return true;
}

bool SingleTbeJsonCreator::GenOutputsJson(const AnfNodePtr &anf_node, nlohmann::json *compute_json) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(compute_json);
  MS_LOG(DEBUG) << "Start.";
  auto op_name = common::AnfAlgo::GetCNodeName(anf_node);
  auto op_info_ptr = mindspore::kernel::tbe::TbeDynamicShapeUtil::FindOp(op_name, anf_node);
  MS_EXCEPTION_IF_NULL(op_info_ptr);
  std::vector<size_t> outputs_tensor_num;
  auto outputs_ptr = op_info_ptr->outputs_ptr();
  if (!TbeJsonUtils::GetOutputsRealNum(anf_node, outputs_ptr, &outputs_tensor_num)) {
    return false;
  }

  size_t sum_outputs_num =
    std::accumulate(outputs_tensor_num.begin(), outputs_tensor_num.end(), static_cast<size_t>(0));
  size_t real_output_num = AnfAlgo::GetOutputElementNum(anf_node);
  std::vector<nlohmann::json> outputs_desc;
  for (size_t i = 0; i < real_output_num; i++) {
    nlohmann::json output_desc;
    GenOutputDescJson(anf_node, i, &output_desc);
    outputs_desc.emplace_back(output_desc);
  }

  for (size_t i = real_output_num; i < sum_outputs_num; i++) {
    nlohmann::json output_desc;
    output_desc[kJValid] = false;
    output_desc[kJShape] = kJNull;
    outputs_desc.emplace_back(output_desc);
  }

  std::vector<nlohmann::json> outputs_json;
  if (!AssignOutputsJson(anf_node, outputs_desc, outputs_tensor_num, outputs_ptr, &outputs_json)) {
    return false;
  }
  (*compute_json)[kJOutputDesc] = outputs_json;
  MS_LOG(DEBUG) << "End.";
  return true;
}

bool SingleTbeJsonCreator::AssignOutputsJson(const AnfNodePtr &anf_node,
                                             const std::vector<nlohmann::json> &outputs_desc,
                                             const std::vector<size_t> &outputs_tensor_num,
                                             const std::vector<OpIOInfoPtr> &outputs_ptr,
                                             std::vector<nlohmann::json> *outputs_json) const {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_LOG(DEBUG) << "Start.";
  size_t outputs_desc_index = 0;
  for (size_t i = 0; i < outputs_tensor_num.size(); i++) {
    auto output_ptr = outputs_ptr[i];
    MS_EXCEPTION_IF_NULL(output_ptr);
    auto param_type = output_ptr->param_type();
    if (param_type == kJParamDynamic) {
      std::vector<nlohmann::json> dynamic_outputs_desc;
      for (size_t j = 0; j < outputs_tensor_num[i]; j++) {
        auto current_input_desc = outputs_desc.at(outputs_desc_index);
        current_input_desc[kJName] = output_ptr->name();
        current_input_desc[kJParamType] = output_ptr->param_type();
        dynamic_outputs_desc.emplace_back(current_input_desc);
        outputs_desc_index++;
      }
      (*outputs_json).emplace_back(dynamic_outputs_desc);
    } else if (param_type == kJParamRequred || param_type == kJParamOptional) {
      auto current_input_desc = outputs_desc.at(outputs_desc_index);
      current_input_desc[kJName] = output_ptr->name();
      current_input_desc[kJParamType] = output_ptr->param_type();
      (*outputs_json).emplace_back(current_input_desc);
      outputs_desc_index++;
    } else {
      MS_LOG(ERROR) << "Unsupported output param type:[" << param_type
                    << "], supported list: {required, dynamic, optional}.";
      return false;
    }
  }
  MS_LOG(DEBUG) << "End.";
  return true;
}

void SingleTbeJsonCreator::GenOtherJson(const AnfNodePtr &anf_node, nlohmann::json *compute_json) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(compute_json);
  MS_LOG(DEBUG) << "Start.";
  (*compute_json)[kJOriName] = {anf_node->fullname_with_scope()};
  (*compute_json)[kJBuildType] = kAccuratelyBuild;
  (*compute_json)[kJMissSupportInfo] = "";
  (*compute_json)[kJMaxKernelID] = kJMaxKernelIDValue;
  (*compute_json)[kJUnknowShape] = tbe::TbeDynamicShapeUtil::GetDynamicShapeAttr(anf_node);
  MS_LOG(DEBUG) << "End.";
}

void SelectTbeJsonCreator::GenDescJson(const AnfNodePtr &anf_node, size_t node_out_idx, size_t desc_output_idx,
                                       nlohmann::json *output_desc) {
  MS_EXCEPTION_IF_NULL(anf_node);
  GenDesJsonCommon(output_desc);
  std::vector<int64_t> shape;
  std::vector<int64_t> ori_shape;
  ori_shape = TbeJsonUtils::GetOutputOriShapeForTbeBuild(anf_node, node_out_idx);
  if (ori_shape.empty()) {
    ori_shape.emplace_back(1);
  }
  shape = ori_shape;

  auto def_format = TbeJsonUtils::IsNeedChangeDefaultFormat(anf_node) ? kOpFormat_NCDHW : kOpFormat_NCHW;
  auto format = def_format;

  (*output_desc)[kJDataType] = tbe::TypeIdToString(common::AnfAlgo::GetOutputInferDataType(anf_node, node_out_idx));
  (*output_desc)[kJDtype] = GetJsonValue<std::string>(*output_desc, kJDataType);
  (*output_desc)[kJFormat] = format;
  (*output_desc)[kJOriFormat] = def_format;
  (*output_desc)[kJOriShape] = ori_shape;
  (*output_desc)[kJShape] = shape;
  (*output_desc)[kJOutputIndex] = desc_output_idx;
}

void SelectTbeJsonCreator::GenInputDescJson(const AnfNodePtr &anf_node, size_t real_input_index,
                                            nlohmann::json *input_desc) {
  MS_EXCEPTION_IF_NULL(anf_node);
  GenDesJsonCommon(input_desc);
  auto ori_shape = TbeJsonUtils::GetInputOriShapeForTbeBuild(anf_node, real_input_index);
  if (ori_shape.empty()) {
    ori_shape.emplace_back(1);
  }
  auto shape = ori_shape;

  auto def_format = TbeJsonUtils::IsNeedChangeDefaultFormat(anf_node) ? kOpFormat_NCDHW : kOpFormat_NCHW;
  auto format = def_format;
  auto d_type = common::AnfAlgo::GetPrevNodeOutputInferDataType(anf_node, real_input_index);
  (*input_desc)[kJDtype] = tbe::TypeIdToString(d_type);
  (*input_desc)[kJDataType] = GetJsonValue<std::string>(*input_desc, kJDtype);
  (*input_desc)[kJOriShape] = ori_shape;
  (*input_desc)[kJOriFormat] = def_format;
  if (common::AnfAlgo::GetCNodeName(anf_node) == kMaxPool3DGradGradDOpName) {
    (*input_desc)[kJOriFormat] = kOpFormat_NDHWC;
  }
  (*input_desc)[kJShape] = shape;
  (*input_desc)[kJFormat] = format;
  (*input_desc)[kJValid] = true;
  auto input_node_with_index = common::AnfAlgo::GetPrevNodeOutput(anf_node, real_input_index);
  MS_EXCEPTION_IF_NULL(input_node_with_index.first);
  if (!input_node_with_index.first->isa<ValueNode>()) {
    (*input_desc)[kJRange] = tbe::TbeDynamicShapeUtil::GetInputDynamicRange(anf_node, real_input_index, format, d_type);
  }
}
bool SelectTbeJsonCreator::AttrsJsonPostProcessing(const AnfNodePtr &anf_node, const OpInfoPtr &op_info_ptr,
                                                   nlohmann::json *attrs_json) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(op_info_ptr);
  MS_EXCEPTION_IF_NULL(attrs_json);
  tbe::TbeAdapter::LayerNormAttrJsonPost(anf_node, attrs_json);
  return true;
}
}  // namespace mindspore::kernel
