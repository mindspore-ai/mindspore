/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/tbe/tbe_json/fusion_tbe_json_creator.h"
#include <algorithm>
#include <string>
#include <vector>
#include "mindspore/core/ops/core_ops.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/device/ascend/kernel/tbe/tbe_adapter.h"
#include "plugin/device/ascend/kernel/tbe/tbe_convert_utils.h"
#include "plugin/device/ascend/kernel/tbe/tbe_dynamic_shape_util.h"
#include "plugin/device/ascend/kernel/tbe/tbe_utils.h"
#include "runtime/dev.h"
#include "include/common/utils/json_operation_utils.h"
#include "plugin/device/ascend/kernel/tbe/tbe_json/tbe_json_utils.h"

namespace mindspore::kernel {
using mindspore::kernel::tbe::TbeAdapter;
bool FusionBuildTbeJsonCreator::GenJson(const FusionScopeInfo &fusion_scope_info, nlohmann::json *fusion_json) {
  MS_EXCEPTION_IF_NULL(fusion_json);
  MS_LOG(DEBUG) << "Start Generate Fusion Json, Fusion Node: " << fusion_scope_info.full_name;
  nlohmann::json soc_info_json = kernel::tbe::TbeUtils::GenSocInfo();
  soc_info_json[kJCoreType] = GetCoreType(fusion_scope_info.compute_nodes.front());
  soc_info_json[kJOpDebugConfig] = tbe::TbeUtils::GetOpDebugConfig();
  (*fusion_json)[kJSocInfo] = soc_info_json;

  std::vector<nlohmann::json> op_list_json;
  if (!GenOpListJson(fusion_scope_info, &op_list_json)) {
    MS_LOG(WARNING) << "Fusion Error: generate fusion json failed.";
    return false;
  }
  (*fusion_json)[kJOpList] = op_list_json;
  GenFusionOpName(fusion_json, kJFusionKernelNamePrefix);
  AddOpNameForComputeNode(fusion_json);
  (*fusion_json)[kJL1Size] = -1;
  (*fusion_json)[kJGraphName] = "";
  (*fusion_json)[kJScopeID] = fusion_scope_info.scope_id;
  (*fusion_json)[kJFullName] = fusion_scope_info.full_name;
  MS_LOG(DEBUG) << "Json name is : " << GetJsonName() << ", fusion json:" << fusion_json->dump();
  return true;
}

bool FusionBuildTbeJsonCreator::GenOpListJson(const FusionScopeInfo &fusion_scope_info,
                                              std::vector<nlohmann::json> *fusion_json) {
  MS_EXCEPTION_IF_NULL(fusion_json);
  MS_LOG(DEBUG) << "Start";
  if (!CheckInput(fusion_scope_info)) {
    for (const auto &cnode : fusion_scope_info.compute_nodes) {
      MS_LOG(WARNING) << "Fusion Error: check input failed, scope id: " << fusion_scope_info.scope_id
                      << ", compute node: " << cnode->fullname_with_scope();
    }
    return false;
  }

  optional_index_ = 0;
  auto compute_nodes = fusion_scope_info.compute_nodes;
  std::vector<nlohmann::json> compute_list;
  for (const auto &compute_node : compute_nodes) {
    nlohmann::json compute_json;
    if (!GenComputeJson(compute_node, &compute_json)) {
      MS_LOG(WARNING) << "Fusion Error: gen fusion compute json failed. node full name: "
                      << compute_node->fullname_with_scope();
      return false;
    }
    compute_json[kJOriName] = {fusion_scope_info.full_name};
    compute_list.push_back(compute_json);
  }

  // FusionDataType fusion_data_type: speceial process json desc output shape [kFusionAddN, kFusionReLUGradV2]
  ANodeFusionDataTypeMap spec_data_input;
  if (!TbeAdapter::GetSpecDataInput(fusion_scope_info, &spec_data_input)) {
    return false;
  }
  GenDataJson(compute_nodes, compute_list, fusion_json, spec_data_input);
  (*fusion_json).insert((*fusion_json).end(), compute_list.cbegin(), compute_list.cend());
  MS_LOG(DEBUG) << "End";
  return true;
}

bool FusionBuildTbeJsonCreator::CheckInput(const FusionScopeInfo &fusion_scope_info) const {
  MS_LOG(DEBUG) << "Start";
  auto input_nodes = fusion_scope_info.input_nodes;
  auto compute_nodes = fusion_scope_info.compute_nodes;
  size_t input_size = 0;
  for (const auto &node : compute_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      MS_LOG(WARNING) << "Fusion Error: fusion compute node must be cnode, but the node is " << node->DebugString();
      return false;
    }
    for (size_t i = 1; i < cnode->inputs().size(); ++i) {
      auto input = cnode->input(i);
      auto find_iter = std::find(input_nodes.begin(), input_nodes.end(), input);
      if (find_iter != input_nodes.end()) {
        input_size++;
      }
    }
  }
  if (input_nodes.size() != input_size) {
    MS_LOG(WARNING) << "Fusion Error: compute node input size: [ " << input_size
                    << " ] is not equal to input nodes num: [ " << input_nodes.size() << " ].";
    return false;
  }
  MS_LOG(DEBUG) << "End";
  return true;
}

void FusionBuildTbeJsonCreator::GenDataJson(const std::vector<AnfNodePtr> &compute_nodes,
                                            const std::vector<nlohmann::json> &compute_json,
                                            std::vector<nlohmann::json> *op_list_json,
                                            const ANodeFusionDataTypeMap &spec_data_input) const {
  MS_EXCEPTION_IF_NULL(op_list_json);
  MS_LOG(DEBUG) << "Start.";
  std::vector<std::string> compute_nodes_fullname;
  std::transform(compute_nodes.begin(), compute_nodes.end(), back_inserter(compute_nodes_fullname),
                 [](const AnfNodePtr &node) { return node->fullname_with_scope(); });
  for (size_t i = 0; i < compute_nodes.size(); i++) {
    auto inputs_desc = GetJsonValue<std::vector<nlohmann::json>>(compute_json.at(i), kJInputDesc);
    for (const auto &input_desc : inputs_desc) {
      if (std::find(compute_nodes_fullname.begin(), compute_nodes_fullname.end(),
                    GetJsonValue<std::string>(input_desc, kJName)) != compute_nodes_fullname.end()) {
        continue;
      }
      nlohmann::json data_json;
      nlohmann::json output_desc = input_desc;
      std::vector<nlohmann::json> output_desc_list;
      if (input_desc.find(kJOriShape) != input_desc.end()) {
        auto input_node = GetInputCNode(compute_nodes[i], output_desc);
        TbeAdapter::FusionDescJsonPass(input_node, &output_desc, spec_data_input);
      }
      output_desc_list.push_back(output_desc);
      data_json[kJName] = GetJsonValue<std::string>(input_desc, kJName);
      data_json[kJType] = kJData;
      data_json[kJOutputDesc] = output_desc_list;
      (*op_list_json).push_back(data_json);
    }
  }
  MS_LOG(DEBUG) << "End.";
}
AnfNodePtr FusionBuildTbeJsonCreator::GetInputCNode(const AnfNodePtr &node, const nlohmann::json &input_desc) const {
  auto input_name = GetJsonValue<std::string>(input_desc, kJName);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  for (size_t i = 1; i < cnode->inputs().size(); i++) {
    auto kernel_idx = common::AnfAlgo::VisitKernel(cnode->input(i), 0);
    auto full_name = kernel_idx.first->fullname_with_scope();
    std::string desc_name = kernel_idx.second > 0 ? (full_name + "_" + std::to_string(kernel_idx.second)) : full_name;
    if (input_name == desc_name) {
      return cnode->input(i);
    }
  }
  MS_LOG(EXCEPTION) << "Can not find node:[" << node->fullname_with_scope() << "]'s input [" << input_name << "]";
}

bool FusionBuildTbeJsonCreator::GenInputsJson(const AnfNodePtr &anf_node, nlohmann::json *compute_json) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(compute_json);
  std::vector<nlohmann::json> input_desc_list_tmp = {};
  auto op_name = common::AnfAlgo::GetCNodeName(anf_node);
  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  bool is_dynamic_input = common::AnfAlgo::HasNodeAttr(kAttrDynInputSizes, cnode);
  if (is_dynamic_input) {
    MS_LOG(INFO) << op_name << " has dynamic input.";
    if (!CheckDynamicInput(cnode)) {
      return false;
    }
  }
  size_t input_index = 0;
  for (size_t i = 1; i < cnode->inputs().size(); ++i) {
    auto input = cnode->input(i);
    if (HasAbstractMonad(input)) {
      continue;
    }
    auto kernel_idx = common::AnfAlgo::VisitKernel(input, 0);
    nlohmann::json input_desc;
    GenDescJson(kernel_idx.first, kernel_idx.second, kernel_idx.second, &input_desc);
    GenInputConstValue(anf_node, i - 1, &input_desc);
    if (is_dynamic_input) {
      input_desc[kJDynIndex] = (i - 1);
    }
    (void)input_desc_list_tmp.emplace_back(input_desc);
    input_index++;
  }
  std::vector<size_t> inputs_tensor_num;
  auto op_info = tbe::TbeDynamicShapeUtil::FindOp(op_name, anf_node);
  if (!TbeJsonUtils::GetInputsRealNum(anf_node, op_info->inputs_ptr(), &inputs_tensor_num)) {
    return false;
  }
  size_t need_input_num = std::accumulate(inputs_tensor_num.begin(), inputs_tensor_num.end(), static_cast<size_t>(0));

  for (size_t i = input_index; i < need_input_num; ++i) {
    nlohmann::json optional_input_desc;
    optional_input_desc[kJName] = std::string(kJOptional) + std::to_string(optional_index_);
    optional_input_desc[kJShape] = kJNull;
    optional_input_desc[kJDataType] = 0;
    optional_index_++;
    (void)input_desc_list_tmp.emplace_back(optional_input_desc);
  }
  std::vector<nlohmann::json> input_desc_list;
  TbeAdapter::InputOrderPass<nlohmann::json>(cnode, input_desc_list_tmp, &input_desc_list);
  (*compute_json)[kJInputDesc] = input_desc_list;
  return true;
}

bool FusionBuildTbeJsonCreator::CheckDynamicInput(const CNodePtr &cnode) const {
  MS_EXCEPTION_IF_NULL(cnode);
  if (!common::AnfAlgo::HasNodeAttr(kAttrDynInputSizes, cnode)) {
    MS_LOG(WARNING) << "Fusion Error: cnode [ " << common::AnfAlgo::GetCNodeName(cnode)
                    << "] has not attr dyn_input_sizes.";
    return false;
  }
  // for dynamic input number, dyn_input_sizes has the info of dynamic input num for each input.
  auto dyn_input_sizes = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(cnode, kAttrDynInputSizes);
  if (dyn_input_sizes.size() != 1) {
    MS_LOG(WARNING) << "Fusion Error: fusion build not support dynamic input size > 1";
    return false;
  }
  auto real_input_size = cnode->inputs().size() - 1;
  if (LongToSize(dyn_input_sizes[0]) != real_input_size) {
    MS_LOG(WARNING) << "Fusion Error: dyn_input_size" << dyn_input_sizes[0] << "not equal real_input_size"
                    << real_input_size;
    return false;
  }
  return true;
}

bool FusionBuildTbeJsonCreator::GenOutputsJson(const AnfNodePtr &anf_node, nlohmann::json *compute_json) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(compute_json);
  auto output_size = AnfAlgo::GetOutputTensorNum(anf_node);
  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  std::vector<nlohmann::json> output_desc_list;
  if (common::AnfAlgo::HasNodeAttr(kAttrOutputUsedNum, cnode)) {
    auto output_used_nums = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(anf_node, kAttrOutputUsedNum);
    if (output_used_nums.size() != output_size) {
      MS_LOG(WARNING) << "Fusion Error: [" << common::AnfAlgo::GetCNodeName(anf_node) << " ]'s output tensor num("
                      << output_size << ")"
                      << " is not match output used num(" << output_used_nums.size() << ")";
      return false;
    }
    auto desc_output_index = GetDescOutputIndex(output_used_nums);
    for (size_t i = 0; i < output_size; ++i) {
      MS_LOG(DEBUG) << "Fusion index: " << i << ", desc_output_index: " << desc_output_index[i];
      nlohmann::json output_desc;
      GenDescJson(anf_node, i, desc_output_index[i], &output_desc);
      output_desc_list.emplace_back(output_desc);
    }
    for (size_t j = output_size; j < desc_output_index.size(); ++j) {
      MS_LOG(DEBUG) << "Fusion index: " << j << ", desc_output_index: " << desc_output_index[j];
      nlohmann::json output_desc;
      GenReusedOutputDesc(anf_node, j, desc_output_index[j], &output_desc, output_size);
      output_desc_list.emplace_back(output_desc);
    }
  } else {
    for (size_t i = 0; i < output_size; ++i) {
      nlohmann::json output_desc;
      GenDescJson(anf_node, i, i, &output_desc);
      output_desc_list.emplace_back(output_desc);
    }
  }
  (*compute_json)[kJOutputDesc] = output_desc_list;
  return true;
}

void FusionBuildTbeJsonCreator::GenReusedOutputDesc(const AnfNodePtr &anf_node, size_t index, size_t output_index,
                                                    nlohmann::json *output_desc, size_t out_size) const {
  GenDesJsonCommon(output_desc);
  std::string output_desc_name = anf_node->fullname_with_scope() + "_" + std::to_string(index);
  (*output_desc)[kJName] = output_desc_name;
  (*output_desc)[kJOutputIndex] = output_index;
  std::vector<size_t> shape;
  (*output_desc)[kJShape] = shape;
  if (out_size < 1) {
    MS_LOG(EXCEPTION) << "Invalid output size: " << out_size;
  }
  (*output_desc)[kJDataType] = tbe::TypeIdToString(AnfAlgo::GetOutputDeviceDataType(anf_node, out_size - 1));
}

std::vector<size_t> FusionBuildTbeJsonCreator::GetDescOutputIndex(const std::vector<int64_t> &output_used_nums) const {
  std::vector<size_t> desc_output_index = {};
  for (size_t idx = 0; idx < output_used_nums.size(); ++idx) {
    auto index = idx;
    (void)desc_output_index.emplace_back(index);
    if (output_used_nums[index] > 1) {
      (void)desc_output_index.emplace_back(index);
    }
  }
  return desc_output_index;
}

bool FusionBuildTbeJsonCreator::AttrsJsonPostProcessing(const AnfNodePtr &, const OpInfoPtr &, nlohmann::json *) {
  return true;
}

void FusionBuildTbeJsonCreator::GenOtherJson(const AnfNodePtr &anf_node, nlohmann::json *compute_json) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(compute_json);
  (*compute_json)[kJUnknowShape] = tbe::TbeDynamicShapeUtil::GetDynamicShapeAttr(anf_node);
}
}  // namespace mindspore::kernel
