/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "runtime/device/ascend/kernel_select_ascend.h"

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "backend/kernel_compiler/kernel_build_info.h"
#include "backend/kernel_compiler/kernel_query.h"
#include "backend/kernel_compiler/oplib/oplib.h"
#include "backend/kernel_compiler/tbe/tbe_dynaminc_shape_util.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "common/trans.h"
#include "debug/anf_ir_dump.h"
#include "frontend/operator/ops.h"
#include "utils/ms_context.h"
#include "utils/ms_utils.h"
#include "utils/trace_base.h"
namespace mindspore {
namespace device {
namespace ascend {
namespace {
const int kWeightUnInitScore = 1;
const int kWeightInitScore = 2;
const int kFeatureMapBaseScore = 10;
constexpr auto kPriChoosenFormat = "pri_format";
enum MatchCountPriority : int {
  MATCH_COUNT_PRIORITY_BEGIN = 0,
  MATCH_DTYPE_COUNT = MATCH_COUNT_PRIORITY_BEGIN,
  MATCH_FORMAT_COUNT,
  MATCH_SPECIAL_FORMAT_COUNT,
  MATCH_DEFAULT_FORMAT_COUNT,
  MATCH_OUTPUT_DTYPE_COUNT,
  MATCH_COUNT_PRIORITY_END
};
const std::map<std::string, std::vector<std::string>> kNextOpFormatList = {
  {prim::kPrimConv2D->name(), {kOpFormat_NC1HWC0, kOpFormat_FRAC_Z}}};

bool MatchInferOutputDataType(const CNodePtr &cnode, const kernel::KernelBuildInfo &kernel_build_info) {
  MS_EXCEPTION_IF_NULL(cnode);
  // Check input data type
  for (size_t input_index = 0; input_index < kernel_build_info.GetInputNum(); ++input_index) {
    TypeId input_origin_type = AnfAlgo::GetPrevNodeOutputInferDataType(cnode, input_index);
    if (kernel_build_info.GetInputDeviceType(input_index) != input_origin_type) {
      return false;
    }
  }
  // Check output data type
  for (size_t output_index = 0; output_index < kernel_build_info.GetOutputNum(); ++output_index) {
    if (kernel_build_info.GetOutputDeviceType(output_index) != AnfAlgo::GetOutputInferDataType(cnode, output_index)) {
      return false;
    }
  }
  return true;
}

string GetPriorityMatchFormat(const CNodePtr &cnode) {
  string priority_matched_format = kOpFormat_NC1HWC0;
  bool is_init = false;
  bool need_change_nd = false;
  bool is_5d_input = false;
  size_t input_num = AnfAlgo::GetInputTensorNum(cnode);
  for (size_t index = 0; index < input_num; ++index) {
    auto pre_output_format = AnfAlgo::GetPrevNodeOutputFormat(cnode, index);
    if (AnfAlgo::IsFeatureMapInput(cnode, index) &&
        kHWSpecialFormatSet.find(pre_output_format) != kHWSpecialFormatSet.end()) {
      priority_matched_format = !is_init ? pre_output_format : priority_matched_format;
      is_init = true;
    }
    // feature map has two or more special format;
    if (priority_matched_format != pre_output_format && pre_output_format != kOpFormat_DEFAULT) {
      priority_matched_format = kOpFormat_DEFAULT;
    }
    auto input_shape_size = AnfAlgo::GetPrevNodeOutputInferShape(cnode, index).size();
    if (input_shape_size == 5) {
      is_5d_input = true;
    }
    need_change_nd = (need_change_nd || (input_shape_size != 4 && input_shape_size > 1));
  }
  if (need_change_nd && priority_matched_format != kOpFormat_FRAC_NZ) {
    priority_matched_format = kOpFormat_DEFAULT;
  }
  if (is_5d_input && priority_matched_format != kOpFormat_FRAC_NZ) {
    priority_matched_format = kOpFormat_NDC1HWC0;
  }
  AnfAlgo::SetNodeAttr(kPriChoosenFormat, MakeValue(priority_matched_format), cnode);
  return priority_matched_format;
}

/**
 * Compare two vector by priority, select a better vector, like compare two num, first compare highest num location,
 * if equal then next num location
 * example:[3,1,1,1] > [2,2,2,2] > [2,2,1,2] > [2,1,1,3]
 */
bool PriorityChooseItem(const std::vector<int> &cur_item, std::vector<int> *best_item) {
  MS_EXCEPTION_IF_NULL(best_item);
  if (cur_item.size() != best_item->size()) {
    MS_LOG(ERROR) << "Item size should be same!";
    return false;
  }
  // Update the best_item by comparing the cur_item and best_item
  for (size_t i = 0; i < cur_item.size(); i++) {
    if (cur_item[i] > best_item->at(i)) {
      *best_item = cur_item;
      return true;
    } else if (cur_item[i] == best_item->at(i)) {
      continue;
    } else {
      return false;
    }
  }
  return false;
}

void UpdateCurMatchCounts(const kernel::KernelBuildInfo &kernel_build_info, const std::shared_ptr<CNode> &kernel_node,
                          std::vector<int> *const cur_kernelinfo_match_counts) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  MS_EXCEPTION_IF_NULL(cur_kernelinfo_match_counts);
  if (cur_kernelinfo_match_counts->size() < MATCH_COUNT_PRIORITY_END) {
    MS_LOG(EXCEPTION) << "Out of range cur_kernel info_match_counts " << MATCH_COUNT_PRIORITY_END;
  }
  auto pri_match_format = GetPriorityMatchFormat(kernel_node);
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  for (size_t input_index = 0; input_index < input_num; ++input_index) {
    auto input_anf_node = AnfAlgo::VisitKernelWithReturnType(AnfAlgo::GetInputNode(kernel_node, input_index), 0).first;
    MS_EXCEPTION_IF_NULL(input_anf_node);
    // we do not take ValueNode into consideration in graph kernel.
    auto base_score = AnfAlgo::IsFeatureMapInput(kernel_node, input_index) ? kFeatureMapBaseScore : kWeightInitScore;
    if (AnfAlgo::GetOutputDeviceDataType(input_anf_node, 0) == kTypeUnknown) {
      base_score = kWeightUnInitScore;
    }
    if (kernel_build_info.GetInputFormat(input_index) == AnfAlgo::GetPrevNodeOutputFormat(kernel_node, input_index)) {
      (*cur_kernelinfo_match_counts)[MATCH_FORMAT_COUNT] += base_score;
    }
    // we match output fix precision first.
    auto prev_device_type = AnfAlgo::GetPrevNodeOutputPrecision(kernel_node, input_index);
    if (prev_device_type == kTypeUnknown) {
      prev_device_type = AnfAlgo::GetPrevNodeOutputDeviceDataType(kernel_node, input_index);
    }
    if (kernel_build_info.GetInputDeviceType(input_index) == prev_device_type) {
      (*cur_kernelinfo_match_counts)[MATCH_DTYPE_COUNT] += base_score;
    }
    if (kernel_build_info.GetInputFormat(input_index) == pri_match_format) {
      (*cur_kernelinfo_match_counts)[MATCH_SPECIAL_FORMAT_COUNT] += base_score;
    }
    if (kernel_build_info.GetInputFormat(input_index) == kOpFormat_DEFAULT ||
        kernel_build_info.GetInputFormat(input_index) == kOpFormat_NCDHW) {
      (*cur_kernelinfo_match_counts)[MATCH_DEFAULT_FORMAT_COUNT] += base_score;
    }
  }

  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  for (size_t output_index = 0; output_index < output_num; ++output_index) {
    // cal count of same output dtype between abstract and kernel info
    if (kernel_build_info.GetOutputDeviceType(output_index) ==
        AnfAlgo::GetOutputInferDataType(kernel_node, output_index)) {
      (*cur_kernelinfo_match_counts)[MATCH_OUTPUT_DTYPE_COUNT] += 1;
    }
    if (kernel_build_info.GetOutputFormat(output_index) == pri_match_format) {
      (*cur_kernelinfo_match_counts)[MATCH_SPECIAL_FORMAT_COUNT] += 1;
    }
  }
}

std::string PrintRaiseOrReducePrecisionSelectedInfo(
  const CNodePtr &cnode, const std::shared_ptr<kernel::KernelBuildInfo> &selected_kernel_build_info,
  bool precision_reduce) {
  MS_EXCEPTION_IF_NULL(selected_kernel_build_info);
  MS_EXCEPTION_IF_NULL(cnode);
  std::ostringstream buffer;
  buffer << cnode->DebugString();
  if (precision_reduce) {
    buffer << " Reduce precision, node datatype: \n";
  } else {
    buffer << " Raise precision, node datatype: \n";
  }
  PrintInputAndOutputInferType(buffer, cnode);
  buffer << ", select kernel:" << selected_kernel_build_info->ToString();
  return buffer.str();
}

std::shared_ptr<kernel::KernelBuildInfo> ChooseMatchedKernelInfo(
  const CNodePtr &kernel_node, const std::vector<std::shared_ptr<kernel::KernelBuildInfo>> &kernel_info_list) {
  if (kernel_info_list.empty()) {
    return nullptr;
  }
  std::vector<int> most_match_counts = {-1, -1, -1, -1, -1};
  size_t selected_index = 0;
  for (size_t info_index = 0; info_index < kernel_info_list.size(); ++info_index) {
    std::vector<int> cur_kernel_info_match_counts = {0, 0, 0, 0, 0};
    auto kernel_info_ptr = kernel_info_list[info_index];
    MS_EXCEPTION_IF_NULL(kernel_info_ptr);
    UpdateCurMatchCounts(*kernel_info_ptr, kernel_node, &cur_kernel_info_match_counts);
    // Currently the selection policy is the match format count first, and then is datatype counts.
    if (PriorityChooseItem(cur_kernel_info_match_counts, &most_match_counts)) {
      selected_index = SizeToInt(info_index);
    }
  }
  return kernel_info_list[selected_index];
}

std::vector<std::shared_ptr<kernel::KernelBuildInfo>> FilteredKernelInfoByDtype(
  const CNodePtr &cnode, const std::vector<std::shared_ptr<kernel::KernelBuildInfo>> &kernel_info_list) {
  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> result;
  for (const auto &kernel_build_info : kernel_info_list) {
    MS_EXCEPTION_IF_NULL(kernel_build_info);
    if (!MatchInferOutputDataType(cnode, *kernel_build_info)) {
      continue;
    }
    result.push_back(kernel_build_info);
  }
  return result;
}

bool CheckHitTargetDtype(const std::map<TypeId, TypeId> &type_map, const TypeId &in_dtype, const TypeId &device_dtype,
                         bool *flag) {
  auto iter = type_map.find(in_dtype);
  // if infer dtype node in type_map and the infer dtype not equal kernel info dtype, return false
  if (iter == type_map.end() && in_dtype != device_dtype) {
    return false;
  }
  // infer dtype in type_map, but can not find dst dtype that supported raise or reduce,
  // or infer dtype not equal kernel info dtype, return false
  if (iter != type_map.end() && iter->second != device_dtype && in_dtype != device_dtype) {
    return false;
  }
  if (in_dtype == kNumberTypeInt64 && device_dtype == kNumberTypeInt32) {
    *flag = true;
  }
  return true;
}

bool TagRaiseReduce(const std::shared_ptr<kernel::KernelBuildInfo> &kernel_build_info, const CNodePtr &cnode,
                    const std::map<TypeId, TypeId> &type_map) {
  // filte kernel info that unsupported raise or reduce datatype
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(kernel_build_info);
  bool flag = false;
  for (size_t input_index = 0; input_index < kernel_build_info->GetInputNum(); ++input_index) {
    auto in_dtype = AnfAlgo::GetPrevNodeOutputInferDataType(cnode, input_index);
    auto device_dtype = kernel_build_info->GetInputDeviceType(input_index);
    if (device_dtype == kNumberTypeFloat || device_dtype == kNumberTypeFloat32) {
      device_dtype = kNumberTypeFloat32;
    }
    if (!CheckHitTargetDtype(type_map, in_dtype, device_dtype, &flag)) {
      return false;
    }
  }

  for (size_t output_index = 0; output_index < kernel_build_info->GetOutputNum(); ++output_index) {
    auto in_dtype = AnfAlgo::GetOutputInferDataType(cnode, output_index);
    auto device_dtype = kernel_build_info->GetOutputDeviceType(output_index);
    if (device_dtype == kNumberTypeFloat || device_dtype == kNumberTypeFloat32) {
      device_dtype = kNumberTypeFloat32;
    }

    if (!CheckHitTargetDtype(type_map, in_dtype, device_dtype, &flag)) {
      return false;
    }
  }
  if (flag) {
    auto node_name = AnfAlgo::GetCNodeName(cnode);
    MS_LOG(WARNING) << "node:[" << node_name << "]reduce precision from int64 to int32";
  }
  return true;
}

std::vector<std::shared_ptr<kernel::KernelBuildInfo>> FilterRaisedOrReducePrecisionMatchedKernelInfo(
  const CNodePtr &cnode, const std::vector<std::shared_ptr<kernel::KernelBuildInfo>> &kernel_info_list,
  bool *precision_reduce) {
  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> filtered_kernel_info_list;
  const std::map<TypeId, TypeId> raise_map = {{kNumberTypeFloat16, kNumberTypeFloat32}};
  const std::map<TypeId, TypeId> reduce_map = {{kNumberTypeInt64, kNumberTypeInt32},
                                               {kNumberTypeFloat, kNumberTypeFloat16},
                                               {kNumberTypeFloat32, kNumberTypeFloat16}};
  // raise precision
  for (size_t info_index = 0; info_index < kernel_info_list.size(); ++info_index) {
    MS_EXCEPTION_IF_NULL(kernel_info_list[info_index]);
    if (TagRaiseReduce(kernel_info_list[info_index], cnode, raise_map)) {
      filtered_kernel_info_list.push_back(kernel_info_list[info_index]);
    }
  }

  if (!filtered_kernel_info_list.empty()) {
    *precision_reduce = false;
    return filtered_kernel_info_list;
  }

  // reduce precision
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->get_param<bool>(MS_CTX_ENABLE_REDUCE_PRECISION)) {
    for (size_t info_index = 0; info_index < kernel_info_list.size(); ++info_index) {
      MS_EXCEPTION_IF_NULL(kernel_info_list[info_index]);
      if (TagRaiseReduce(kernel_info_list[info_index], cnode, reduce_map)) {
        filtered_kernel_info_list.push_back(kernel_info_list[info_index]);
      }
    }
  }
  if (!filtered_kernel_info_list.empty()) {
    *precision_reduce = true;
  }
  return filtered_kernel_info_list;
}

void SetCastAndWeightFormat(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  if (!AnfAlgo::HasNodeAttr(kAttrPynativeNextIndex, kernel_node) ||
      !AnfAlgo::HasNodeAttr(kAttrPynativeNextOpName, kernel_node)) {
    MS_LOG(EXCEPTION) << "The node [" << kernel_node->DebugString() << "] attr of " << kAttrPynativeNextIndex << " or "
                      << kAttrPynativeNextOpName << " has not been set yet!"
                      << " trace: " << trace::DumpSourceLines(kernel_node);
  }
  auto next_index = AnfAlgo::GetNodeAttr<size_t>(kernel_node, kAttrPynativeNextIndex);
  auto next_op_name = AnfAlgo::GetNodeAttr<std::string>(kernel_node, kAttrPynativeNextOpName);
  auto iter = kNextOpFormatList.find(next_op_name);
  if (iter == kNextOpFormatList.end()) {
    MS_LOG(INFO) << "The op name " << next_op_name << "has not been set in the next op map ";
    return;
  }
  if (iter->second.size() < next_index) {
    MS_LOG(EXCEPTION) << "Next input index " << next_index << "is out of range in the next op map max size is "
                      << iter->second.size() << " trace: " << trace::DumpSourceLines(kernel_node);
  }
  if (AnfAlgo::GetCNodeName(kernel_node) != prim::kPrimCast->name()) {
    MS_LOG(INFO) << "Only supported to change the node Cast's build info!!!";
    return;
  }
  auto format = iter->second[next_index];
  auto info_builder =
    std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(AnfAlgo::GetSelectKernelBuildInfo(kernel_node));
  info_builder->SetInputsFormat({format});
  info_builder->SetOutputsFormat({format});
  AnfAlgo::SetSelectKernelBuildInfo(info_builder->Build(), kernel_node.get());
}

void SetWeightFormat(const AnfNodePtr &real_input_node, const std::vector<string> &output_format,
                     const CNodePtr &kernel_node, size_t input_index) {
  auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  // we set special device info of a input tensor.
  bool is_ref = false;
  auto op_info = kernel::tbe::TbeDynamicShapeUtil::FindOp(AnfAlgo::GetCNodeName(kernel_node), kernel_node);
  if (op_info != nullptr) {
    is_ref = op_info->is_ref();
  }
  auto selected_kernel_info = AnfAlgo::GetSelectKernelBuildInfo(kernel_node);
  if (IsValueNode<tensor::Tensor>(real_input_node) &&
      AnfAlgo::GetOutputDeviceDataType(real_input_node, 0) == kTypeUnknown) {
    builder->SetOutputsFormat(output_format);
    std::vector<TypeId> output_type = {selected_kernel_info->GetInputDeviceType(input_index)};
    builder->SetOutputsDeviceType(output_type);
    AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), real_input_node.get());
    return;
  }
  if (AnfAlgo::GetOutputDeviceDataType(real_input_node, 0) == kTypeUnknown || is_ref) {
    builder->SetOutputsFormat(output_format);
    std::vector<TypeId> output_type = {AnfAlgo::GetOutputInferDataType(real_input_node, 0)};
    builder->SetOutputsDeviceType(output_type);
    AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), real_input_node.get());
  }
}

bool RefreshCastAndParamWeightFormat(const AnfNodePtr &input_node, const string &format) {
  MS_EXCEPTION_IF_NULL(input_node);
  if (!input_node->isa<CNode>()) {
    return false;
  }
  auto cast_node = input_node->cast<CNodePtr>();
  if (AnfAlgo::GetCNodeName(cast_node) != prim::kPrimCast->name()) {
    return true;
  }
  if (AnfAlgo::IsFeatureMapOutput(cast_node)) {
    return true;
  }
  auto info_builder =
    std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(AnfAlgo::GetSelectKernelBuildInfo(input_node));
  info_builder->SetInputsFormat({format});
  info_builder->SetOutputsFormat({format});
  AnfAlgo::SetSelectKernelBuildInfo(info_builder->Build(), cast_node.get());
  auto cast_input_node = AnfAlgo::VisitKernel(AnfAlgo::GetInputNode(cast_node, 0), 0);
  SetWeightFormat(cast_input_node.first, {format}, cast_node, 0);
  return true;
}
}  // namespace
void SetTensorDeviceInfo(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  auto selected_kernel_info = AnfAlgo::GetSelectKernelBuildInfo(kernel_node);
  MS_EXCEPTION_IF_NULL(selected_kernel_info);
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  for (size_t input_index = 0; input_index < input_num; ++input_index) {
    auto input_kernel_node = AnfAlgo::GetInputNode(kernel_node, input_index);
    MS_EXCEPTION_IF_NULL(input_kernel_node);
    auto input_with_index = AnfAlgo::VisitKernel(input_kernel_node, 0);
    MS_EXCEPTION_IF_NULL(input_with_index.first);
    auto real_input_node = input_with_index.first;
    if (RefreshCastAndParamWeightFormat(real_input_node, selected_kernel_info->GetInputFormat(input_index))) {
      continue;
    }
    if (real_input_node->isa<Parameter>() && !AnfAlgo::IsParameterWeight(real_input_node->cast<ParameterPtr>())) {
      continue;
    }
    if (AnfAlgo::OutputAddrExist(real_input_node, 0)) {
      continue;
    }
    auto refresh_format = selected_kernel_info->GetInputFormat(input_index);
    std::vector<std::string> output_format = {refresh_format};
    // if not find in host convert format map means the host has not registered the convert function of this format
    if (trans::kTransFormatMapOfHostToDevice.find(refresh_format) == trans::kTransFormatMapOfHostToDevice.end() &&
        refresh_format != kOpFormat_DEFAULT) {
      output_format = {AnfAlgo::GetOutputFormat(real_input_node, 0)};
    }
    SetWeightFormat(real_input_node, output_format, kernel_node, input_index);
  }
}

KernelSelectStatus SetMatchedKernelInfo(const CNodePtr &kernel_node,
                                        const std::vector<std::shared_ptr<kernel::KernelBuildInfo>> &kernel_info_list) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  KernelSelectStatus select_status = kNoMatched;
  bool precision_reduce = false;
  std::shared_ptr<kernel::KernelBuildInfo> selected_kernel_info = nullptr;
  // Matched kernel info
  // Filter kernel info matched with me inferred type
  auto filtered_kernel_info_list = FilteredKernelInfoByDtype(kernel_node, kernel_info_list);
  if (!filtered_kernel_info_list.empty()) {
    selected_kernel_info = ChooseMatchedKernelInfo(kernel_node, filtered_kernel_info_list);
    select_status = kStatusAllMatched;
  } else {
    // selected kernel info using raised precision or reduce precision
    filtered_kernel_info_list =
      FilterRaisedOrReducePrecisionMatchedKernelInfo(kernel_node, kernel_info_list, &precision_reduce);
    selected_kernel_info = ChooseMatchedKernelInfo(kernel_node, filtered_kernel_info_list);
    if (selected_kernel_info == nullptr) {
      return select_status;
    } else {
      MS_LOG(INFO) << PrintRaiseOrReducePrecisionSelectedInfo(kernel_node, selected_kernel_info, precision_reduce);
      select_status = precision_reduce ? kStatusReducePrecision : kStatusRaisePrecision;
    }
  }
  // Set kernel info to the anfnode
  AnfAlgo::SetSelectKernelBuildInfo(selected_kernel_info, kernel_node.get());
  // Set format and data type for input tensor.
  if (AnfAlgo::HasNodeAttr(kAttrPynativeNextOpName, kernel_node)) {
    SetCastAndWeightFormat(kernel_node);
  }
  SetTensorDeviceInfo(kernel_node);
  return select_status;
}

KernelSelectStatus SelectKernelInfo(const CNodePtr &kernel_node, KernelType kernel_type) {
  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> kernel_info_list;
  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> aicpu_kernel_info_list;
  MS_EXCEPTION_IF_NULL(kernel_node);
  if (AnfAlgo::IsGraphKernel(kernel_node)) {
    auto func_graph = GetValueNode<FuncGraphPtr>(kernel_node->input(kAnfPrimitiveIndex));
    MS_EXCEPTION_IF_NULL(func_graph);
    SelectGraphKernelInfo(kernel_node, func_graph);
    return kStatusAllMatched;
  }
  kernel::KernelQuery(kernel_node, &kernel_info_list, kernel_type);
  auto select_status = SetMatchedKernelInfo(kernel_node, kernel_info_list);
  // If aicore not find valid kernel info reloading aicpu kernel info list to find it
  if (select_status == kNoMatched) {
    MS_LOG(WARNING) << "The node [" << kernel_node->DebugString()
                    << "] cannot find valid TBE kernel info, try to get aicpu kernel info";
    kernel::AICPUQuery(kernel_node, &aicpu_kernel_info_list);
    select_status = SetMatchedKernelInfo(kernel_node, aicpu_kernel_info_list);
    AnfAlgo::SetNodeAttr(kAttrIsAICPUKernel, MakeValue(true), kernel_node);
  }
  // The kernel info not finded both in the aicpu kernel list & aicore kernel list
  if (select_status == kNoMatched) {
    std::ostringstream buffer;
    PrintInputAndOutputInferType(buffer, kernel_node);
    MS_LOG(WARNING) << ">>> Candidates kernel info list:";
    for (size_t index = 0; index < kernel_info_list.size(); ++index) {
      MS_LOG(WARNING) << "Kernel [" << index << "] :" << kernel_info_list[index]->ToString();
    }
    for (size_t index = 0; index < aicpu_kernel_info_list.size(); ++index) {
      MS_LOG(WARNING) << "Kernel [" << (kernel_info_list.size() + index)
                      << "] :" << aicpu_kernel_info_list[index]->ToString();
    }
    if (IsPrimitiveCNode(kernel_node, prim::kPrimLabelSwitch)) {
      auto selected_kernel_info = ChooseMatchedKernelInfo(kernel_node, kernel_info_list);
      AnfAlgo::SetSelectKernelBuildInfo(selected_kernel_info, kernel_node.get());
      // Set format and data type for input tensor.
      SetTensorDeviceInfo(kernel_node);
    } else {
      MS_LOG(WARNING) << " <<<";
      MS_EXCEPTION(TypeError) << "The node [" << kernel_node->DebugString()
                              << "] cannot find valid kernel info, not supported the type:" << buffer.str()
                              << ", please refer to the supported dtypes in candidates kernel info list."
                              << " trace: " << trace::DumpSourceLines(kernel_node);
    }
  }
  return select_status;
}

void SetKernelInfo(const CNodePtr &kernel_node, KernelType kernel_type) {
  auto kernel_info = static_cast<device::KernelInfo *>(kernel_node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto kernel_build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(kernel_build_info);

  if (AnfAlgo::IsGraphKernel(kernel_node)) {
    return;
  }

  auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  builder->SetOriginDataFormat(kernel_build_info->GetOriginDataFormat());
  builder->SetInputsFormat(kernel_build_info->GetAllInputFormats());
  builder->SetInputsDeviceType(kernel_build_info->GetAllInputDeviceTypes());
  builder->SetOutputsFormat(kernel_build_info->GetAllOutputFormats());
  builder->SetOutputsDeviceType(kernel_build_info->GetAllOutputDeviceTypes());
  builder->SetOpPattern(kernel_build_info->op_pattern());
  builder->SetFusionType(kernel_build_info->fusion_type());

  auto new_kernel_type = kernel_type;
  auto new_processor = kernel_build_info->processor();
  if (kernel_type == UNKNOWN_KERNEL_TYPE) {
    std::vector<std::shared_ptr<kernel::KernelBuildInfo>> kernel_info_list;
    std::vector<std::shared_ptr<kernel::KernelBuildInfo>> aicpu_kernel_info_list;
    kernel::KernelQuery(kernel_node, &kernel_info_list, kernel_type);
    auto select_status = SetMatchedKernelInfo(kernel_node, kernel_info_list);
    if (select_status != kNoMatched) {
      new_kernel_type = TBE_KERNEL;
      new_processor = kernel::Processor::AICORE;
      MS_LOG(INFO) << kernel_node->fullname_with_scope() << " uses TBE_KERNEL";
    } else {
      kernel::AICPUQuery(kernel_node, &aicpu_kernel_info_list);
      select_status = SetMatchedKernelInfo(kernel_node, aicpu_kernel_info_list);
      if (select_status != kNoMatched) {
        new_kernel_type = AICPU_KERNEL;
        new_processor = kernel::Processor::AICPU;
        MS_LOG(INFO) << kernel_node->fullname_with_scope() << " uses AICPU_KERNEL";
      }
    }
  }
  if (new_kernel_type == UNKNOWN_KERNEL_TYPE) {
    new_kernel_type = AKG_KERNEL;
    new_processor = kernel::Processor::AICORE;
    MS_LOG(INFO) << kernel_node->fullname_with_scope() << " uses AKG_KERNEL";
  }
  builder->SetKernelType(new_kernel_type);
  builder->SetProcessor(new_processor);
  AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), kernel_node.get());
}

}  // namespace ascend
}  // namespace device
}  // namespace mindspore
