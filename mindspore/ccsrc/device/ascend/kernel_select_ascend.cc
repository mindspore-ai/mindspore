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

#include "device/ascend/kernel_select_ascend.h"

#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <unordered_set>

#include "common/utils.h"
#include "debug/anf_ir_dump.h"
#include "operator/ops.h"
#include "ir/func_graph.h"
#include "utils/context/ms_context.h"
#include "session/anf_runtime_algorithm.h"
#include "device/kernel_info.h"
#include "kernel/common_utils.h"
#include "kernel/kernel_query.h"
#include "kernel/oplib/oplib.h"
#include "kernel/kernel_build_info.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace {
const float kWegihtBaseScore = 1;
const float kFeatureMapBaseScore = 10;
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

const int kUnSupportMixedDataTypeIndex = -1;

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
  for (size_t index = 0; index < AnfAlgo::GetInputTensorNum(cnode); ++index) {
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
    need_change_nd = (need_change_nd || (input_shape_size != 4 && input_shape_size > 1));
  }
  if (need_change_nd && priority_matched_format != kOpFormat_FRAC_NZ) {
    priority_matched_format = kOpFormat_DEFAULT;
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
    MS_LOG(EXCEPTION) << "Out of range cur_kernelinfo_match_counts " << MATCH_COUNT_PRIORITY_END;
  }
  auto pri_match_format = GetPriorityMatchFormat(kernel_node);
  for (size_t input_index = 0; input_index < AnfAlgo::GetInputTensorNum(kernel_node); ++input_index) {
    auto input_anf_node = kernel_node->input(input_index + 1);
    // we do not take ValueNode into consideration in graph kernel.
    if (kernel_build_info.kernel_type() == KernelType::AKG_KERNEL) {
      if (input_anf_node->isa<ValueNode>() && AnfAlgo::GetOutputDeviceDataType(input_anf_node, 0) == kTypeUnknown) {
        continue;
      }
    }
    auto base_score = AnfAlgo::IsFeatureMapInput(kernel_node, input_index) ? kFeatureMapBaseScore : kWegihtBaseScore;
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
    if (kernel_build_info.GetInputFormat(input_index) == kOpFormat_DEFAULT) {
      (*cur_kernelinfo_match_counts)[MATCH_DEFAULT_FORMAT_COUNT] += base_score;
    }
  }

  for (size_t output_index = 0; output_index < AnfAlgo::GetOutputTensorNum(kernel_node); ++output_index) {
    // cal count of same output dtype between abstract and kernel info
    if (kernel_build_info.GetOutputDeviceType(output_index) ==
        AnfAlgo::GetOutputInferDataType(kernel_node, output_index)) {
      (*cur_kernelinfo_match_counts)[MATCH_OUTPUT_DTYPE_COUNT] += 1;
    }
  }
}

void AddSupportMixedPrecisionDataTypeIndex(TypeId data_type, std::vector<int> *support_index) {
  MS_EXCEPTION_IF_NULL(support_index);
  int index = kUnSupportMixedDataTypeIndex;
  switch (data_type) {
    case kNumberTypeFloat16:
      index = 0;
      break;
    case kNumberTypeFloat32:
    case kNumberTypeFloat:
      index = 1;
      break;
    default:
      break;
  }
  support_index->push_back(index);
}

void AddKernelInputSupportDataType(const kernel::KernelBuildInfo &kernel_build_info, size_t input_index,
                                   std::vector<int> *support_datatype_index, std::vector<TypeId> *support_datatype) {
  MS_EXCEPTION_IF_NULL(support_datatype);
  auto data_type = kernel_build_info.GetInputDeviceType(input_index);
  support_datatype->push_back(data_type);
  AddSupportMixedPrecisionDataTypeIndex(data_type, support_datatype_index);
}

void AddKernelOutputSupportDataType(const kernel::KernelBuildInfo &kernel_build_info, size_t output_index,
                                    std::vector<int> *support_datatype_index, std::vector<TypeId> *support_datatype) {
  MS_EXCEPTION_IF_NULL(support_datatype);
  auto data_type = kernel_build_info.GetOutputDeviceType(output_index);
  support_datatype->push_back(data_type);
  AddSupportMixedPrecisionDataTypeIndex(data_type, support_datatype_index);
}

void AddNodeInputDataType(const CNodePtr &kernel_node, size_t input_index,
                          std::vector<int> *node_mix_precision_datatype_index,
                          std::vector<TypeId> *node_mix_precision_datatype) {
  AnfNodePtr cur_input = AnfAlgo::GetInputNode(kernel_node, input_index);
  MS_EXCEPTION_IF_NULL(cur_input);
  MS_EXCEPTION_IF_NULL(node_mix_precision_datatype);
  TypeId input_origin_type = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, input_index);
  AddSupportMixedPrecisionDataTypeIndex(input_origin_type, node_mix_precision_datatype_index);
  node_mix_precision_datatype->push_back(input_origin_type);
}

void AddNodeOutputDataType(const CNodePtr &kernel_node, size_t output_index,
                           std::vector<int> *node_mix_precision_datatype_index,
                           std::vector<TypeId> *node_mix_precision_datatype) {
  MS_EXCEPTION_IF_NULL(node_mix_precision_datatype);
  auto output_origin_type = AnfAlgo::GetOutputInferDataType(kernel_node, output_index);
  AddSupportMixedPrecisionDataTypeIndex(output_origin_type, node_mix_precision_datatype_index);
  node_mix_precision_datatype->push_back(output_origin_type);
}

void CheckDataTypeInputs(const std::vector<int> &node_mix_precision_datatype_index,
                         const std::vector<TypeId> &node_mix_precision_datatype,
                         const std::map<size_t, std::vector<TypeId>> &kernel_support_datatypes,
                         std::map<size_t, std::vector<int>> *kernel_match_datatype_idx) {
  if (node_mix_precision_datatype_index.size() != node_mix_precision_datatype.size()) {
    MS_LOG(EXCEPTION) << "Node datatype index size " << node_mix_precision_datatype_index.size() << " != datatype size "
                      << node_mix_precision_datatype.size();
  }
  MS_EXCEPTION_IF_NULL(kernel_match_datatype_idx);
  if (kernel_support_datatypes.size() != kernel_match_datatype_idx->size()) {
    MS_LOG(EXCEPTION) << "Kernel datatype index size " << kernel_match_datatype_idx->size() << " != datatype size "
                      << kernel_support_datatypes.size();
  }
}

bool RaiseDataTypePrecisionSelect(const std::vector<int> &node_mix_precision_datatype_index,
                                  const std::vector<TypeId> &node_mix_precision_datatype,
                                  const std::map<size_t, std::vector<TypeId>> &kernel_support_datatypes,
                                  std::map<size_t, std::vector<int>> *kernel_match_datatype_idx) {
  MS_EXCEPTION_IF_NULL(kernel_match_datatype_idx);
  CheckDataTypeInputs(node_mix_precision_datatype_index, node_mix_precision_datatype, kernel_support_datatypes,
                      kernel_match_datatype_idx);
  for (size_t i = 0; i < node_mix_precision_datatype_index.size(); ++i) {
    if (node_mix_precision_datatype[i] == kTypeUnknown) {
      continue;
    }
    auto iter = kernel_match_datatype_idx->begin();
    while (iter != kernel_match_datatype_idx->end()) {
      if (node_mix_precision_datatype_index[i] == kUnSupportMixedDataTypeIndex) {
        auto find_iter = kernel_support_datatypes.find(iter->first);
        if (find_iter == kernel_support_datatypes.end()) {
          MS_LOG(EXCEPTION) << "Kernel datatype index:%lu can not be found " << iter->first;
        }
        if (i >= find_iter->second.size()) {
          MS_LOG(EXCEPTION) << "Node index " << i << "kernel datatype size " << find_iter->second.size();
        }
        if (node_mix_precision_datatype[i] != find_iter->second[i]) {
          iter = kernel_match_datatype_idx->erase(iter);
        } else {
          ++iter;
        }
        continue;
      }
      auto datatype_indexes = iter->second;
      if (i >= datatype_indexes.size()) {
        MS_LOG(EXCEPTION) << "Node datatype index: " << i << " kernel support size " << datatype_indexes.size();
      }
      if (datatype_indexes[i] < node_mix_precision_datatype_index[i]) {
        iter = kernel_match_datatype_idx->erase(iter);
      } else {
        ++iter;
      }
    }
  }
  return !kernel_match_datatype_idx->empty();
}

bool CanDataTypeReduce(const std::vector<int> &datatype_indexes, int check_index,
                       const std::vector<int> &node_mix_precision_datatype_index) {
  auto check_index_tmp = IntToSize(check_index);
  if (check_index_tmp < datatype_indexes.size() && check_index_tmp < node_mix_precision_datatype_index.size()) {
    return datatype_indexes[check_index] != kUnSupportMixedDataTypeIndex &&
           datatype_indexes[check_index] <= node_mix_precision_datatype_index[check_index];
  }
  MS_LOG(EXCEPTION) << "Check index " << check_index << "is outof range";
}

bool RaiseOrReduceDataTypePrecisionSelect(const std::vector<int> &node_mix_precision_datatype_index,
                                          const std::vector<TypeId> &node_mix_precision_datatype,
                                          const std::map<size_t, std::vector<TypeId>> &kernel_support_datatypes,
                                          std::map<size_t, std::vector<int>> *kernel_match_datatype_idx) {
  MS_EXCEPTION_IF_NULL(kernel_match_datatype_idx);
  CheckDataTypeInputs(node_mix_precision_datatype_index, node_mix_precision_datatype, kernel_support_datatypes,
                      kernel_match_datatype_idx);
  for (size_t i = 0; i < node_mix_precision_datatype_index.size(); ++i) {
    if (node_mix_precision_datatype[i] == kTypeUnknown) {
      continue;
    }
    auto iter = kernel_match_datatype_idx->begin();
    while (iter != kernel_match_datatype_idx->end()) {
      if (node_mix_precision_datatype_index[i] == kUnSupportMixedDataTypeIndex) {
        auto find_iter = kernel_support_datatypes.find(iter->first);
        if (find_iter == kernel_support_datatypes.end()) {
          MS_LOG(EXCEPTION) << "Kernel datatype index:%lu can not be found " << iter->first;
        }
        if (i >= find_iter->second.size()) {
          MS_LOG(EXCEPTION) << "Node index " << i << " >= kernel datatype size " << find_iter->second.size();
        }
        if (node_mix_precision_datatype[i] != find_iter->second[i]) {
          iter = kernel_match_datatype_idx->erase(iter);
        } else {
          ++iter;
        }
        continue;
      }
      auto datatype_indexes = iter->second;
      if (i >= datatype_indexes.size()) {
        MS_LOG(EXCEPTION) << "Index " << i << "> kernel datatype indexes size " << datatype_indexes.size();
      }
      if (!CanDataTypeReduce(datatype_indexes, i, node_mix_precision_datatype_index)) {
        iter = kernel_match_datatype_idx->erase(iter);
      } else {
        ++iter;
      }
    }
  }
  return !kernel_match_datatype_idx->empty();
}

void AddNodeAndKernelDataType(const CNodePtr &kernel_node, const kernel::KernelBuildInfo &kernel_build_info,
                              std::vector<int> *support_indexes, std::vector<TypeId> *node_mix_precision_datatype,
                              std::vector<TypeId> *support_datatypes,
                              std::vector<int> *node_mix_precision_datatype_index) {
  MS_EXCEPTION_IF_NULL(node_mix_precision_datatype);
  bool add_node_datatype_flag = false;
  if (node_mix_precision_datatype->empty()) {
    add_node_datatype_flag = true;
  }
  for (size_t input_index = 0; input_index < kernel_build_info.GetInputNum(); ++input_index) {
    AddKernelInputSupportDataType(kernel_build_info, input_index, support_indexes, support_datatypes);
    if (add_node_datatype_flag) {
      AddNodeInputDataType(kernel_node, input_index, node_mix_precision_datatype_index, node_mix_precision_datatype);
    }
  }
  // Check output data type
  for (size_t output_index = 0; output_index < kernel_build_info.GetOutputNum(); ++output_index) {
    AddKernelOutputSupportDataType(kernel_build_info, output_index, support_indexes, support_datatypes);
    if (add_node_datatype_flag) {
      AddNodeOutputDataType(kernel_node, output_index, node_mix_precision_datatype_index, node_mix_precision_datatype);
    }
  }
}

void PrecisionReduce(const std::vector<int> &node_mix_precision_datatype_index,
                     const std::vector<TypeId> &node_mix_precision_datatype,
                     const std::map<size_t, std::vector<TypeId>> &kernel_support_datatype,
                     std::map<size_t, std::vector<int>> *kernel_match_datatype_idx, bool *precision_reduce) {
  MS_EXCEPTION_IF_NULL(kernel_match_datatype_idx);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  MS_EXCEPTION_IF_NULL(precision_reduce);
  std::map<size_t, std::vector<int>> kernel_match_datatype_idx_copy = *kernel_match_datatype_idx;
  // raise precision
  bool selected_ret = RaiseDataTypePrecisionSelect(node_mix_precision_datatype_index, node_mix_precision_datatype,
                                                   kernel_support_datatype, kernel_match_datatype_idx);
  if (selected_ret) {
    *precision_reduce = false;
    return;
  }
  if (context_ptr->enable_reduce_precision()) {
    selected_ret = RaiseOrReduceDataTypePrecisionSelect(node_mix_precision_datatype_index, node_mix_precision_datatype,
                                                        kernel_support_datatype, &kernel_match_datatype_idx_copy);
  }
  if (selected_ret) {
    *precision_reduce = true;
    *kernel_match_datatype_idx = kernel_match_datatype_idx_copy;
  }
}

void PrintRaiseOrReducePrecisionSelectedInfo(const CNodePtr &cnode,
                                             const std::shared_ptr<kernel::KernelBuildInfo> &selected_kernel_build_info,
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
  MS_LOG(INFO) << buffer.str();
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

std::vector<std::shared_ptr<kernel::KernelBuildInfo>> FilterRaisedOrReducePrecisionMatchedKernelInfo(
  const CNodePtr &cnode, const std::vector<std::shared_ptr<kernel::KernelBuildInfo>> &kernel_info_list,
  bool *precision_reduce) {
  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> filtered_kernel_info_list;
  std::map<size_t, std::vector<int>> kernel_match_datatype_idx;
  std::map<size_t, std::vector<TypeId>> kernel_support_datatype;
  std::vector<int> node_mix_precision_datatype_index;
  std::vector<TypeId> node_mix_precision_datatype;
  for (size_t info_index = 0; info_index < kernel_info_list.size(); ++info_index) {
    std::vector<int> support_indexes;
    std::vector<TypeId> support_datatypes;
    MS_EXCEPTION_IF_NULL(kernel_info_list[info_index]);
    AddNodeAndKernelDataType(cnode, *kernel_info_list[info_index], &support_indexes, &node_mix_precision_datatype,
                             &support_datatypes, &node_mix_precision_datatype_index);
    kernel_match_datatype_idx[info_index] = support_indexes;
    kernel_support_datatype[info_index] = support_datatypes;
  }
  PrecisionReduce(node_mix_precision_datatype_index, node_mix_precision_datatype, kernel_support_datatype,
                  &kernel_match_datatype_idx, precision_reduce);
  std::transform(
    kernel_match_datatype_idx.begin(), kernel_match_datatype_idx.end(), std::back_inserter(filtered_kernel_info_list),
    [&](const std::pair<size_t, std::vector<int>> &matched_idx) -> std::shared_ptr<kernel::KernelBuildInfo> {
      return kernel_info_list[matched_idx.first];
    });
  return filtered_kernel_info_list;
}
}  // namespace

void SetTensorDeviceInfo(const kernel::KernelBuildInfo &selected_kernel_info, const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  for (size_t input_index = 0; input_index < AnfAlgo::GetInputTensorNum(kernel_node); ++input_index) {
    auto input_kernel_node = AnfAlgo::GetInputNode(kernel_node, input_index);
    MS_EXCEPTION_IF_NULL(input_kernel_node);
    auto input_with_index = AnfAlgo::VisitKernel(input_kernel_node, 0);
    MS_EXCEPTION_IF_NULL(input_with_index.first);
    auto real_input_node = input_with_index.first;
    if (real_input_node->isa<CNode>()) {
      continue;
    }
    if (real_input_node->isa<Parameter>() && !AnfAlgo::IsParameterWeight(real_input_node->cast<ParameterPtr>())) {
      continue;
    }
    auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
    if (IsValueNode<tensor::Tensor>(input_kernel_node) &&
        AnfAlgo::GetOutputDeviceDataType(input_kernel_node, 0) == kTypeUnknown) {
      std::vector<std::string> output_format = {selected_kernel_info.GetInputFormat(input_index)};
      builder->SetOutputsFormat(output_format);
      std::vector<TypeId> output_type = {selected_kernel_info.GetInputDeviceType(input_index)};
      builder->SetOutputsDeviceType(output_type);
      AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), input_kernel_node.get());
      continue;
    }
    // we set special device info of a input tensor.
    bool is_ref = false;
    auto op_info = kernel::OpLib::FindOp(AnfAlgo::GetCNodeName(kernel_node), kernel::kTBE);
    if (op_info != nullptr) {
      is_ref = op_info->is_ref();
    }
    MS_EXCEPTION_IF_NULL(MsContext::GetInstance());
    if (MsContext::GetInstance()->execution_mode() == kPynativeMode &&
        AnfAlgo::GetOutputDeviceDataType(real_input_node, 0) != kTypeUnknown) {
      continue;
    }
    if (AnfAlgo::GetOutputDeviceDataType(real_input_node, 0) == kTypeUnknown || is_ref) {
      std::vector<std::string> output_format = {selected_kernel_info.GetInputFormat(input_index)};
      builder->SetOutputsFormat(output_format);
      std::vector<TypeId> output_type = {selected_kernel_info.GetInputDeviceType(input_index)};
      builder->SetOutputsDeviceType(output_type);
      AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), real_input_node.get());
    }
  }
}

KernelSelectStatus SetMatchedKernelInfo(const CNodePtr &kernel_node,
                                        const std::vector<std::shared_ptr<kernel::KernelBuildInfo>> &kernel_info_list) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  KernelSelectStatus select_status = kNoMatched;
  bool precision_reduce = false;
  std::shared_ptr<kernel::KernelBuildInfo> selected_kernel_info = nullptr;
  // Matched kernel info
  // Filter kernel info matched with me infered type
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
      PrintRaiseOrReducePrecisionSelectedInfo(kernel_node, selected_kernel_info, precision_reduce);
      select_status = precision_reduce ? kStatusReducePrecision : kStatusRaisePrecision;
    }
  }
  // Set kernel info to the anfnode
  AnfAlgo::SetSelectKernelBuildInfo(selected_kernel_info, kernel_node.get());
  // Set format and data type for input tensor.
  SetTensorDeviceInfo(*selected_kernel_info, kernel_node);
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
    MS_LOG(WARNING) << " <<<";
    MS_EXCEPTION(TypeError) << "The node [" << kernel_node->DebugString()
                            << "] cannot find valid kernel info, not supported the type:" << buffer.str()
                            << ", please refer to the supported dtypes in candidates kernel info list";
  }
  return select_status;
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
