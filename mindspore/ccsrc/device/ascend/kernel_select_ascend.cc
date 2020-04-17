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
#include <set>
#include <unordered_map>
#include "kernel/oplib/oplib.h"
#include "kernel/kernel_query.h"
#include "session/anf_runtime_algorithm.h"
#include "kernel/kernel_build_info.h"
#include "utils/context/ms_context.h"
#include "operator/ops.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace {
const float kWegihtBaseScore = 1;
const float kFeatureMapBaseScore = 10;
enum MatchCountPriority : int {
  MATCH_COUNT_PRIORITY_BEGIN = 0,
  MATCH_DTYPE_COUNT = MATCH_COUNT_PRIORITY_BEGIN,
  MATCH_FORMAT_COUNT,
  MATCH_SPECIAL_FORMAT_COUNT,
  MATCH_OUTPUT_DTYPE_COUNT,
  MATCH_COUNT_PRIORITY_END
};

const size_t kMaxCount = 0xffffffff;
const int kUnSupportMixedDataTypeIndex = -1;

const std::set<std::string> kOpFormatList = {kOpFormat_DEFAULT, kOpFormat_NC1KHKWHWC0, kOpFormat_ND,
                                             kOpFormat_NCHW,    kOpFormat_NHWC,        kOpFormat_HWCN,
                                             kOpFormat_NC1HWC0, kOpFormat_FRAC_Z,      kOpFormat_C1HWNCoC0,
                                             kOpFormat_FRAC_NZ, kOpFormat_NC1HWC0_C04};

bool IsShapeMatchFormat(const std::vector<size_t> &shape, const std::string &format) {
  // if format is default, it remarkes support all format
  if (kOpFormatList.find(format) == kOpFormatList.end()) {
    MS_LOG(EXCEPTION) << "got the unknown format " << format;
  }
  if (format == kOpFormat_DEFAULT) {
    return true;
  }
  // if shape size is 0, the shape will be a scalar
  if (shape.empty()) {
    return true;
  }
  if (shape.size() > kShapeSupportFormatMap.size()) {
    return false;
  }
  if (format == kOpFormat_FRAC_NZ && shape.size() >= 2) {
    return true;
  }
  return !(kShapeSupportFormatMap[shape.size() - 1].find(format) == kShapeSupportFormatMap[shape.size() - 1].end());
}

bool IsValidKernelInfo(const std::shared_ptr<CNode> &kernel_node, const kernel::KernelBuildInfo &kernel_build_info) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  auto check_function = [](const std::vector<size_t> &shape, const std::string &format) -> bool {
    if (!IsShapeMatchFormat(shape, format)) {
      return false;
    }
    for (auto shape_value : shape) {
      if (shape_value == 0) {
        MS_LOG(EXCEPTION) << "dimension size of the tensor shape should be a positive integer, but got " << shape_value;
      }
    }
    return true;
  };
  if (AnfAlgo::GetCNodeName(kernel_node) == prim::kPrimCast->name()) {
    return AnfAlgo::GetOutputInferDataType(kernel_node, 0) == kernel_build_info.GetOutputDeviceType(0) &&
           AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 0) == kernel_build_info.GetInputDeviceType(0);
  }
  for (size_t index = 0; index < kernel_build_info.GetOutputNum(); ++index) {
    auto output_shape = AnfAlgo::GetOutputInferShape(kernel_node, index);
    if (!check_function(output_shape, kernel_build_info.GetOutputFormat(index))) {
      return false;
    }
  }
  for (size_t index = 0; index < kernel_build_info.GetInputNum(); ++index) {
    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, index);
    if (!check_function(input_shape, kernel_build_info.GetInputFormat(index))) {
      return false;
    }
  }
  return true;
}

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
        kNeedTransFormatSet.find(pre_output_format) != kNeedTransFormatSet.end()) {
      priority_matched_format = !is_init ? priority_matched_format : pre_output_format;
      is_init = true;
    }
    // feature map has two or more special format;
    if (priority_matched_format != pre_output_format && pre_output_format != kOpFormat_DEFAULT) {
      priority_matched_format = kOpFormat_DEFAULT;
    }
    auto input_shape_size = AnfAlgo::GetPrevNodeOutputInferShape(cnode, index).size();
    need_change_nd = (need_change_nd || (input_shape_size != 4 && input_shape_size > 1));
  }
  if (need_change_nd) {
    priority_matched_format = kOpFormat_DEFAULT;
  }
  return priority_matched_format;
}
/**
 * compare two vector by priority, select a better vector, like compare two num, first compare highest num location,
 * if equal then next num location
 * example:[3,1,1,1] > [2,2,2,2] > [2,2,1,2] > [2,1,1,3]
 */
bool PriorityChooseItem(const std::vector<int> &cur_item, std::vector<int> *best_item) {
  MS_EXCEPTION_IF_NULL(best_item);
  if (cur_item.size() != best_item->size()) {
    MS_LOG(ERROR) << "item size should be same!";
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
    auto base_score = AnfAlgo::IsFeatureMapInput(kernel_node, input_index) ? kFeatureMapBaseScore : kWegihtBaseScore;
    if (kernel_build_info.GetInputFormat(input_index) == AnfAlgo::GetPrevNodeOutputFormat(kernel_node, input_index)) {
      (*cur_kernelinfo_match_counts)[MATCH_FORMAT_COUNT] += base_score;
    }
    if (kernel_build_info.GetInputDeviceType(input_index) ==
        AnfAlgo::GetPrevNodeOutputDeviceDataType(kernel_node, input_index)) {
      (*cur_kernelinfo_match_counts)[MATCH_DTYPE_COUNT] += base_score;
    }
    if (kernel_build_info.GetInputFormat(input_index) == pri_match_format) {
      (*cur_kernelinfo_match_counts)[MATCH_SPECIAL_FORMAT_COUNT] += base_score;
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
    std::shared_ptr<kernel::KernelBuildInfo::KernelBuildInfoBuilder> builder =
      std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
    // we set special device info of a input tensor.
    bool is_ref = false;
    auto op_info = mindspore::kernel::OpLib::FindOp(AnfAlgo::GetCNodeName(kernel_node), kernel::kTBE);
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

void AddSupportMixedPrecisionDataTypeIndex(TypeId data_type, std::vector<int> *support_index) {
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
  auto data_type = kernel_build_info.GetInputDeviceType(input_index);
  support_datatype->push_back(data_type);
  AddSupportMixedPrecisionDataTypeIndex(data_type, support_datatype_index);
}

void AddKernelOutputSupportDataType(const kernel::KernelBuildInfo &kernel_build_info, size_t output_index,
                                    std::vector<int> *support_datatype_index, std::vector<TypeId> *support_datatype) {
  auto data_type = kernel_build_info.GetOutputDeviceType(output_index);
  support_datatype->push_back(data_type);
  AddSupportMixedPrecisionDataTypeIndex(data_type, support_datatype_index);
}

void AddNodeInputDataType(const CNodePtr &kernel_node, size_t input_index,
                          std::vector<int> *node_mix_precision_datatype_index,
                          std::vector<TypeId> *node_mix_precision_datatype) {
  AnfNodePtr cur_input = AnfAlgo::GetInputNode(kernel_node, input_index);
  MS_EXCEPTION_IF_NULL(cur_input);
  TypeId input_origin_type;
  if (cur_input->isa<Parameter>() && AnfAlgo::IsParameterWeight(cur_input->cast<ParameterPtr>())) {
    // weight
    input_origin_type = AnfAlgo::GetOutputDeviceDataType(cur_input, 0);
  } else if (cur_input->isa<ValueNode>()) {
    input_origin_type = AnfAlgo::GetOutputDeviceDataType(cur_input, 0);
  } else {
    // feature map
    input_origin_type = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, input_index);
  }
  AddSupportMixedPrecisionDataTypeIndex(input_origin_type, node_mix_precision_datatype_index);
  node_mix_precision_datatype->push_back(input_origin_type);
}

void AddNodeOutputDataType(const CNodePtr &kernel_node, size_t output_index,
                           std::vector<int> *node_mix_precision_datatype_index,
                           std::vector<TypeId> *node_mix_precision_datatype) {
  auto output_origin_type = AnfAlgo::GetOutputInferDataType(kernel_node, output_index);
  AddSupportMixedPrecisionDataTypeIndex(output_origin_type, node_mix_precision_datatype_index);
  node_mix_precision_datatype->push_back(output_origin_type);
}

void CheckDataTypeInputs(const std::vector<int> &node_mix_precision_datatype_index,
                         const std::vector<TypeId> &node_mix_precision_datatype,
                         const std::unordered_map<size_t, std::vector<TypeId>> &kernel_support_datatypes,
                         std::unordered_map<size_t, std::vector<int>> *kernel_match_datatype_idx) {
  if (node_mix_precision_datatype_index.size() != node_mix_precision_datatype.size()) {
    MS_LOG(EXCEPTION) << "node datatype index size " << node_mix_precision_datatype_index.size() << " != datatype size "
                      << node_mix_precision_datatype.size();
  }
  MS_EXCEPTION_IF_NULL(kernel_match_datatype_idx);
  if (kernel_support_datatypes.size() != kernel_match_datatype_idx->size()) {
    MS_LOG(EXCEPTION) << "kernel datatype index size " << kernel_match_datatype_idx->size() << " != datatype size "
                      << kernel_support_datatypes.size();
  }
}

int RaiseDataTypePrecisionSelect(const std::vector<int> &node_mix_precision_datatype_index,
                                 const std::vector<TypeId> &node_mix_precision_datatype,
                                 const std::unordered_map<size_t, std::vector<TypeId>> &kernel_support_datatypes,
                                 std::unordered_map<size_t, std::vector<int>> *kernel_match_datatype_idx) {
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
          MS_LOG(EXCEPTION) << "kernel datatype index:%lu can not be found " << iter->first;
        }
        if (i >= find_iter->second.size()) {
          MS_LOG(EXCEPTION) << "node index " << i << "kernel datatype size " << find_iter->second.size();
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
        MS_LOG(EXCEPTION) << "node datatype index: " << i << " kernel support size " << datatype_indexes.size();
      }
      if (datatype_indexes[i] < node_mix_precision_datatype_index[i]) {
        iter = kernel_match_datatype_idx->erase(iter);
      } else {
        ++iter;
      }
    }
  }

  if (kernel_match_datatype_idx->size() >= 1) {
    return SizeToInt(kernel_match_datatype_idx->begin()->first);
  }
  return -1;
}

int GetMinReducePrecisionCountIndex(std::unordered_map<size_t, std::vector<int>> *kernel_match_datatype_idx,
                                    const std::unordered_map<size_t, size_t> &precision_reduce_count) {
  int selected_index = -1;
  size_t min_reduce_precision_count = kMaxCount;
  auto iter = kernel_match_datatype_idx->begin();
  while (iter != kernel_match_datatype_idx->end()) {
    auto find_iter = precision_reduce_count.find(iter->first);
    if (find_iter == precision_reduce_count.end()) {
      continue;
    }
    if (min_reduce_precision_count > find_iter->second) {
      selected_index = SizeToInt(iter->first);
      min_reduce_precision_count = find_iter->second;
    }
    ++iter;
  }
  return selected_index;
}

int RaiseOrReduceDataTypePrecisionSelect(
  const std::vector<int> &node_mix_precision_datatype_index, const std::vector<TypeId> &node_mix_precision_datatype,
  const std::unordered_map<size_t, std::vector<TypeId>> &kernel_support_datatypes,
  std::unordered_map<size_t, std::vector<int>> *kernel_match_datatype_idx) {
  CheckDataTypeInputs(node_mix_precision_datatype_index, node_mix_precision_datatype, kernel_support_datatypes,
                      kernel_match_datatype_idx);
  // reduce / raise
  std::unordered_map<size_t, size_t> precision_reduce_count;
  for (size_t i = 0; i < node_mix_precision_datatype_index.size(); ++i) {
    if (node_mix_precision_datatype[i] == kTypeUnknown) {
      continue;
    }
    auto iter = kernel_match_datatype_idx->begin();
    while (iter != kernel_match_datatype_idx->end()) {
      if (node_mix_precision_datatype_index[i] == kUnSupportMixedDataTypeIndex) {
        auto find_iter = kernel_support_datatypes.find(iter->first);
        if (find_iter == kernel_support_datatypes.end()) {
          MS_LOG(EXCEPTION) << "kernel datatype index:%lu can not be found " << iter->first;
        }
        if (i >= find_iter->second.size()) {
          MS_LOG(EXCEPTION) << "node index " << i << " >= kernel datatype size " << find_iter->second.size();
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
        MS_LOG(EXCEPTION) << "index " << i << "> kernel datatype indexes size " << datatype_indexes.size();
      }
      if (datatype_indexes[i] == kUnSupportMixedDataTypeIndex) {
        iter = kernel_match_datatype_idx->erase(iter);
      } else {
        if (datatype_indexes[i] < node_mix_precision_datatype_index[i]) {
          auto count_iter = precision_reduce_count.find(iter->first);
          if (count_iter != precision_reduce_count.end()) {
            count_iter->second++;
          } else {
            precision_reduce_count[iter->first] = 1;
          }
        }
        ++iter;
      }
    }
  }

  return GetMinReducePrecisionCountIndex(kernel_match_datatype_idx, precision_reduce_count);
}

void AddNodeAndKernelDataType(const CNodePtr &kernel_node, const kernel::KernelBuildInfo &kernel_build_info,
                              std::vector<int> *support_indexes, std::vector<TypeId> *node_mix_precision_datatype,
                              std::vector<TypeId> *support_datatypes,
                              std::vector<int> *node_mix_precision_datatype_index) {
  bool add_node_datatype_flag = false;
  if (node_mix_precision_datatype->size() == 0) {
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

int PrecisionReduce(const std::vector<int> &node_mix_precision_datatype_index,
                    const std::vector<TypeId> &node_mix_precision_datatype,
                    const std::unordered_map<size_t, std::vector<TypeId>> &kernel_support_datatype,
                    std::unordered_map<size_t, std::vector<int>> *kernel_match_datatype_idx, bool *precision_reduce) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  MS_EXCEPTION_IF_NULL(precision_reduce);
  std::unordered_map<size_t, std::vector<int>> kernel_match_datatype_idx_copy = *kernel_match_datatype_idx;
  // raise precision
  int selected_index = RaiseDataTypePrecisionSelect(node_mix_precision_datatype_index, node_mix_precision_datatype,
                                                    kernel_support_datatype, kernel_match_datatype_idx);
  if (selected_index == -1 && context_ptr->enable_reduce_precision()) {
    selected_index =
      RaiseOrReduceDataTypePrecisionSelect(node_mix_precision_datatype_index, node_mix_precision_datatype,
                                           kernel_support_datatype, &kernel_match_datatype_idx_copy);
    if (selected_index != -1) {
      *precision_reduce = true;
    }
  }
  return selected_index;
}

void SelectKernel(const CNodePtr &kernel_node, bool precision_reduce, const std::vector<TypeId> &node_datatype,
                  const std::shared_ptr<kernel::KernelBuildInfo> &selected_kernel_info_ptr) {
  MS_EXCEPTION_IF_NULL(selected_kernel_info_ptr);
  if (precision_reduce) {
    std::ostringstream datatype;
    size_t input_num = selected_kernel_info_ptr->GetInputNum();
    size_t i = 0;
    datatype << "(";
    for (; i < input_num && i < node_datatype.size(); ++i) {
      datatype << static_cast<int>(node_datatype[i]);
      if (i < input_num - 1) {
        datatype << ", ";
      }
    }
    datatype << ") -> (";
    for (; i < node_datatype.size(); ++i) {
      datatype << static_cast<int>(node_datatype[i]);
      if (i < node_datatype.size() - 1) {
        datatype << ", ";
      }
    }
    datatype << ")";
    MS_LOG(WARNING) << kernel_node->DebugString() << " reduce precision, node datatype: " << datatype.str()
                    << ", select kernel: %s" << selected_kernel_info_ptr->ToString();
  }
  AnfAlgo::SetSelectKernelBuildInfo(selected_kernel_info_ptr, kernel_node.get());
  // Set format and data type for input tensor.
  SetTensorDeviceInfo(*selected_kernel_info_ptr, kernel_node);
}
}  // namespace

void SelectKernelInfo(const CNodePtr &kernel_node) {
  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> kernel_info_list;
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel::KernelQuery(kernel_node, &kernel_info_list);
  std::vector<int> most_match_counts = {-1, -1, -1, -1};
  int selected_index = -1;
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool auto_mixed_precision = context_ptr->auto_mixed_precision_flag();
  std::unordered_map<size_t, std::vector<int>> kernel_match_datatype_idx;
  std::unordered_map<size_t, std::vector<TypeId>> kernel_support_datatype;
  std::vector<int> node_mix_precision_datatype_index;
  std::vector<TypeId> node_mix_precision_datatype;
  for (size_t info_index = 0; info_index < kernel_info_list.size(); ++info_index) {
    std::vector<int> cur_kernel_info_match_counts = {0, 0, 0, 0};
    auto kernel_build_info = *(kernel_info_list[info_index]);
    if (!IsValidKernelInfo(kernel_node, kernel_build_info)) {
      continue;
    }
    std::vector<int> support_indexes;
    std::vector<TypeId> support_datatypes;
    AddNodeAndKernelDataType(kernel_node, kernel_build_info, &support_indexes, &node_mix_precision_datatype,
                             &support_datatypes, &node_mix_precision_datatype_index);
    kernel_match_datatype_idx[info_index] = support_indexes;
    kernel_support_datatype[info_index] = support_datatypes;
    if (!auto_mixed_precision && !MatchInferOutputDataType(kernel_node, kernel_build_info)) {
      continue;
    }
    std::shared_ptr<kernel::KernelBuildInfo> kernel_info_ptr = kernel_info_list[info_index];
    UpdateCurMatchCounts(*kernel_info_ptr, kernel_node, &cur_kernel_info_match_counts);
    // Currently the selection policy is the match format count first, and then is datatype counts.
    if (PriorityChooseItem(cur_kernel_info_match_counts, &most_match_counts)) {
      selected_index = SizeToInt(info_index);
    }
  }

  bool precision_reduce = false;
  if (selected_index == -1) {
    selected_index = PrecisionReduce(node_mix_precision_datatype_index, node_mix_precision_datatype,
                                     kernel_support_datatype, &kernel_match_datatype_idx, &precision_reduce);
  }
  if (selected_index == -1) {
    MS_LOG(EXCEPTION) << kernel_node->DebugString() << "Cannot find valid kernel Info !";
  }
  auto index = IntToSize(selected_index);
  if (index >= kernel_info_list.size()) {
    MS_LOG(EXCEPTION) << "index outof range";
  }
  std::shared_ptr<kernel::KernelBuildInfo> selected_kernel_info_ptr = kernel_info_list[index];
  MS_EXCEPTION_IF_NULL(selected_kernel_info_ptr);
  SelectKernel(kernel_node, precision_reduce, node_mix_precision_datatype, selected_kernel_info_ptr);
}

bool CheckKernelAccuracySupported(const CNodePtr &kernel_node,
                                  const kernel::KernelBuildInfoPtr &new_kernel_build_info) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> kernel_info_list;
  kernel::KernelQuery(kernel_node, &kernel_info_list);
  auto result = std::find_if(kernel_info_list.begin(), kernel_info_list.end(),
                             [&new_kernel_build_info](const kernel::KernelBuildInfoPtr item) {
                               MS_EXCEPTION_IF_NULL(item);
                               return *item == *new_kernel_build_info;
                             });
  return result != kernel_info_list.end();
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
