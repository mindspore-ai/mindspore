/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/hal/device/kernel_select_ascend.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/kernel_info.h"
#include "ir/func_graph.h"
#include "kernel/common_utils.h"
#include "plugin/device/ascend/kernel/kernel_query.h"
#include "kernel/kernel_build_info.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace {
// sort format according the number of occurrences.
bool cmp_format_num(const std::pair<std::string, size_t> &a, const std::pair<std::string, size_t> &b) {
  if (a.second != b.second) {
    return a.second > b.second;
  } else if (a.first == kOpFormat_DEFAULT) {
    return a.second + 1 > b.second;
  } else if (b.first == kOpFormat_DEFAULT) {
    return a.second > b.second + 1;
  }
  return a.second > b.second;
}

TypeId GetPrimitivePrecision(const CNodePtr &cnode) {
  auto primitive = common::AnfAlgo::GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(primitive);

  TypeId except_type = kTypeUnknown;
  if (primitive->GetAttr(kAttrFixPrecision) != nullptr) {
    auto strExceptDtype = GetValue<std::string>(primitive->GetAttr(kAttrFixPrecision));
    if (strExceptDtype == "float16") {
      except_type = kNumberTypeFloat16;
    } else if (strExceptDtype == "float32") {
      except_type = kNumberTypeFloat32;
    } else {
      MS_LOG(EXCEPTION) << "The fix precision must be float16 or float32, but got" << strExceptDtype;
    }
  }

  return except_type;
}
}  // namespace

void ResetKernelBuildInfo(const CNodePtr &kernel_node) {
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  for (size_t input_index = 0; input_index < input_num; ++input_index) {
    auto input_kernel_node = common::AnfAlgo::GetInputNode(kernel_node, input_index);
    MS_EXCEPTION_IF_NULL(input_kernel_node);
    auto kernel_with_index = common::AnfAlgo::VisitKernel(input_kernel_node, 0);
    if (!kernel::IsWeightBoundary(kernel_with_index.first)) {
      continue;
    }
    // reset format and dtype.
    kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
    builder.SetOutputsFormat(std::vector<std::string>{kOpFormat_DEFAULT});
    builder.SetOutputsDeviceType(std::vector<TypeId>{kTypeUnknown});
    builder.SetOutputsKernelObjectType({kernel::KernelObjectType::TENSOR});
    AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), input_kernel_node.get());
  }
}

void UpdateKernelInfo(const std::vector<AnfNodePtr> &node_list) {
  for (size_t i = 0; i < node_list.size(); ++i) {
    // select nodes in subgraph.
    auto anf_node = node_list[i];
    MS_EXCEPTION_IF_NULL(anf_node);
    auto cnode = anf_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto fix_precision_type = GetPrimitivePrecision(cnode);
    if (fix_precision_type != kTypeUnknown) {
      std::vector<std::shared_ptr<kernel::KernelBuildInfo>> kernel_info_list;
      kernel::KernelQuery(cnode, &kernel_info_list, KernelType::AKG_KERNEL);

      for (auto &kernel_info : kernel_info_list) {
        // only math the first input
        if (kernel_info->GetInputDeviceType(0) == fix_precision_type &&
            kernel_info->GetInputFormat(0) == AnfAlgo::GetPrevNodeOutputFormat(cnode, 0) &&
            AnfAlgo::GetInputDeviceDataType(cnode, 0) != fix_precision_type) {
          auto selected_kernel_info_ptr = kernel_info;
          ResetKernelBuildInfo(cnode);
          AnfAlgo::SetSelectKernelBuildInfo(selected_kernel_info_ptr, cnode.get());
          SetTensorDeviceInfo(cnode);
          break;
        }
      }
    }
  }
}

bool CanConvertDefaultShapeToNZ(const ShapeVector &shape) {
  for (size_t i = 1; i <= shape.size(); ++i) {
    if (i > 2) {
      break;
    }
    if (LongToInt(shape[shape.size() - i]) != 1 && shape[shape.size() - i] % SizeToLong(kCubeSize) != 0) {
      return false;
    }
  }
  return true;
}

std::vector<int64_t> DefaultToFracNZAxis(const ShapeVector &ori_shape, const std::vector<int64_t> &axis) {
  std::vector<int64_t> frac_nz_axis = axis;
  auto shape_len = ori_shape.size();
  for (size_t i = 0; i < axis.size(); ++i) {
    auto axis_idx = (static_cast<size_t>(frac_nz_axis[i]) + shape_len) % shape_len;
    if (axis_idx == shape_len - kIndex1) {
      frac_nz_axis[i] = static_cast<int64_t>(axis_idx) - static_cast<int64_t>(kIndex1);
      frac_nz_axis.push_back(axis_idx + kIndex2);
    } else if (axis_idx == shape_len - kIndex2) {
      frac_nz_axis[i] = static_cast<int64_t>(axis_idx) + static_cast<int64_t>(kIndex1);
      frac_nz_axis.push_back(axis_idx + kIndex2);
    } else {
      frac_nz_axis[i] = static_cast<int64_t>(axis_idx);
    }
  }
  return frac_nz_axis;
}

void UpdateFracNZReduceOp(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto input_format = AnfAlgo::GetPrevNodeOutputFormat(cnode, 0);
  if (input_format == kOpFormat_FRAC_NZ) {
    // Clone primitive to modify it
    auto prim = GetCNodePrimitive(cnode);
    auto new_prim = std::make_shared<Primitive>(*prim);
    auto new_prim_node = NewValueNode(new_prim);
    cnode->set_input(0, new_prim_node);

    auto axis_value = new_prim->GetAttr(kAttrAxis);
    std::vector<int64_t> default_axis;
    if (axis_value->isa<ValueList>()) {
      auto value_list = dyn_cast<ValueList>(axis_value);
      for (const auto &item : value_list->value()) {
        if (item->isa<Int64Imm>()) {
          default_axis.push_back(GetValue<int64_t>(item));
        } else {
          MS_LOG(EXCEPTION) << "GetValue type should be int64";
        }
      }
    } else if (axis_value->isa<ValueTuple>()) {
      auto value_tuple = dyn_cast<ValueTuple>(axis_value);
      for (const auto &item : value_tuple->value()) {
        if (item->isa<Int64Imm>()) {
          default_axis.push_back(GetValue<int64_t>(item));
        } else {
          MS_LOG(EXCEPTION) << "GetValue type should be int64";
        }
      }
    } else {
      MS_LOG(ERROR) << "Axis attr type is not correct!";
    }
    auto infer_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(cnode, 0);
    std::vector<int64_t> frac_nz_axis = DefaultToFracNZAxis(infer_shape, default_axis);
    common::AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue<std::vector<int64_t>>(frac_nz_axis), cnode);
    auto output_shape = common::AnfAlgo::GetOutputInferShape(cnode, 0);
    if (output_shape.size() == 1) {
      common::AnfAlgo::SetNodeAttr(kAttrOutputDefault, MakeValue<bool>(true), cnode);
    }
  }
}

void GetDefaultFormat(const CNodePtr &kernel_node, std::string *default_format, bool *use_same_format) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  MS_EXCEPTION_IF_NULL(default_format);
  MS_EXCEPTION_IF_NULL(use_same_format);
  std::unordered_map<std::string, size_t> all_input_formats;
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  for (size_t i = 0; i < input_num; ++i) {
    auto input_kernel_node = common::AnfAlgo::VisitKernel(kernel_node->input(i + 1), 0).first;
    MS_EXCEPTION_IF_NULL(input_kernel_node);
    if (!input_kernel_node->isa<Parameter>()) {
      ++all_input_formats[AnfAlgo::GetPrevNodeOutputFormat(kernel_node, i)];
      continue;
    }
    auto para = input_kernel_node->cast<ParameterPtr>();
    if (AnfAlgo::GetOutputDeviceDataType(para, 0) != kTypeUnknown) {
      ++all_input_formats[AnfAlgo::GetOutputFormat(para, 0)];
      continue;
    }
    *use_same_format = false;
  }

  if (all_input_formats.empty()) {
    // all inputs are parameter.
    *default_format = kOpFormat_NC1HWC0;
  } else {
    std::vector<std::pair<std::string, size_t>> pairs;
    for (auto iter = all_input_formats.begin(); iter != all_input_formats.end(); ++iter) {
      (void)pairs.emplace_back(std::make_pair(iter->first, iter->second));
    }

    std::sort(pairs.begin(), pairs.end(), cmp_format_num);
    *default_format = pairs.begin()->first;
  }

  for (size_t i = 0; i < input_num; ++i) {
    auto input_kernel_node = common::AnfAlgo::VisitKernel(kernel_node->input(i + 1), 0).first;
    MS_EXCEPTION_IF_NULL(input_kernel_node);
    if (!input_kernel_node->isa<Parameter>() ||
        AnfAlgo::GetOutputDeviceDataType(input_kernel_node, 0) != kTypeUnknown) {
      continue;
    }
    auto weight_infer_shape = common::AnfAlgo::GetOutputInferShape(input_kernel_node, 0);
    if (weight_infer_shape.size() < kShape2dDims && *default_format == kOpFormat_FRAC_NZ) {
      *default_format = kOpFormat_DEFAULT;
      *use_same_format = true;
      break;
    }
  }
}

void UpdateInputsKernelInfo(const CNodePtr &kernel_node, const std::vector<AnfNodePtr> &input_list,
                            const std::string &default_format, bool use_same_format,
                            std::vector<std::string> *graph_input_format, std::vector<TypeId> *graph_input_type) {
  MS_EXCEPTION_IF_NULL(graph_input_format);
  MS_EXCEPTION_IF_NULL(graph_input_type);
  // We set same format to all inputs of graph kernel subgraph, and process this latter.
  // We set dtype to inputs of graph kernel subgraph same as infer dtypes.
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  for (size_t i = 0; i < input_num; ++i) {
    auto input_kernel_node = common::AnfAlgo::VisitKernel(kernel_node->input(i + 1), 0).first;
    MS_EXCEPTION_IF_NULL(input_kernel_node);
    if (use_same_format) {
      bool can_convert = true;
      if (default_format == kOpFormat_FRAC_NZ) {
        auto infer_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, i);
        if (!CanConvertDefaultShapeToNZ(infer_shape)) {
          MS_LOG(WARNING) << "Shape can't be converted to frac nz shape, so use default format instead";
          can_convert = false;
        }
      }
      if (can_convert) {
        (void)graph_input_format->emplace_back(default_format);
      } else {
        (void)graph_input_format->emplace_back(kOpFormat_DEFAULT);
      }
      graph_input_type->push_back(AnfAlgo::GetPrevNodeOutputDeviceDataType(kernel_node, i));
      continue;
    }

    if (!input_kernel_node->isa<Parameter>()) {
      // subgraph parameter from output of other nodes.
      graph_input_format->push_back(AnfAlgo::GetPrevNodeOutputFormat(kernel_node, i));
      graph_input_type->push_back(AnfAlgo::GetPrevNodeOutputDeviceDataType(kernel_node, i));
      continue;
    }

    auto para = input_kernel_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(para);
    if (AnfAlgo::GetOutputDeviceDataType(para, 0) != kTypeUnknown) {
      // parameter already selected.
      graph_input_format->push_back(AnfAlgo::GetOutputFormat(para, 0));
      graph_input_type->push_back(AnfAlgo::GetOutputDeviceDataType(para, 0));
      continue;
    }

    // weight parameter.
    graph_input_format->push_back(default_format);
    graph_input_type->push_back(common::AnfAlgo::GetOutputInferDataType(input_kernel_node, 0));
  }

  for (size_t i = 0; i < input_num; ++i) {
    kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
    std::vector<std::string> outputs_format = {(*graph_input_format)[i]};
    std::vector<TypeId> outputs_device_type = {(*graph_input_type)[i]};
    builder.SetOutputsFormat(outputs_format);
    builder.SetOutputsDeviceType(outputs_device_type);
    builder.SetOutputsKernelObjectType({kernel::KernelObjectType::TENSOR});
    AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), input_list[i].get());
  }
}

void UpdateEquivFormat(const std::vector<AnfNodePtr> &node_list, const FuncGraphPtr &func_graph,
                       const FuncGraphManagerPtr &mng) {
  MS_EXCEPTION_IF_NULL(mng);
  for (size_t i = 0; i < node_list.size(); ++i) {
    // select nodes in subgraph.
    auto anf_node = node_list[i];
    MS_EXCEPTION_IF_NULL(anf_node);
    auto cnode = anf_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    cnode->set_kernel_info(std::make_shared<device::KernelInfo>());
    (void)SelectKernelInfo(cnode, KernelType::AKG_KERNEL);
    // Update ReduceSum
    if (!IsPrimitiveCNode(cnode, prim::kPrimReduceSum)) {
      continue;
    }
    UpdateFracNZReduceOp(cnode);
    // If ReduceSum's output is 1d and not Default format, convert it to Default format
    auto out_format = AnfAlgo::GetOutputFormat(cnode, 0);
    if (out_format == kOpFormat_DEFAULT || !common::AnfAlgo::HasNodeAttr(kAttrOutputDefault, cnode)) {
      continue;
    }
    // Insert EquivFormat node, then select kernel info again
    std::vector<AnfNodePtr> trans_inputs;
    trans_inputs.push_back(NewValueNode(prim::kPrimEquivFormat));
    trans_inputs.push_back(cnode);
    CNodePtr trans_node = func_graph->NewCNode(trans_inputs);
    common::AnfAlgo::SetOutputInferTypeAndShape({common::AnfAlgo::GetPrevNodeOutputInferDataType(cnode, 0)},
                                                {common::AnfAlgo::GetOutputInferShape(cnode, 0)}, trans_node.get());
    common::AnfAlgo::SetNodeAttr(kAttrInputNames, MakeValue<std::vector<std::string>>({"x"}), trans_node);

    if (trans_node->kernel_info() == nullptr) {
      trans_node->set_kernel_info(std::make_shared<device::KernelInfo>());
    }
    (void)SelectKernelInfo(trans_node, KernelType::AKG_KERNEL);
    (void)mng->Replace(cnode, trans_node);
  }
}

void CheckFormatsAndDtypes(const CNodePtr &kernel_node, const std::vector<AnfNodePtr> &input_list,
                           const FuncGraphManagerPtr &mng, const std::string &default_format,
                           std::vector<std::string> *graph_input_format, std::vector<TypeId> *graph_input_type,
                           std::vector<bool> *need_update) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  MS_EXCEPTION_IF_NULL(mng);
  MS_EXCEPTION_IF_NULL(graph_input_format);
  MS_EXCEPTION_IF_NULL(graph_input_type);
  MS_EXCEPTION_IF_NULL(need_update);
  // check graph input format and dtype use inner ops.
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  if (graph_input_format->size() != input_num || graph_input_type->size() != input_num ||
      need_update->size() != input_num) {
    MS_LOG(EXCEPTION) << "Graph input format size is not equal to input num of cnode[" << kernel_node->DebugString()
                      << "], [" << graph_input_format->size() << "] != [" << input_num << "]";
  }
  auto &node_users = mng->node_users();
  for (size_t i = 0; i < input_num; ++i) {
    auto &input = input_list[i];
    auto iter = node_users.find(input);
    if (iter == node_users.end() || iter->second.empty()) {
      continue;
    }
    for (auto &node_user : iter->second) {
      if (node_user.first->kernel_info() == nullptr || !node_user.first->kernel_info()->has_build_info()) {
        // maybe not a real kernel.
        continue;
      }
      auto user_format = AnfAlgo::GetInputFormat(node_user.first, IntToSize(node_user.second - 1));
      if (user_format != (*graph_input_format)[i]) {
        MS_LOG(WARNING) << "Users of input: [" << i << "][" << input->DebugString() << " of ["
                        << kernel_node->DebugString()
                        << "] selected different format. we use default: " << default_format;
        (*graph_input_format)[i] = default_format;
        (*need_update)[i] = true;
      }

      if (kernel_node->input(i + 1)->isa<Parameter>() ||
          AnfAlgo::GetInputDeviceDataType(node_user.first, IntToSize(node_user.second - 1)) == (*graph_input_type)[i]) {
        continue;
      }

      TypeId default_dtype = common::AnfAlgo::GetOutputInferDataType(input, 0);
      MS_LOG(WARNING) << "Users of input: [" << i << "][" << input->DebugString() << " of ["
                      << kernel_node->DebugString()
                      << "] selected different dtype. we use default: " << TypeIdLabel(default_dtype);
      (*graph_input_type)[i] = default_dtype;
      (*need_update)[i] = true;
    }
  }
}

void UpdateFormatsAndDtypes(const CNodePtr &kernel_node, const std::vector<AnfNodePtr> &node_list,
                            const std::vector<AnfNodePtr> &input_list, const std::vector<bool> &need_update,
                            const std::vector<std::string> &graph_input_format,
                            const std::vector<TypeId> &graph_input_type) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  // update graph input format and dtype use inner ops.
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  if (graph_input_format.size() != input_num || graph_input_type.size() != input_num ||
      need_update.size() != input_num) {
    MS_LOG(EXCEPTION) << "Graph input format size is not equal to input num of cnode[" << kernel_node->DebugString()
                      << "], [" << graph_input_format.size() << "] != [" << input_num << "]";
  }
  for (size_t i = 0; i < input_num; ++i) {
    if (!need_update[i]) {
      continue;
    }

    MS_LOG(DEBUG) << "Update input format: " << i << " of: [" << kernel_node->DebugString()
                  << "] to: " << graph_input_format[i];
    MS_LOG(DEBUG) << "Update input dtype: " << i << " of: [" << kernel_node->DebugString()
                  << "] to: " << TypeIdLabel(graph_input_type[i]);
    kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
    std::vector<std::string> outputs_format = {graph_input_format[i]};
    std::vector<TypeId> outputs_device_type = {graph_input_type[i]};
    builder.SetOutputsFormat(outputs_format);
    builder.SetOutputsDeviceType(outputs_device_type);
    builder.SetOutputsKernelObjectType({kernel::KernelObjectType::TENSOR});
    AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), input_list[i].get());
  }

  ResetKernelBuildInfo(kernel_node);
  // select nodes in subgraph again.
  for (size_t i = 0; i < node_list.size(); ++i) {
    auto anf_node = node_list[i];
    MS_EXCEPTION_IF_NULL(anf_node);
    auto cnode = anf_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
    size_t cnode_input_num = common::AnfAlgo::GetInputTensorNum(cnode);
    for (size_t j = 0; j < cnode_input_num; ++j) {
      auto input_node = cnode->input(j + 1);
      MS_EXCEPTION_IF_NULL(input_node);
      if (!IsValueNode<tensor::Tensor>(input_node)) {
        continue;
      }
      // reset format and dtype of const tensor.
      builder.SetOutputsFormat(std::vector<std::string>{kOpFormat_DEFAULT});
      builder.SetOutputsDeviceType(std::vector<TypeId>{kTypeUnknown});
      AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), input_node.get());
    }
    (void)SelectKernelInfo(node_list[i]->cast<CNodePtr>(), KernelType::AKG_KERNEL);
  }
}

void SetGraphKernelInfo(const CNodePtr &kernel_node, const std::vector<std::pair<AnfNodePtr, size_t>> &output_index,
                        const std::vector<std::string> &graph_input_format,
                        const std::vector<TypeId> &graph_input_type) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::vector<std::string> graph_output_format;
  std::vector<TypeId> graph_output_type;
  std::vector<kernel::KernelObjectType> graph_output_object_type;
  for (size_t i = 0; i < output_index.size(); ++i) {
    auto const &output = output_index[i];
    graph_output_format.push_back(AnfAlgo::GetOutputFormat(output.first, output.second));
    TypeId output_type(kTypeUnknown);
    if (output.first->isa<CNode>()) {
      output_type = common::AnfAlgo::GetCNodeOutputPrecision(output.first);
    }
    if (output_type == kTypeUnknown) {
      output_type = AnfAlgo::GetOutputDeviceDataType(output.first, output.second);
    }
    graph_output_type.push_back(output_type);
    graph_output_object_type.push_back(kernel::KernelObjectType::TENSOR);
  }

  std::vector<kernel::KernelObjectType> graph_input_object_type;
  for (size_t i = 0; i < graph_input_type.size(); ++i) {
    graph_input_object_type.push_back(kernel::KernelObjectType::TENSOR);
  }

  kernel::KernelBuildInfo::KernelBuildInfoBuilder graph_info_builder;
  graph_info_builder.SetInputsFormat(graph_input_format);
  graph_info_builder.SetInputsDeviceType(graph_input_type);
  graph_info_builder.SetInputsKernelObjectType(graph_input_object_type);
  graph_info_builder.SetOutputsFormat(graph_output_format);
  graph_info_builder.SetOutputsDeviceType(graph_output_type);
  graph_info_builder.SetOutputsKernelObjectType(graph_output_object_type);
  graph_info_builder.SetProcessor(kernel::Processor::AICORE);
  graph_info_builder.SetKernelType(KernelType::AKG_KERNEL);
  graph_info_builder.SetFusionType(kernel::kPatternOpaque);
  auto graph_selected_info = graph_info_builder.Build();
  MS_EXCEPTION_IF_NULL(graph_selected_info);
  AnfAlgo::SetSelectKernelBuildInfo(graph_selected_info, kernel_node.get());
  SetTensorDeviceInfo(kernel_node);
}

void SelectGraphKernelInfo(const CNodePtr &kernel_node, const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  MS_EXCEPTION_IF_NULL(func_graph);

  // collect input info of funcgraph
  std::vector<AnfNodePtr> node_list;
  std::vector<AnfNodePtr> input_list;
  std::vector<AnfNodePtr> output_list;
  kernel::GetValidKernelNodes(func_graph, &node_list, &input_list, &output_list);
  if (input_list.size() != kernel_node->inputs().size() - 1) {
    MS_EXCEPTION(ArgumentError) << "Input num of funcgraph[" << func_graph->ToString() << "] not equal input of cnode["
                                << kernel_node->DebugString() << "], [%" << input_list.size() << "] != ["
                                << kernel_node->inputs().size() << "]";
  }

  std::string default_format;
  bool use_same_format = true;
  GetDefaultFormat(kernel_node, &default_format, &use_same_format);
  MS_LOG(DEBUG) << "GraphKernel[" << func_graph->ToString() << "] use same input format[" << default_format
                << "] for ParameterWeight.";

  std::vector<std::string> graph_input_format;
  std::vector<TypeId> graph_input_type;
  UpdateInputsKernelInfo(kernel_node, input_list, default_format, use_same_format, &graph_input_format,
                         &graph_input_type);

  auto mng = func_graph->manager();
  if (mng == nullptr) {
    mng = Manage(func_graph, true);
  }
  UpdateEquivFormat(node_list, func_graph, mng);
  node_list.clear();
  input_list.clear();
  output_list.clear();
  kernel::GetValidKernelNodes(func_graph, &node_list, &input_list, &output_list);

  // update graph input format and dtype use inner ops.
  std::vector<bool> need_update(common::AnfAlgo::GetInputTensorNum(kernel_node), false);
  CheckFormatsAndDtypes(kernel_node, input_list, mng, default_format, &graph_input_format, &graph_input_type,
                        &need_update);
  UpdateFormatsAndDtypes(kernel_node, node_list, input_list, need_update, graph_input_format, graph_input_type);

  // set fix_precision for kernel when the me prim has fix_precision attr
  UpdateKernelInfo(node_list);

  auto output_index = kernel::GetOutputIndex(node_list, input_list, output_list);
  SetGraphKernelInfo(kernel_node, output_index, graph_input_format, graph_input_type);
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
