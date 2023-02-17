/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/hal/device/kernel_info_setter.h"
#include <algorithm>
#include <memory>
#include <tuple>
#include <string>
#include <set>
#include "kernel/common_utils.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "kernel/kernel.h"
#include "kernel/kernel_build_info.h"
#include "kernel/oplib/opinfo.h"
#include "kernel/oplib/oplib.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/device/gpu/kernel/custom/custom_aot_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
#include "utils/ms_context.h"
#include "utils/ms_utils.h"
#include "include/common/utils/utils.h"
#include "mindspore/core/ops/core_ops.h"
#include "mindspore/core/ops/op_name.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"

namespace mindspore {
namespace device {
namespace gpu {
using AnfAlgo = mindspore::session::AnfRuntimeAlgorithm;
using mindspore::kernel::KernelBuildInfo;
namespace {
static const std::set<std::string> kVmapGPUWhiteList = {kUnsortedSegmentSumOpName,
                                                        kUnsortedSegmentProdOpName,
                                                        kUniqueWithPadOpName,
                                                        kMaskedFillOpName,
                                                        kDataFormatDimMapOpName,
                                                        kInstanceNorm,
                                                        kInstanceNormGradOpName,
                                                        kRandomChoiceWithMaskOpName,
                                                        kAdamOpName,
                                                        kSplitOpName,
                                                        kApplyAdagradDAOpName,
                                                        kApplyRMSPropOpName,
                                                        kApplyCenteredRMSPropOpName,
                                                        kRandomShuffleOpName,
                                                        kApplyAdamWithAmsgradOpName,
                                                        kApplyProximalAdagradOpName,
                                                        prim::kMatrixBandPart,
                                                        prim::kDiag,
                                                        prim::kSparseSegmentMean};

kernel::OpImplyType GetImplyType(KernelType kernel_type) {
  kernel::OpImplyType imply_type =
    kernel_type == KernelType::GPU_KERNEL ? kernel::OpImplyType::kImplyGPU : kernel::OpImplyType::kImplyAKG;
  return imply_type;
}

bool CheckKernelInfo(const std::shared_ptr<KernelBuildInfo> &alternative_kernel_info,
                     const std::shared_ptr<KernelBuildInfo> &selected_kernel_info, bool match_none = false) {
  MS_EXCEPTION_IF_NULL(selected_kernel_info);
  MS_EXCEPTION_IF_NULL(alternative_kernel_info);
  size_t selected_input_num = selected_kernel_info->GetInputNum();
  size_t alternative_input_num = alternative_kernel_info->GetInputNum();
  if (selected_input_num != alternative_input_num) {
    return false;
  }
  for (size_t i = 0; i < selected_input_num; i++) {
    auto format = alternative_kernel_info->GetInputFormat(i);
    if (selected_kernel_info->GetInputFormat(i) != format && (!match_none || !format.empty())) {
      return false;
    }
    auto type = alternative_kernel_info->GetInputDeviceType(i);
    if (selected_kernel_info->GetInputDeviceType(i) != type && (!match_none || type != TypeId::kMetaTypeNone)) {
      return false;
    }
  }

  size_t selected_output_num = selected_kernel_info->GetOutputNum();
  size_t alternative_output_num = alternative_kernel_info->GetOutputNum();
  if (selected_output_num != alternative_output_num) {
    return false;
  }
  for (size_t i = 0; i < selected_output_num; i++) {
    auto format = alternative_kernel_info->GetOutputFormat(i);
    if (selected_kernel_info->GetOutputFormat(i) != format && (!match_none || !format.empty())) {
      return false;
    }
    auto type = alternative_kernel_info->GetOutputDeviceType(i);
    if (selected_kernel_info->GetOutputDeviceType(i) != type && (!match_none || type != TypeId::kMetaTypeNone)) {
      return false;
    }
  }
  return true;
}

std::string GetSupportedTypesStr(const CNodePtr &kernel_node, KernelType kernel_type) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::string supported_type_lists;
  // Custom op gets reg info from OpLib instead of NativeGpuKernelMod.
  if (!IsPrimitiveCNode(kernel_node, prim::kPrimCustom)) {
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    if (kernel::Factory<kernel::NativeGpuKernelMod>::Instance().IsRegistered(kernel_name)) {
      auto kernel_attr_list = kernel::NativeGpuKernelMod::GetGpuSupportedList(kernel_name);
      if (!kernel_attr_list.empty()) {
        for (size_t attr_index = 0; attr_index < kernel_attr_list.size(); ++attr_index) {
          std::string type_list = "input[";
          auto attr = kernel_attr_list[attr_index];
          for (size_t input_index = 0; input_index < attr.GetInputSize(); ++input_index) {
            type_list = type_list + TypeIdToString(attr.GetInputAttr(input_index).dtype) +
                        ((input_index == (attr.GetInputSize() - 1)) ? "" : " ");
          }
          type_list = type_list + "], output[";
          for (size_t input_index = 0; input_index < attr.GetOutputSize(); ++input_index) {
            type_list = type_list + TypeIdToString(attr.GetOutputAttr(input_index).dtype) +
                        ((input_index == (attr.GetOutputSize() - 1)) ? "" : " ");
          }
          supported_type_lists = supported_type_lists + type_list + "]; ";
        }

        return supported_type_lists;
      }
    } else {
      supported_type_lists =
        kernel::NativeGpuKernelModFactory::GetInstance().SupportedTypeList(common::AnfAlgo::GetCNodeName(kernel_node));
      if (!supported_type_lists.empty()) {
        return supported_type_lists;
      }
    }
  }

  std::vector<std::shared_ptr<KernelBuildInfo>> kernel_info_list;
  std::string op_name = common::AnfAlgo::GetCNodeName(kernel_node);
  kernel::OpImplyType imply_type = GetImplyType(kernel_type);
  auto op_info_ptr = mindspore::kernel::OpLib::FindOp(op_name, imply_type);
  if (op_info_ptr == nullptr) {
    return supported_type_lists;
  }
  (void)ParseMetadata(kernel_node, op_info_ptr, kernel::Processor::CUDA, &kernel_info_list);
  for (size_t i = 0; i < kernel_info_list.size(); i++) {
    auto supported_akg_type = kernel_info_list[i]->GetAllInputDeviceTypes();
    auto supported_akg_type_out = kernel_info_list[i]->GetAllOutputDeviceTypes();
    std::string supported_akg_type_list = "input[";
    for (auto type : supported_akg_type) {
      supported_akg_type_list = supported_akg_type_list + TypeIdToString(type) + " ";
    }
    supported_type_lists = supported_type_lists + supported_akg_type_list + "], output[";
    supported_akg_type_list.clear();
    for (auto type : supported_akg_type_out) {
      supported_akg_type_list = supported_akg_type_list + TypeIdToString(type) + " ";
    }
    supported_type_lists = supported_type_lists + supported_akg_type_list + "]; ";
  }

  return supported_type_lists;
}

bool SelectAkgKernel(const CNodePtr &kernel_node, const std::shared_ptr<KernelBuildInfo> &selected_kernel_info) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  MS_EXCEPTION_IF_NULL(selected_kernel_info);
  std::vector<std::shared_ptr<KernelBuildInfo>> kernel_info_list;
  if (common::AnfAlgo::IsNodeInGraphKernel(kernel_node)) {
    // The op_info in OpLib is only used for basic ops,
    // we don't care it in GraphKernel.
    return true;
  }

  std::string op_name = common::AnfAlgo::GetCNodeName(kernel_node);

  auto op_info_ptr = mindspore::kernel::OpLib::FindOp(op_name, kernel::OpImplyType::kImplyAKG);
  if (op_info_ptr == nullptr) {
    MS_LOG(DEBUG) << "Not find op[" << op_name << "] in akg";
    return false;
  }
  if (!ParseMetadata(kernel_node, op_info_ptr, kernel::Processor::CUDA, &kernel_info_list)) {
    MS_LOG(EXCEPTION) << "Parsed metadata of op[" << op_name << "] failed.";
  }
  if (kernel_info_list.empty()) {
    MS_LOG(EXCEPTION) << "Akg dose not has metadata of op[" << op_name << "].";
  }

  bool match = std::any_of(kernel_info_list.begin(), kernel_info_list.end(),
                           [&](const std::shared_ptr<KernelBuildInfo> &alternative_kernel_info) {
                             return CheckKernelInfo(alternative_kernel_info, selected_kernel_info);
                           });
  if (!match) {
    MS_LOG(ERROR) << "Not find op[" << op_name << "] which both match data type and format in akg";
    return false;
  }
  return true;
}

bool SelectCustomKernel(const CNodePtr &kernel_node, const std::shared_ptr<KernelBuildInfo> &selected_kernel_info,
                        KernelType *kernel_type) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  MS_EXCEPTION_IF_NULL(selected_kernel_info);
  MS_EXCEPTION_IF_NULL(kernel_type);
  std::string op_name = common::AnfAlgo::GetCNodeName(kernel_node);
  // Custom op's kernel type can be one of [GPU_KERNEL, AKG_KERNEL] on GPU
  auto func_type = common::AnfAlgo::GetNodeAttr<std::string>(kernel_node, kAttrFuncType);
  if (func_type == kCustomTypeAOT) {
    *kernel_type = KernelType::GPU_KERNEL;
    if (!kernel::Factory<kernel::NativeGpuKernelMod>::Instance().IsRegistered(op_name)) {
      kernel::Factory<kernel::NativeGpuKernelMod>::Instance().Register(
        op_name, []() { return std::make_shared<kernel::CustomAOTGpuKernelMod>(); });
    }
  } else if (IsOneOfCustomAkgType(func_type)) {
    *kernel_type = KernelType::AKG_KERNEL;
  } else {
    MS_LOG(EXCEPTION) << "Unsupported func type [" << func_type << "] for Custom op [" << op_name << "] on GPU";
  }
  kernel::OpImplyType imply_type = GetImplyType(*kernel_type);
  auto op_info_ptr = mindspore::kernel::OpLib::FindOp(op_name, imply_type);
  // If Custom op has not set reg info,
  // or the no info about inputs in reg info(the case of undetermined input size),
  // then infer info from inputs
  if (op_info_ptr == nullptr || op_info_ptr->inputs_ptr().size() == 0) {
    MS_LOG(WARNING) << "Not find operator information for op[" << op_name
                    << "]. Infer operator information from inputs.";
    return true;
  }
  std::vector<std::shared_ptr<KernelBuildInfo>> kernel_info_list;
  if (!ParseMetadata(kernel_node, op_info_ptr, kernel::Processor::CUDA, &kernel_info_list)) {
    MS_LOG(EXCEPTION) << "Parsed metadata of op[" << op_name << "] failed.";
  }
  if (kernel_info_list.empty()) {
    MS_LOG(EXCEPTION) << "Not find valid metadata of op[" << op_name << "].";
  }
  bool match = std::any_of(kernel_info_list.begin(), kernel_info_list.end(),
                           [&](const std::shared_ptr<KernelBuildInfo> &alternative_kernel_info) {
                             return CheckKernelInfo(alternative_kernel_info, selected_kernel_info, true);
                           });
  if (!match) {
    MS_LOG(ERROR) << "Not find op[" << op_name << "] which both match data type and format.";
    return false;
  }
  return true;
}

void SetTensorDeviceInfo(const kernel::KernelBuildInfo &selected_kernel_info, const CNodePtr &kernel_node,
                         const std::vector<std::tuple<size_t, TypeId, TypeId>> &input_reduce_detail) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  for (size_t input_index = 0; input_index < input_num; ++input_index) {
    auto input_kernel_node = kernel_node->input(input_index + 1);
    MS_EXCEPTION_IF_NULL(input_kernel_node);
    auto input_with_index = common::AnfAlgo::VisitKernel(input_kernel_node, 0);
    MS_EXCEPTION_IF_NULL(input_with_index.first);
    auto real_input_node = input_with_index.first;
    if (!real_input_node->isa<Parameter>()) {
      continue;
    }
    std::shared_ptr<kernel::KernelBuildInfo::KernelBuildInfoBuilder> builder =
      std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();

    auto param = real_input_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param);
    if (!common::AnfAlgo::IsParameterWeight(param)) {
      std::vector<std::string> output_format = {kOpFormat_DEFAULT};
      builder->SetOutputsFormat(output_format);
      std::vector<TypeId> output_type = {common::AnfAlgo::GetOutputInferDataType(real_input_node, 0)};
      builder->SetOutputsDeviceType(output_type);
      AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), real_input_node.get());
      continue;
    }
    if ((AnfAlgo::GetOutputDeviceDataType(real_input_node, 0) == kTypeUnknown) ||
        (common::AnfAlgo::GetCNodeName(kernel_node) == "ApplyMomentum")) {
      std::vector<std::string> output_format = {selected_kernel_info.GetInputFormat(input_index)};
      builder->SetOutputsFormat(output_format);
      std::vector<TypeId> output_type;
      auto reduce_flag = kernel::NativeGpuKernelModFactory::GetInstance().reduce_flag_;
      if (std::find(reduce_flag.first.begin(), reduce_flag.first.end(), input_index) != reduce_flag.first.end()) {
        output_type = {reduce_flag.second};
      } else {
        auto iter = std::find_if(input_reduce_detail.begin(), input_reduce_detail.end(),
                                 [input_index](const std::tuple<size_t, TypeId, TypeId> &reduce_detail) {
                                   return std::get<0>(reduce_detail) == input_index;
                                 });
        if (iter != input_reduce_detail.end()) {
          output_type = {std::get<1>(*iter)};
        } else {
          output_type = {selected_kernel_info.GetInputDeviceType(input_index)};
        }
      }
      builder->SetOutputsDeviceType(output_type);
      AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), real_input_node.get());
    }
  }
  kernel::NativeGpuKernelModFactory::GetInstance().reduce_flag_.first.clear();
}

void TransformFormatPosition(std::vector<size_t> *format_position, size_t position_num) {
  MS_EXCEPTION_IF_NULL(format_position);
  if (format_position->size() == 0) {
    return;
  }

  // If the inserted position is kAllPositions, then insert all the positions.
  if ((*format_position)[0] == kAllPositions) {
    format_position->clear();
    for (size_t index = 0; index < position_num; index++) {
      format_position->push_back(index);
    }
  }
}

bool IsNeedProcessFormatInfo(const CNodePtr &kernel_node, const std::vector<TypeId> &inputs_type) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  if (!FormatTransformChecker::GetInstance().format_transform()) {
    return false;
  }
  if (!AnfUtils::IsRealCNodeKernel(kernel_node)) {
    return false;
  }
  auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
  auto iter = kKernelFormatPositionMap.find(kernel_name);
  if (iter == kKernelFormatPositionMap.end()) {
    return false;
  }
  if (inputs_type.size() == 0) {
    return false;
  }

  auto inputs_format_position = iter->second.first;
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  TransformFormatPosition(&inputs_format_position, input_num);
  for (const auto &input_format_position : inputs_format_position) {
    auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, input_format_position);
    // Only support the transformer between NCHW and NHWC, so need the shape is 4 dimension.
    if (input_shape.size() != kFormatTransformDimension) {
      return false;
    }
  }

  auto outputs_format_position = iter->second.second;
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  TransformFormatPosition(&outputs_format_position, output_num);
  for (const auto &output_format_position : outputs_format_position) {
    auto output_shape = common::AnfAlgo::GetOutputInferShape(kernel_node, output_format_position);
    // Only support the transformer between NCHW and NHWC, so need the shape is 4 dimension.
    if (output_shape.size() != kFormatTransformDimension) {
      return false;
    }
  }
  return true;
}

void UpdateKernelFormatInfo(const CNodePtr &kernel_node, const std::vector<TypeId> &inputs_type,
                            std::vector<std::string> *inputs_format, std::vector<std::string> *outputs_format,
                            std::string *origin_data_format) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  MS_EXCEPTION_IF_NULL(inputs_format);
  MS_EXCEPTION_IF_NULL(outputs_format);
  auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
  auto iter = kKernelFormatPositionMap.find(kernel_name);
  if (iter == kKernelFormatPositionMap.end()) {
    return;
  }
  auto cal_format = (inputs_type[0] == kNumberTypeFloat16) ? kOpFormat_NHWC : kOpFormat_NCHW;
  MS_LOG(DEBUG) << "Kernel node: " << kernel_node->fullname_with_scope() << ", format: " << cal_format;
  auto inputs_format_position = iter->second.first;
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  TransformFormatPosition(&inputs_format_position, input_num);
  for (const auto &input_format_position : inputs_format_position) {
    if (input_format_position >= inputs_format->size()) {
      MS_LOG(EXCEPTION) << "The position [" << input_format_position << "] is out of range of the input size ["
                        << inputs_format->size() << "] #kernel_node [" << kernel_node->fullname_with_scope() << "]";
    }
    (*inputs_format)[input_format_position] = cal_format;
  }

  auto outputs_format_position = iter->second.second;
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  TransformFormatPosition(&outputs_format_position, output_num);
  for (const auto &output_format_position : outputs_format_position) {
    if (output_format_position >= outputs_format->size()) {
      MS_LOG(EXCEPTION) << "The position [" << output_format_position << "] is out of range of the output size ["
                        << outputs_format->size() << "] #kernel_node [" << kernel_node->fullname_with_scope() << "]";
    }
    (*outputs_format)[output_format_position] = cal_format;
  }
  auto prim = common::AnfAlgo::GetCNodePrimitive(kernel_node);
  MS_EXCEPTION_IF_NULL(prim);
  if (prim->HasAttr("format")) {
    *origin_data_format = common::AnfAlgo::GetNodeAttr<std::string>(kernel_node, "format");
  }
}

void SetGraphKernelInfo(const CNodePtr &kernel_node, const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> node_list, input_list, output_list;
  kernel::GetValidKernelNodes(func_graph, &node_list, &input_list, &output_list);

  std::vector<std::string> graph_input_format;
  std::vector<TypeId> graph_input_type;
  // set graph kernel inputs kernel info.
  for (size_t i = 0; i < input_list.size(); ++i) {
    kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
    std::vector<std::string> outputs_format = {kOpFormat_DEFAULT};
    std::vector<TypeId> outputs_device_type = {common::AnfAlgo::GetOutputInferDataType(input_list[i], 0)};
    graph_input_format.push_back(kOpFormat_DEFAULT);
    graph_input_type.push_back(common::AnfAlgo::GetOutputInferDataType(input_list[i], 0));
    builder.SetOutputsFormat(outputs_format);
    builder.SetOutputsDeviceType(outputs_device_type);
    AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), input_list[i].get());
  }

  // set graph kernel innner nodes kernel info.
  auto kernel_info_setter = GraphKernelInfoManager::Instance().GetGraphKernelInfo(kGPUDevice);
  MS_EXCEPTION_IF_NULL(kernel_info_setter);
  for (size_t i = 0; i < node_list.size(); ++i) {
    const auto &anf_node = node_list[i];
    MS_EXCEPTION_IF_NULL(anf_node);
    auto cnode = anf_node->cast<CNodePtr>();
    cnode->set_kernel_info(std::make_shared<device::KernelInfo>());
    kernel_info_setter->SetKernelInfo(cnode, KernelType::AKG_KERNEL);
  }

  // set graph kernel node kernel info.
  auto mng = func_graph->manager();
  if (mng == nullptr) {
    mng = Manage(func_graph, true);
    func_graph->set_manager(mng);
  }
  auto output_index = kernel::GetOutputIndex(node_list, input_list, output_list);
  std::vector<std::string> graph_output_format;
  std::vector<TypeId> graph_output_type;
  for (size_t i = 0; i < output_index.size(); ++i) {
    auto const &output = output_index[i];
    graph_output_format.push_back(AnfAlgo::GetOutputFormat(output.first, output.second));
    graph_output_type.push_back(AnfAlgo::GetOutputDeviceDataType(output.first, output.second));
  }

  kernel::KernelBuildInfo::KernelBuildInfoBuilder graph_info_builder;
  graph_info_builder.SetInputsFormat(graph_input_format);
  graph_info_builder.SetInputsDeviceType(graph_input_type);
  graph_info_builder.SetOutputsFormat(graph_output_format);
  graph_info_builder.SetOutputsDeviceType(graph_output_type);
  graph_info_builder.SetProcessor(kernel::Processor::CUDA);
  graph_info_builder.SetKernelType(KernelType::AKG_KERNEL);
  graph_info_builder.SetFusionType(kernel::kPatternOpaque);
  auto graph_selected_info = graph_info_builder.Build();
  MS_EXCEPTION_IF_NULL(graph_selected_info);
  AnfAlgo::SetSelectKernelBuildInfo(graph_selected_info, kernel_node.get());
  SetTensorDeviceInfo(*graph_selected_info, kernel_node, {});
}

std::pair<std::string, ExceptionType> PrintUnsupportedTypeWarning(const CNodePtr &kernel_node,
                                                                  const std::vector<TypeId> &inputs_type,
                                                                  const std::vector<TypeId> &outputs_type,
                                                                  KernelType kernel_type) {
  auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
  std::string build_type = "input[";
  std::for_each(std::begin(inputs_type), std::end(inputs_type),
                [&build_type](auto i) { build_type += TypeIdToString(i) + " "; });
  build_type += "] output[";
  std::for_each(std::begin(outputs_type), std::end(outputs_type),
                [&build_type](auto i) { build_type += TypeIdToString(i) + " "; });
  build_type += "]";
  auto supported_type_lists = GetSupportedTypesStr(kernel_node, kernel_type);
  std::stringstream ss;
  ExceptionType etype;
  if (supported_type_lists.empty()) {
    ss << "Unsupported op [" << kernel_name << "] on GPU, Please confirm whether the device target setting is correct, "
       << "or refer to 'mindspore.ops' at https://www.mindspore.cn to query the operator support list.";
    etype = NotSupportError;
  } else {
    ss << "Select GPU operator[" << kernel_name << "] fail! Unsupported data type!\nThe supported data types are "
       << supported_type_lists << ", but get " << build_type;
    etype = TypeError;
  }
  return {ss.str(), etype};
}
}  // namespace

void FormatTransformChecker::CheckSupportFormatTransform(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<bool>(MS_CTX_DISABLE_FORMAT_TRANSFORM)) {
    MS_LOG(INFO) << "Disable the automatic format transform function.";
    format_transform_ = false;
    return;
  }
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
    format_transform_ = false;
    return;
  }
  // Format transform will case the different infer shape and device shape, so the dynamic shape graph can't be support.
  if (kernel_graph->is_dynamic_shape()) {
    MS_LOG(INFO) << "Dynamic shape doesn't support the automatic format transform function.";
    format_transform_ = false;
    return;
  }

  // TensorCore can be used only in Volta or newer devices.
  const int marjor_sm = GET_MAJOR_SM;
  if (marjor_sm < RECOMMEND_SM) {
    format_transform_ = false;
    return;
  }
  auto kernels = kernel_graph->execution_order();
  size_t conv_cnt = 0;
  size_t bn_cnt = 0;
  for (const auto &kernel : kernels) {
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel);
    if (kernel_name == prim::kPrimLayerNorm->name()) {
      format_transform_ = false;
      return;
    }
    auto value = common::AnfAlgo::GetCNodePrimitive(kernel);
    if (value != nullptr && value->GetAttr("format") != nullptr &&
        GetValue<std::string>(value->GetAttr("format")) == kOpFormat_NHWC) {
      format_transform_ = false;
      return;
    }
    if (kernel_name == prim::kPrimConv2D->name()) {
      conv_cnt++;
    }
    if (kernel_name == prim::kPrimBatchNorm->name()) {
      bn_cnt++;
    }
  }
  if (conv_cnt + bn_cnt > 1) {
    format_transform_ = true;
    return;
  }
  format_transform_ = false;
}

bool GetSelectKernelResult(const CNodePtr &kernel_node,
                           const std::shared_ptr<KernelBuildInfo::KernelBuildInfoBuilder> &builder,
                           KernelType *kernel_type,
                           std::vector<std::tuple<size_t, TypeId, TypeId>> *input_reduce_index) {
  bool result = false;
  std::vector<std::tuple<size_t, TypeId, TypeId>> output_reduce_index;
  if (IsPrimitiveCNode(kernel_node, prim::kPrimCustom)) {
    // Custom op select kernel from OpLib
    result = SelectCustomKernel(kernel_node, builder->Build(), kernel_type);
  } else if (*kernel_type == UNKNOWN_KERNEL_TYPE) {
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    if (kernel::Factory<kernel::NativeGpuKernelMod>::Instance().IsRegistered(kernel_name)) {
      result = kernel::NativeGpuKernelMod::GpuCheckSupport(kernel_name, GetKernelAttrFromBuildInfo(builder->Build()));
      if (!result) {
        std::tie(result, *input_reduce_index, output_reduce_index) =
          kernel::NativeGpuKernelMod::GpuReducePrecisionCheck(kernel_name,
                                                              GetKernelAttrFromBuildInfo(builder->Build()));
        if (result) {
          const size_t kReduceToTypeIdx = 2;
          for (const auto &item : *input_reduce_index) {
            auto idx = std::get<0>(item);
            auto to_type_id = std::get<kReduceToTypeIdx>(item);
            builder->SetInputDeviceType(to_type_id, idx);
          }
          for (const auto &item : output_reduce_index) {
            auto idx = std::get<0>(item);
            auto to_type_id = std::get<kReduceToTypeIdx>(item);
            builder->SetOutputDeviceType(to_type_id, idx);
          }
        }
      }
    } else {
      result = kernel::NativeGpuKernelModFactory::GetInstance().SearchRegistered(
        common::AnfAlgo::GetCNodeName(kernel_node), builder->Build());
      if (!result) {
        result = kernel::NativeGpuKernelModFactory::GetInstance().ReducePrecision(
          common::AnfAlgo::GetCNodeName(kernel_node), builder);
      }
    }

    if (!result && (!common::AnfAlgo::IsControlOpExecInBackend(kernel_node))) {
      result = SelectAkgKernel(kernel_node, builder->Build());
      *kernel_type = AKG_KERNEL;
    }
  } else if (*kernel_type == AKG_KERNEL) {
    result = SelectAkgKernel(kernel_node, builder->Build());
  }
  return result;
}

#ifdef ENABLE_TUPLE_UNFOLD
bool GetSelectKernelObjectTypeResult(const CNodePtr &kernel_node, KernelType kernel_type) {
  auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
  // Only the kernel nodes that register kernel attr can support the backoff.
  bool backoff_support_condition =
    ((kernel_type == UNKNOWN_KERNEL_TYPE) && !IsPrimitiveCNode(kernel_node, prim::kPrimCustom) &&
     !common::AnfAlgo::IsGraphKernel(kernel_node));
  std::vector<kernel::KernelAttr> kernel_attrs;
  if (kernel::NativeGpuKernelModFactory::GetInstance().IsRegistered(kernel_name)) {
    kernel_attrs = kernel::NativeGpuKernelModFactory::GetInstance().GetGpuSupportedList(kernel_name);
  } else if (backoff_support_condition) {
    // Kernel that is not supported can try to backed off on CPU and use the CPU kernel attrs to set object type.
    kernel_attrs = kernel::NativeCpuKernelMod::GetCpuSupportedList(kernel_name);
  }

  // Some dynamic kernels may not set the kernel attrs on GPU. Skip check only supports the tuple fold.
  if (kernel_attrs.empty() || kernel_attrs[0].GetSkipCheck()) {
    auto input_object_types =
      kernel::TypeIdToKernelObjectTypeForTupleUnfold(AnfAlgo::GetAllInputObjectType(kernel_node));
    auto output_object_types =
      kernel::TypeIdToKernelObjectTypeForTupleUnfold(AnfAlgo::GetAllOutputObjectType(kernel_node));
    kernel::SetKernelObjectTypeBuildInfo(kernel_node, input_object_types, output_object_types);
    if (!kernel_attrs.empty()) {
      auto kernel_build_info = AnfAlgo::GetSelectKernelBuildInfo(kernel_node);
      kernel_build_info->SetOpType(kernel::OpType::SKIP);
    }
    return true;
  }

  std::vector<kernel::KernelAttr> object_selected_kernel_attrs;
  if (!kernel::SelectKernelByObjectType(kernel_node, kernel_attrs, &object_selected_kernel_attrs, true) &&
      !kernel::SelectKernelByObjectType(kernel_node, kernel_attrs, &object_selected_kernel_attrs, false)) {
    return false;
  }

  kernel::SetKernelObjectTypeWithSelectedAttr(kernel_node, object_selected_kernel_attrs[0]);
  return true;
}
#endif

std::pair<std::string, ExceptionType> SetKernelInfoWithMsg(const CNodePtr &kernel_node, KernelType kernel_type) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  if (common::AnfAlgo::IsGraphKernel(kernel_node)) {
    auto func_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(kernel_node);
    MS_EXCEPTION_IF_NULL(func_graph);
    SetGraphKernelInfo(kernel_node, func_graph);
    return {};
  }
  auto builder = std::make_shared<KernelBuildInfo::KernelBuildInfoBuilder>();
  AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), kernel_node.get());
#ifdef ENABLE_TUPLE_UNFOLD
  bool selected = GetSelectKernelObjectTypeResult(kernel_node, kernel_type);
  if (!selected) {
    std::stringstream ss;
    ss << "kernel object types are not supported for " << common::AnfAlgo::GetCNodeName(kernel_node)
       << " on GPU currently.";
    return {ss.str(), NotSupportError};
  }
#endif
  std::vector<std::string> inputs_format;
  std::vector<TypeId> inputs_type;
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  for (size_t input_index = 0; input_index < input_num; ++input_index) {
    inputs_format.emplace_back(kOpFormat_DEFAULT);
    inputs_type.push_back(common::AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, input_index));
  }

  std::vector<std::string> outputs_format;
  std::vector<TypeId> outputs_type;
#ifdef ENABLE_TUPLE_UNFOLD
  auto output_kernel_object_types = builder->Build()->GetAllOutputKernelObjectTypes();
  if (output_kernel_object_types.size() == 1 && output_kernel_object_types[0] == kernel::KernelObjectType::TUPLE) {
    outputs_type = {common::AnfAlgo::GetOutputInferDataType(kernel_node, 0)};
    outputs_format = {kOpFormat_DEFAULT};
  } else {
#endif
    size_t output_num = AnfAlgo::GetOutputElementNum(kernel_node);
    for (size_t output_index = 0; output_index < output_num; ++output_index) {
      outputs_format.emplace_back(kOpFormat_DEFAULT);
      outputs_type.push_back(common::AnfAlgo::GetOutputInferDataType(kernel_node, output_index));
    }
#ifdef ENABLE_TUPLE_UNFOLD
  }
#endif
  std::string origin_data_format = kOpFormat_DEFAULT;
  if (IsNeedProcessFormatInfo(kernel_node, inputs_type)) {
    UpdateKernelFormatInfo(kernel_node, inputs_type, &inputs_format, &outputs_format, &origin_data_format);
  }
  builder->SetOriginDataFormat(origin_data_format);
  builder->SetInputsFormat(inputs_format);
  builder->SetInputsDeviceType(inputs_type);
  builder->SetOutputsFormat(outputs_format);
  builder->SetOutputsDeviceType(outputs_type);
#ifdef ENABLE_TUPLE_UNFOLD
  kernel::UnfoldKernelBuildInfo(kernel_node);
  if (!common::AnfAlgo::HasNodeAttr(kAttrDynInputSizes, kernel_node)) {
    kernel::SetDynamicInputSizeAttr(kernel_node);
  }
#endif
  MS_LOG(INFO) << kernel_node->fullname_with_scope() << " kernel attr info: "
               << kernel::FetchPrintInfoByKernelAttr(kernel::GetKernelAttrFromBuildInfo(builder->Build()));

  std::vector<std::tuple<size_t, TypeId, TypeId>> input_reduce_index;
  bool result = GetSelectKernelResult(kernel_node, builder, &kernel_type, &input_reduce_index);
  SetTensorDeviceInfo(*(builder->Build()), kernel_node, input_reduce_index);

  // Return the kernel select failure info.
  if (common::AnfAlgo::HasNodeAttr(ops::kBatchRank, kernel_node) &&
      !kVmapGPUWhiteList.count(common::AnfAlgo::GetCNodeName(kernel_node))) {
    builder->SetKernelType(UNKNOWN_KERNEL_TYPE);
    builder->SetProcessor(kernel::Processor::UNKNOWN);
    std::stringstream ss;
    ss << common::AnfAlgo::GetCNodeName(kernel_node)
       << " does not support 'batch_rank' on GPU, which means that 'vmap' cannot support "
       << common::AnfAlgo::GetCNodeName(kernel_node) << " on GPU currently.";
    return {ss.str(), NotSupportError};
  }
  if (!result && (!common::AnfAlgo::IsControlOpExecInBackend(kernel_node))) {
    builder->SetKernelType(UNKNOWN_KERNEL_TYPE);
    builder->SetProcessor(kernel::Processor::UNKNOWN);
    return PrintUnsupportedTypeWarning(kernel_node, inputs_type, outputs_type, kernel_type);
  }

  builder->SetKernelType(kernel_type);
  builder->SetProcessor(kernel::Processor::CUDA);
  return {};
}

void GPUGraphKernelInfo::SetKernelInfo(const CNodePtr &kernel_node, KernelType kernel_type) {
  auto [msg, etype] = SetKernelInfoWithMsg(kernel_node, kernel_type);
  if (msg.empty()) {
    return;
  }
  MS_EXCEPTION(etype) << msg;
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
