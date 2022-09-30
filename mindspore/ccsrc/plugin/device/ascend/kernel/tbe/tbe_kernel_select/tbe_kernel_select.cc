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

#include "plugin/device/ascend/kernel/tbe/tbe_kernel_select/tbe_kernel_select.h"

#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>
#include <iterator>
#include <algorithm>
#include "kernel/common_utils.h"
#include "plugin/device/ascend/kernel/tbe/tbe_convert_utils.h"
#include "plugin/device/ascend/kernel/tbe/tbe_dynamic_shape_util.h"
#include "plugin/device/ascend/kernel/tbe/tbe_json/single_tbe_json_creator.h"
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_compile.h"
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_select/tbe_kernel_broadcast_selecter.h"
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_select/tbe_kernel_reduce_selecter.h"
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_select/tbe_property_checker.h"
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_select/tbe_selector_creator.h"
#include "backend/common/optimizer/helper.h"

#include "include/common/utils/json_operation_utils.h"

namespace mindspore::kernel {
constexpr auto kName = "name";
constexpr auto kDtype = "dtype";
constexpr auto kFormat = "format";
constexpr auto kPrefixInput = "input";
constexpr auto kPrefixOutput = "output";
constexpr char kParamTypeDynamic[] = "dynamic";
constexpr char kParamTypeRequre[] = "required";
constexpr char kParamTypeOptional[] = "optional";
constexpr int64_t kDynamicInvalidNum = -1;
constexpr size_t kDynamicFirstInputIndex = 0;

void TbeMetadataInfo(const CNodePtr &kernel_node, std::vector<std::shared_ptr<KernelBuildInfo>> *kernel_info_list) {
  auto tbe_selector = TbeKernelSelect(kernel_node, kernel_info_list);
  tbe_selector.TbeMetadataInfoEx();
}

bool NeedCheckDynamicImpl(const CNodePtr &cnode) {
  auto node_name = common::AnfAlgo::GetCNodeName(cnode);
  auto op_info_ptr = tbe::TbeDynamicShapeUtil::FindOp(node_name, cnode);
  MS_EXCEPTION_IF_NULL(op_info_ptr);
  auto is_op_dynamic_shape = common::AnfAlgo::IsDynamicShape(cnode);
  auto is_kernel_dynamic_shape = op_info_ptr->dynamic_shape();
  auto is_kernel_dynamic_compile_static = op_info_ptr->dynamic_compile_static();
  if (is_op_dynamic_shape && is_kernel_dynamic_shape) {
    return true;
  }

  if (!is_op_dynamic_shape && is_kernel_dynamic_compile_static) {
    return true;
  }
  return false;
}

bool TbeCheckIsSupportedSpec(const CNodePtr &kernel_node, const KernelBuildInfoPtr &select_kernel_build_info) {
  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> kernel_info_list;
  auto tbe_selector = TbeKernelSelect(kernel_node, &kernel_info_list);
  if (NeedCheckDynamicImpl(kernel_node)) {
    common::AnfAlgo::SetNodeAttr(kAttrIsKernelDynamicImpl, MakeValue(true), kernel_node);
    auto ret = tbe_selector.FindKernelInfo(select_kernel_build_info);
    if (ret) {
      if (common::AnfAlgo::IsDynamicShape(kernel_node)) {
        common::AnfAlgo::SetNodeAttr(kAttrIsKernelDynamicShape, MakeValue(true), kernel_node);
      }
      return true;
    } else {
      common::AnfAlgo::SetNodeAttr(kAttrIsKernelDynamicImpl, MakeValue(false), kernel_node);
    }
  }
  return tbe_selector.FindKernelInfo(select_kernel_build_info);
}

bool TbeCheckIsSupportedAny(const CNodePtr &kernel_node) {
  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> kernel_info_list;
  auto tbe_selector = TbeKernelSelect(kernel_node, &kernel_info_list);
  if (NeedCheckDynamicImpl(kernel_node)) {
    common::AnfAlgo::SetNodeAttr(kAttrIsKernelDynamicImpl, MakeValue(true), kernel_node);
    auto ret = tbe_selector.CheckIsAnyKernelInfo();
    if (ret) {
      if (common::AnfAlgo::IsDynamicShape(kernel_node)) {
        common::AnfAlgo::SetNodeAttr(kAttrIsKernelDynamicShape, MakeValue(true), kernel_node);
      }
      return true;
    } else {
      common::AnfAlgo::SetNodeAttr(kAttrIsKernelDynamicImpl, MakeValue(false), kernel_node);
    }
  }
  return tbe_selector.CheckIsAnyKernelInfo();
}

TbeKernelSelect::TbeKernelSelect(CNodePtr kernel_node, std::vector<std::shared_ptr<KernelBuildInfo>> *kernel_info_list)
    : cnode_ptr_(std::move(kernel_node)), kernel_info_list_(kernel_info_list), check_cnode(CheckCNode()) {}

bool TbeKernelSelect::CheckCNode() {
  MS_EXCEPTION_IF_NULL(cnode_ptr_);
  MS_EXCEPTION_IF_NULL(kernel_info_list_);
  node_name_ = common::AnfAlgo::GetCNodeName(cnode_ptr_);
  auto op_info_ptr = tbe::TbeDynamicShapeUtil::FindOp(node_name_, cnode_ptr_);
  if (!op_info_ptr) {
    return false;
  }
  if (!TbePropertyChecker::CheckTbeProperties(cnode_ptr_)) {
    MS_LOG(INFO) << "Warning: node(" << full_name_ << ") is not supported by tbe ai_core.";
    return false;
  }
  return true;
}

bool TbeKernelSelect::CheckIsAnyKernelInfo() {
  TbeMetadataInfoEx();
  return !kernel_info_list_->empty();
}

bool TbeKernelSelect::FindKernelInfo(const KernelBuildInfoPtr &select_kernel_build_info) {
  TbeMetadataInfoEx();
  for (auto &kernel_info : *kernel_info_list_) {
    auto builder = KernelBuildInfo::KernelBuildInfoBuilder(kernel_info);
    auto item = builder.Build();
    if (*item == *select_kernel_build_info) {
      return true;
    }
  }

  return false;
}

void TbeKernelSelect::GetKernelHashName() {
  auto json_creator = std::make_shared<SelectTbeJsonCreator>();
  MS_EXCEPTION_IF_NULL(json_creator);
  bool ret = json_creator->GenJson(cnode_ptr_, &kernel_json);
  if (!ret) {
    MS_LOG(EXCEPTION) << "Gen node hash failed. [" << cnode_ptr_->fullname_with_scope() << "]";
  }
  kernel_hash_name = json_creator->GetJsonName();
}

void TbeKernelSelect::TbeMetadataInfoEx() {
  std::string new_select_process = common::GetEnv("XXX");
  if (new_select_process.empty()) {
    // Step1: Initialize, get node info
    if (!Initialize()) {
      MS_LOG(DEBUG) << "Initialize failed, node name: " << full_name_;
      return;
    }
    // Step2: if kernel build info in cache, use cache return
    if (GetKernelBuildInfoFromCache()) {
      return;
    }
    // Step3:
    auto get_support_format_dtype_func = GetSelectorFunc(cnode_ptr_);
    if (!get_support_format_dtype_func) {
      MS_LOG(ERROR) << "Get selector func failed, node name: " << full_name_;
      return;
    }
    SupportFormatDType support_format_dtype;
    get_support_format_dtype_func(cnode_ptr_, &support_format_dtype);
    PrintSupportedFormatDtype(support_format_dtype);
    GenerateKernelBuildInfo(support_format_dtype);
    FilterInvalidKernelInfo();
    AddKernelBuildInfoToCache();
  } else {
    if (!check_cnode) {
      return;
    }
    GetKernelHashName();
    node_name_ = common::AnfAlgo::GetCNodeName(cnode_ptr_);
    full_name_ = cnode_ptr_->fullname_with_scope();
    auto op_info_ptr = tbe::TbeDynamicShapeUtil::FindOp(node_name_, cnode_ptr_);
    MS_EXCEPTION_IF_NULL(op_info_ptr);
    // if op pattern is FormatAgnostic, can't use select cache
    bool skip_cache = (op_info_ptr->op_pattern() == kFormatAgnosticPattern);

    auto iter = select_cache_.find(kernel_hash_name);
    if (iter != select_cache_.end() && !skip_cache) {
      for (auto &cache_info : iter->second) {
        auto builder = KernelBuildInfo::KernelBuildInfoBuilder(cache_info);
        (void)kernel_info_list_->emplace_back(builder.Build());
      }
      MS_LOG(DEBUG) << "Select kernel cache hit " << kernel_hash_name << " for node "
                    << cnode_ptr_->fullname_with_scope();
      return;
    }

    if (op_info_ptr->is_dynamic_format()) {
      GetDynamicFormatPatternKernelInfo(*op_info_ptr);
    } else {
      OpPattern pattern = op_info_ptr->op_pattern();
      if (pattern == kCommonPattern) {
        GetCommonPatternKernelInfo(*op_info_ptr);
      } else if (pattern == kFormatAgnosticPattern) {
        GetAgnosticPatternKernelInfo(*op_info_ptr);
      } else if (pattern == kBroadcastPattern) {
        GetBroadcastPatternKernelInfo(*op_info_ptr);
      } else if (pattern == kReducePattern) {
        GetReducePatternKernelInfo(*op_info_ptr);
      } else {
        MS_LOG(INFO) << "Warning: op pattern is invailed.";
      }
    }
    // check support
    FilterInvalidKernelInfo();
    AddKernelBuildInfoToCache();
  }
}

void TbeKernelSelect::AddKernelBuildInfoToCache() {
  if (kernel_info_list_->empty()) {
    select_cache_[kernel_hash_name] = {};
    return;
  }
  select_cache_[kernel_hash_name] = *kernel_info_list_;
  MS_LOG(INFO) << "Add select kernel cache " << kernel_hash_name << " from node " << cnode_ptr_->fullname_with_scope()
               << ", cache size: " << select_cache_.size();
}

void TbeKernelSelect::GetCommonPatternKernelInfo(const OpInfo &op_info) {
  auto dyn_input_sizes = GetNodeDynamicInputs();
  // get real input/output num
  size_t real_input_tensor_num = common::AnfAlgo::GetInputTensorNum(cnode_ptr_);
  const auto inputs_info = op_info.inputs_ptr();
  size_t real_output_tensor_num = common::AnfAlgo::GetOutputTensorNum(cnode_ptr_);
  const auto outputs_info = op_info.outputs_ptr();
  if (inputs_info.empty() && outputs_info.empty()) {
    MS_LOG(EXCEPTION) << common::AnfAlgo::GetCNodeName(cnode_ptr_)
                      << "'s op info input & output is null, please check.";
  }
  // create kernel build info from opinfo
  size_t kernel_build_info_num =
    inputs_info.empty() ? outputs_info[0]->dtypes().size() : inputs_info[0]->dtypes().size();
  for (size_t kernel_build_info_index = 0; kernel_build_info_index < kernel_build_info_num; ++kernel_build_info_index) {
    auto builder = KernelBuildInfo::KernelBuildInfoBuilder();
    SetTbeBuildCommonInfo(op_info, &builder);
    std::vector<std::string> inputs_format;
    std::vector<TypeId> inputs_device_type;
    std::vector<std::string> inputs_reshape_type;
    std::vector<std::string> inputs_value_depend;
    // input
    if (!GenBuilderItem(true, kernel_build_info_index, real_input_tensor_num, inputs_info, dyn_input_sizes,
                        &inputs_format, &inputs_device_type, &inputs_reshape_type, &inputs_value_depend)) {
      break;
    }
    builder.SetInputsDeviceType(inputs_device_type);
    builder.SetInputsFormat(inputs_format);
    builder.SetInputsReshapeType(inputs_reshape_type);
    builder.SetInputsValueDepend(inputs_value_depend);
    // output
    std::vector<std::string> outputs_format;
    std::vector<TypeId> outputs_device_type;
    std::vector<std::string> outputs_reshape_type;
    std::vector<std::string> outputs_value_depend;
    if (!GenBuilderItem(false, kernel_build_info_index, real_output_tensor_num, outputs_info, dyn_input_sizes,
                        &outputs_format, &outputs_device_type, &outputs_reshape_type, &outputs_value_depend)) {
      break;
    }
    builder.SetOutputsDeviceType(outputs_device_type);
    builder.SetOutputsFormat(outputs_format);
    builder.SetOutputsReshapeType(outputs_reshape_type);
    (void)kernel_info_list_->emplace_back(builder.Build());
  }
}

void TbeKernelSelect::GetDynamicFormatPatternKernelInfo(const OpInfo &op_info) {
  OpInfo op_info_new;
  CreateNewOpInfo(op_info, &op_info_new);
  GetCommonPatternKernelInfo(op_info_new);
}

void TbeKernelSelect::GetAgnosticPatternKernelInfo(const OpInfo &op_info) {
  if (op_info.inputs_ptr().size() != 1) {
    MS_LOG(EXCEPTION) << "AgnosticPattern only support one input.";
  }
  auto format = AnfAlgo::GetPrevNodeOutputFormat(cnode_ptr_, 0);
  if (!IsOneOfFormat(format)) {
    MS_LOG(INFO) << "Got the unknown format " << format;
    format = kOpFormat_DEFAULT;
  }
  SupportFormat support_format;
  SupportFormatItem input_item;
  SupportFormatItem output_item;
  input_item.assign(op_info.inputs_ptr().size(), format);
  output_item.assign(op_info.outputs_ptr().size(), format);
  (void)support_format.input_format.emplace_back(input_item);
  (void)support_format.output_format.emplace_back(output_item);
  OpInfo op_info_new;
  CreateNewOpInfo(op_info, support_format, &op_info_new);
  GetCommonPatternKernelInfo(op_info_new);
}

void TbeKernelSelect::GetBroadcastPatternKernelInfo(const OpInfo &op_info) {
  auto broadcast_selecter = TbeKernelBroadCastSelecter(cnode_ptr_);
  SupportFormat support_format;
  (void)broadcast_selecter.GetShapeInfo(&support_format);
  (void)broadcast_selecter.IsBroadCastSupport5HD(&support_format);
  (void)broadcast_selecter.IsBroadCastSupportFracZ(&support_format);
  (void)broadcast_selecter.IsBroadCastSupportC1HWNCoC0(&support_format);
  (void)broadcast_selecter.IsBroadCastSupportFracNZ(&support_format);
  (void)broadcast_selecter.IsBroadCastSupportNDC1HWC0(&support_format);
  OpInfo op_info_new;
  CreateNewOpInfo(op_info, support_format, &op_info_new);
  GetCommonPatternKernelInfo(op_info_new);
}

void TbeKernelSelect::GetReducePatternKernelInfo(const OpInfo &op_info) {
  auto reduce_selecter = TbeKernelReduceSelecter(cnode_ptr_);
  SupportFormat support_format;
  (void)reduce_selecter.GetShapeInfo(&support_format);
  (void)reduce_selecter.IsReduceSupport5HD(&support_format);
  (void)reduce_selecter.IsReduceSupportFracZ(&support_format);
  (void)reduce_selecter.IsReduceSupportC1HWNCoC0(&support_format);
  (void)reduce_selecter.IsReduceSupportFracNZ(&support_format);
  OpInfo op_info_new;
  CreateNewOpInfo(op_info, support_format, &op_info_new);
  GetCommonPatternKernelInfo(op_info_new);
}

void TbeKernelSelect::FilterInvalidKernelInfo() {
  if (kernel_info_list_->empty()) {
    MS_LOG(INFO) << "Warning: get kernel build info failed. Skip check supported. Op name: " << full_name_;
    return;
  }
  std::vector<std::shared_ptr<KernelBuildInfo>> kernel_info_list;
  auto dynamic_inputs = GetNodeDynamicInputs();
  for (const auto &kernel_build_info : *kernel_info_list_) {
    if (!FilterInvalidShape(kernel_build_info, dynamic_inputs)) {
      continue;
    }
    if (!TbeCheckSupported(kernel_build_info)) {
      continue;
    }
    (void)kernel_info_list.emplace_back(kernel_build_info);
  }
  if (kernel_info_list.empty()) {
    MS_LOG(DEBUG) << "After tbe check supported, all valid AI CORE kernel infos were filtered out. Node:" << full_name_;
  }
  (*kernel_info_list_).swap(kernel_info_list);
}

bool TbeKernelSelect::FilterInvalidShape(const KernelBuildInfoPtr &kernel_build_info,
                                         const std::vector<int64_t> &dynamic_inputs) {
  MS_EXCEPTION_IF_NULL(kernel_build_info);
  const auto &kernel_build_info_inputs_format = kernel_build_info->GetAllInputFormats();
  // dynamic input just need to check first input, because other inputs copy from 1th input;
  auto iter_num = dynamic_inputs.empty() ? kernel_build_info_inputs_format.size() : dynamic_inputs.size();
  size_t input_index = kDynamicFirstInputIndex;
  for (size_t i = 0; i < iter_num; ++i) {
    if (dynamic_inputs.empty()) {
      input_index = i;
    } else if (i > 0) {
      input_index += dynamic_inputs[i - 1] > 0 ? dynamic_inputs[i - 1] : 1;
    }
    auto shape = common::AnfAlgo::GetPrevNodeOutputInferShape(cnode_ptr_, input_index);
    const auto &format = kernel_build_info_inputs_format.at(input_index);
    if (!IsShapeMatchFormat(shape, format)) {
      return false;
    }
  }
  const auto &kernel_build_info_outputs_format = kernel_build_info->GetAllOutputFormats();
  for (size_t j = 0; j < kernel_build_info_outputs_format.size(); ++j) {
    auto shape = common::AnfAlgo::GetOutputInferShape(cnode_ptr_, j);
    const auto &format = kernel_build_info_outputs_format[j];
    if (!IsShapeMatchFormat(shape, format)) {
      return false;
    }
  }
  return true;
}

bool TbeKernelSelect::IsShapeMatchFormat(const ShapeVector &shape, const std::string &format) {
  // if format is default, it means support all format
  if (format == kOpFormat_DEFAULT) {
    return true;
  }
  // server not support format with C04 suffix
  if (IsOneOfServerFormatC04(format)) {
    MS_LOG(INFO) << "Warning: Server not support format with C04 suffix.";
    return false;
  }
  // kOpFormat_FRAC_NZ >=2 || %16 == 0
  if (format == kOpFormat_FRAC_NZ) {
    return (shape.size() >= kShape2dDims) ||
           (shape.size() == 1 && (shape[0] == 1 || (shape[0] % SizeToLong(kCubeSize) == 0)));
  }
  // RNN
  if (!IsShapeMatchFormatRNN(shape, format)) {
    return false;
  }
  // not support format:
  // 3D formats with shape size > 5
  if (IsOneOf3DFormat(format)) {
    return shape.size() <= kShape5dDims;
  }
  // check format is valid.
  if (!IsOneOfFormat(format)) {
    MS_LOG(INFO) << "Got the unknown format " << format;
    return false;
  }
  return true;
}

bool TbeKernelSelect::IsShapeMatchFormatRNN(const ShapeVector &shape, const std::string &format) {
  // kOpFormat_FRACTAL_ZN_RNN >=2
  if (format == kOpFormat_FRACTAL_ZN_RNN || format == kOpFormat_ND_RNN_BIAS) {
    if (!common::AnfAlgo::HasNodeAttr(kAttrInputSize, cnode_ptr_) ||
        !common::AnfAlgo::HasNodeAttr(kAttrHiddenSize, cnode_ptr_)) {
      return false;
    }
    auto input_size = common::AnfAlgo::GetNodeAttr<int64_t>(cnode_ptr_, kAttrInputSize);
    auto hidden_size = common::AnfAlgo::GetNodeAttr<int64_t>(cnode_ptr_, kAttrHiddenSize);

    // kOpFormat_FRACTAL_ZN_RNN >=2
    if (format == kOpFormat_FRACTAL_ZN_RNN) {
      if (shape.size() < kDim2) {
        return false;
      }
      auto last_but_one_dim = shape[shape.size() - kDim2];
      if ((last_but_one_dim != abstract::Shape::kShapeDimAny) && (last_but_one_dim != input_size) &&
          (last_but_one_dim != hidden_size) && (last_but_one_dim != input_size + hidden_size)) {
        return false;
      }
    }
    // kOpFormat_ND_RNN_BIAS shape not empty()
    if (format == kOpFormat_ND_RNN_BIAS) {
      if (shape.empty()) {
        return false;
      }
      auto last_dim = shape[shape.size() - kDim1];
      if (last_dim != abstract::Shape::kShapeDimAny && last_dim % hidden_size != 0) {
        return false;
      }
    }
  }
  return true;
}

bool TbeKernelSelect::TbeCheckSupported(const KernelBuildInfoPtr &kernel_build_info) {
  auto op_info = tbe::TbeDynamicShapeUtil::FindOp(cnode_ptr_);
  if (!op_info->need_check_supported()) {
    return true;
  }
  MS_EXCEPTION_IF_NULL(kernel_build_info);
  // replace kernel_info with current kernel info
  auto kernel_build_info_tmp = AnfAlgo::GetSelectKernelBuildInfo(cnode_ptr_);
  AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info, cnode_ptr_.get());
  auto &build_manager = kernel::ascend::TbeKernelCompileManager::GetInstance();
  auto ret =
    HostCheck::CheckValidDeviceShape(cnode_ptr_) && build_manager.TbeOpCheckSupported(cnode_ptr_, &kernel_json);
  AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info_tmp, cnode_ptr_.get());
  return ret;
}

void TbeKernelSelect::SetTbeBuildCommonInfo(const mindspore::kernel::OpInfo &op_info,
                                            mindspore::kernel::KernelBuildInfo::KernelBuildInfoBuilder *builder) {
  MS_EXCEPTION_IF_NULL(builder);
  builder->SetProcessor(AICORE);
  std::string fusion_name = op_info.fusion_type();
  auto fusion_type = kernel::GetFusionTypeByName(fusion_name);
  if (fusion_type != UNKNOWN_FUSION_TYPE) {
    builder->SetFusionType(fusion_type);
  }
  builder->SetOpPattern(op_info.op_pattern());
  builder->SetKernelType(TBE_KERNEL);
}

std::vector<int64_t> TbeKernelSelect::GetNodeDynamicInputs() {
  // get dynamic inputs
  auto primitive = common::AnfAlgo::GetCNodePrimitive(cnode_ptr_);
  MS_EXCEPTION_IF_NULL(primitive);
  std::vector<int64_t> dyn_input_sizes = {};
  if (primitive->HasAttr(kAttrDynInputSizes)) {
    dyn_input_sizes = GetValue<std::vector<int64_t>>(primitive->GetAttr(kAttrDynInputSizes));
  }
  return dyn_input_sizes;
}

bool TbeKernelSelect::GenBuilderItem(bool is_input, size_t kernel_build_info_index, size_t real_io_tensor_num,
                                     const std::vector<std::shared_ptr<OpIOInfo>> &ios_info,
                                     const std::vector<int64_t> &dyn_input_sizes, std::vector<std::string> *formats,
                                     std::vector<TypeId> *device_types, std::vector<std::string> *reshape_types,
                                     std::vector<std::string> *value_depends) {
  MS_EXCEPTION_IF_NULL(formats);
  MS_EXCEPTION_IF_NULL(device_types);
  MS_EXCEPTION_IF_NULL(reshape_types);
  MS_EXCEPTION_IF_NULL(value_depends);
  size_t dynamic_input_index = 0;
  size_t real_io_tensor_index = 0;
  size_t io_info_index = 0;
  size_t io_info_num = ios_info.size();
  for (; io_info_index < io_info_num && real_io_tensor_index < real_io_tensor_num; io_info_index++) {
    std::shared_ptr<OpIOInfo> io_info_item = ios_info[io_info_index];
    const auto &kernel_build_info_dtype = io_info_item->dtypes()[kernel_build_info_index];
    std::string kernel_build_info_format;
    if (!io_info_item->formats().empty()) {
      kernel_build_info_format = io_info_item->formats()[kernel_build_info_index];
    }
    const std::string &io_param_type = io_info_item->param_type();
    auto reshape_type = io_info_item->reshape_type();
    auto value_depend = io_info_item->value_depend();
    if (io_param_type == kParamTypeDynamic) {
      // dynamic io
      if (is_input) {
        if (dynamic_input_index >= dyn_input_sizes.size()) {
          MS_LOG(EXCEPTION) << "dyn_input_sizes attr set error, dynamic_input_index: " << dynamic_input_index
                            << ", dyn_input_sizes size: " << dyn_input_sizes.size();
        }
        int64_t dynamic_input_size = dyn_input_sizes[dynamic_input_index];
        for (int64_t i = 0; i < dynamic_input_size; ++i) {
          (void)device_types->emplace_back(tbe::DtypeToTypeId(kernel_build_info_dtype));
          (void)formats->emplace_back(kernel_build_info_format);
          (void)reshape_types->emplace_back(reshape_type);
          (void)value_depends->emplace_back(value_depend);
        }
        dynamic_input_index++;
        real_io_tensor_index = SizetAddWithOverflowCheck(real_io_tensor_index, LongToSize(dynamic_input_size));
      } else {
        if (ios_info.size() != 1) {
          MS_LOG(EXCEPTION) << "if output is dynamic, so output must has one output.";
        }
        for (size_t i = 0; i < real_io_tensor_num; ++i) {
          (void)device_types->emplace_back(tbe::DtypeToTypeId(kernel_build_info_dtype));
          (void)formats->emplace_back(kernel_build_info_format);
          (void)reshape_types->emplace_back(reshape_type);
          (void)value_depends->emplace_back(value_depend);
        }
        real_io_tensor_index = SizetAddWithOverflowCheck(real_io_tensor_index, real_io_tensor_num);
      }
    } else if (io_param_type == kParamTypeRequre || io_param_type == kParamTypeOptional) {
      // require or optional io
      (void)device_types->emplace_back(tbe::DtypeToTypeId(kernel_build_info_dtype));
      (void)formats->emplace_back(kernel_build_info_format);
      (void)reshape_types->emplace_back(reshape_type);
      (void)value_depends->emplace_back(value_depend);
      real_io_tensor_index++;
    } else {
      MS_LOG(EXCEPTION) << "op info's param type is not match: " << io_param_type;
    }
  }

  if (real_io_tensor_index != real_io_tensor_num) {
    std::string io_type = is_input ? "inputs " : "outputs";
    MS_LOG(INFO) << node_name_ << "'s " << io_type << "op io info num: " << io_info_num
                 << ", real io tensor num:" << real_io_tensor_num << "real_io_tensor_index(" << real_io_tensor_index
                 << ") != real_io_tensor_num(" << real_io_tensor_num << ")";
    return false;
  }
  return true;
}

void TbeKernelSelect::CreateNewOpIOInfo(const mindspore::kernel::OpIOInfo &op_io_info,
                                        const std::vector<std::vector<std::string>> &support_format_item, size_t index,
                                        mindspore::kernel::OpIOInfo *op_io_info_new) {
  MS_EXCEPTION_IF_NULL(op_io_info_new);
  op_io_info_new->set_index(op_io_info.index());
  op_io_info_new->set_name(op_io_info.name());
  op_io_info_new->set_param_type(op_io_info.param_type());
  op_io_info_new->set_need_compile(op_io_info.need_compile());
  op_io_info_new->set_reshape_type(op_io_info.reshape_type());
  op_io_info_new->set_shape(op_io_info.shape());
  op_io_info_new->set_value_depend(op_io_info.value_depend());
  // dtype
  std::vector<std::string> dtype_new;
  auto dtype = op_io_info.dtypes();
  for (size_t i = 0; i < support_format_item.size(); ++i) {
    (void)dtype_new.insert(dtype_new.end(), dtype.begin(), dtype.end());
  }
  op_io_info_new->set_dtypes(dtype_new);
  // format
  std::vector<std::string> format_new;
  for (const auto &formats : support_format_item) {
    auto format = formats.at(index);
    for (size_t j = 0; j < dtype.size(); ++j) {
      (void)format_new.emplace_back(format);
    }
  }
  op_io_info_new->set_formats(format_new);
}

std::vector<std::string> TbeKernelSelect::SplitStrToVec(const std::string &op_select_json_item) {
  const std::map<std::string, std::string> kDynamicFormatMap = {
    {"NCHW", "DefaultFormat"}, {"ND", "DefaultFormat"}, {"NCDHW", "DefaultFormat"}};
  if (op_select_json_item.empty()) {
    MS_LOG(EXCEPTION) << "Op select ret item is null.";
  }
  const char space = ' ';
  const char sep = ',';
  std::string op_select_tmp = op_select_json_item + ",";
  std::vector<std::string> ret;
  auto begin = op_select_tmp.find_first_not_of(space, 0);
  auto sep_pos = op_select_tmp.find(sep);
  if (begin >= sep_pos) {
    MS_LOG(EXCEPTION) << "Select ret json is error.";
  }
  while (sep_pos != std::string::npos) {
    auto obj = op_select_tmp.substr(begin, sep_pos - begin);
    if (kDynamicFormatMap.find(obj) != kDynamicFormatMap.end()) {
      obj = kDynamicFormatMap.at(obj);
    }
    (void)ret.emplace_back(obj);
    begin = op_select_tmp.find_first_not_of(space, sep_pos + 1);
    sep_pos = op_select_tmp.find(sep, begin);
  }
  return ret;
}

std::string TbeKernelSelect::OpSelectFormat() {
  std::string res_json_str;
  MS_LOG(DEBUG) << "Format select for node:[" << cnode_ptr_->fullname_with_scope() << "].";
  auto &build_manager = kernel::ascend::TbeKernelCompileManager::GetInstance();
  res_json_str = build_manager.TbeOpSelectFormat(cnode_ptr_);
  return res_json_str;
}

void TbeKernelSelect::CreateNewOpInfo(const mindspore::kernel::OpInfo &op_info, const SupportFormat &support_format,
                                      mindspore::kernel::OpInfo *op_info_new) {
  MS_EXCEPTION_IF_NULL(op_info_new);
  if (support_format.input_format.empty() || support_format.output_format.empty()) {
    MS_LOG(EXCEPTION) << "Support input format and output format size can not be empty, but the input format size is: "
                      << support_format.input_format.size()
                      << ", output format size is: " << support_format.output_format.size();
  }
  if (op_info.inputs_ptr().size() != support_format.input_format[0].size() ||
      op_info.outputs_ptr().size() != support_format.output_format[0].size()) {
    MS_LOG(EXCEPTION) << "BroadCast input/output size not match, op info input size:" << op_info.inputs_ptr().size()
                      << ", input support size: " << support_format.input_format[0].size()
                      << ", op info output size: " << op_info.outputs_ptr().size()
                      << ", output support size: " << support_format.output_format[0].size();
  }
  *op_info_new = op_info;
  op_info_new->ClearInputs();
  op_info_new->ClearOutputs();
  for (size_t i = 0; i < op_info.inputs_ptr().size(); ++i) {
    auto inputs_ptr = op_info.inputs_ptr();
    auto input = inputs_ptr.at(i);
    auto input_new = std::make_shared<OpIOInfo>();
    CreateNewOpIOInfo(*input, support_format.input_format, i, input_new.get());
    op_info_new->add_inputs_ptr(input_new);
  }
  for (size_t j = 0; j < op_info.outputs_ptr().size(); ++j) {
    auto outputs_ptr = op_info.outputs_ptr();
    auto output = outputs_ptr.at(j);
    auto output_new = std::make_shared<OpIOInfo>();
    CreateNewOpIOInfo(*output, support_format.output_format, j, output_new.get());
    op_info_new->add_outputs_ptr(output_new);
  }
}

struct SelectOpIOInfo {
  std::string name;
  std::vector<std::string> dtypes;
  std::vector<std::string> formats;
};

void TbeKernelSelect::CreateNewOpInfo(const mindspore::kernel::OpInfo &op_info,
                                      mindspore::kernel::OpInfo *op_info_new) {
  MS_EXCEPTION_IF_NULL(op_info_new);
  auto op_seclect_json = OpSelectFormat();
  if (!op_seclect_json.empty()) {
    nlohmann::json json_obj;
    if (!ParseJson(op_seclect_json, &json_obj)) {
      MS_LOG(EXCEPTION) << "Parse op_select_json error.";
    }
    if (!json_obj.is_object()) {
      MS_LOG(EXCEPTION) << "JsonStr is not an object, the jsonStr is:" << op_seclect_json;
    }
    std::vector<SelectOpIOInfo> inputs;
    std::vector<SelectOpIOInfo> outputs;
    for (const auto &item : json_obj.items()) {
      const std::string &item_name = item.key();
      bool is_input = (item_name.find(kPrefixInput) != std::string::npos);
      bool is_output = (item_name.find(kPrefixOutput) != std::string::npos);
      if (!is_input && !is_output) {
        MS_LOG(EXCEPTION) << "op select ret json is error.";
      }
      if (is_input) {
        SelectOpIOInfo select_input;
        select_input.name = item.value().at(kName);
        std::string input_dtype_item = item.value().at(kDtype);
        select_input.dtypes = SplitStrToVec(input_dtype_item);
        std::string input_format_item = item.value().at(kFormat);
        select_input.formats = SplitStrToVec(input_format_item);
        (void)inputs.emplace_back(select_input);
      } else {
        SelectOpIOInfo select_output;
        select_output.name = item.value().at(kName);
        std::string input_dtype_item = item.value().at(kDtype);
        select_output.dtypes = SplitStrToVec(input_dtype_item);
        std::string input_format_item = item.value().at(kFormat);
        select_output.formats = SplitStrToVec(input_format_item);
        (void)outputs.emplace_back(select_output);
      }
    }

    if (op_info.inputs_ptr().size() != inputs.size() || op_info.outputs_ptr().size() != outputs.size()) {
      MS_LOG(EXCEPTION) << "select format input/output size not equal, please check register.";
    }

    *op_info_new = op_info;
    op_info_new->ClearInputs();
    op_info_new->ClearOutputs();
    for (size_t i = 0; i < op_info.inputs_ptr().size(); ++i) {
      auto input_new = std::make_shared<OpIOInfo>();
      const auto &inputs_ptr = op_info.inputs_ptr();
      CreateNewOpIOInfo(*(inputs_ptr.at(i)), inputs.at(i).dtypes, inputs.at(i).formats, input_new.get());
      op_info_new->add_inputs_ptr(input_new);
    }
    for (size_t i = 0; i < op_info.outputs_ptr().size(); ++i) {
      auto output_new = std::make_shared<OpIOInfo>();
      const auto &outputs_ptr = op_info.outputs_ptr();
      CreateNewOpIOInfo(*(outputs_ptr.at(i)), outputs.at(i).dtypes, outputs.at(i).formats, output_new.get());
      op_info_new->add_outputs_ptr(output_new);
    }
  }
}

void TbeKernelSelect::CreateNewOpIOInfo(const mindspore::kernel::OpIOInfo &op_io_info,
                                        const std::vector<std::string> &support_dtype,
                                        const std::vector<std::string> &support_format,
                                        mindspore::kernel::OpIOInfo *op_io_info_new) {
  MS_EXCEPTION_IF_NULL(op_io_info_new);
  op_io_info_new->set_index(op_io_info.index());
  op_io_info_new->set_name(op_io_info.name());
  op_io_info_new->set_param_type(op_io_info.param_type());
  op_io_info_new->set_need_compile(op_io_info.need_compile());
  op_io_info_new->set_reshape_type(op_io_info.reshape_type());
  op_io_info_new->set_shape(op_io_info.shape());
  op_io_info_new->set_value_depend(op_io_info.value_depend());
  // dtype  && format
  op_io_info_new->set_dtypes(support_dtype);
  op_io_info_new->set_formats(support_format);
}

void TbeKernelSelect::PrintSupportedFormatDtype(const SupportFormatDType &support_format_dtype) {
  MS_LOG(DEBUG) << "full_name: " << full_name_;
  MS_LOG(DEBUG) << "==============input dtype=============";
  for (auto input : support_format_dtype.input_dtypes) {
    std::stringstream ss;
    (void)copy(input.begin(), input.end(), std::ostream_iterator<std::string>(ss, ","));
    MS_LOG(DEBUG) << "[ " << ss.str() << " ]";
  }
  MS_LOG(DEBUG) << "==============input format=============";
  for (auto input : support_format_dtype.input_formats) {
    std::stringstream ss;
    (void)copy(input.begin(), input.end(), std::ostream_iterator<std::string>(ss, ","));
    MS_LOG(DEBUG) << "[ " << ss.str() << " ]";
  }
  MS_LOG(DEBUG) << "==============output dtype=============";
  for (auto input : support_format_dtype.output_dtypes) {
    std::stringstream ss;
    (void)copy(input.begin(), input.end(), std::ostream_iterator<std::string>(ss, ","));
    MS_LOG(DEBUG) << "[ " << ss.str() << " ]";
  }
  MS_LOG(DEBUG) << "==============output format=============";
  for (auto input : support_format_dtype.output_formats) {
    std::stringstream ss;
    (void)copy(input.begin(), input.end(), std::ostream_iterator<std::string>(ss, ","));
    MS_LOG(DEBUG) << "[ " << ss.str() << " ]";
  }
}

bool TbeKernelSelect::Initialize() {
  // Init 1.op_name, 2.full_name, 3.op_info, 4.kernel_json, 5.kernel_hash_name
  node_name_ = common::AnfAlgo::GetCNodeName(cnode_ptr_);
  full_name_ = cnode_ptr_->fullname_with_scope();
  op_info_ = tbe::TbeDynamicShapeUtil::FindOp(node_name_, cnode_ptr_);
  if (!op_info_) {
    return false;
  }
  if (!TbePropertyChecker::CheckTbeProperties(cnode_ptr_)) {
    MS_LOG(INFO) << "Warning: node(" << full_name_ << ") is not supported by tbe ai_core.";
    return false;
  }
  auto json_creator = std::make_shared<SelectTbeJsonCreator>();
  MS_EXCEPTION_IF_NULL(json_creator);
  auto ret = json_creator->GenJson(cnode_ptr_, &kernel_json);
  if (!ret) {
    MS_LOG(EXCEPTION) << "Gen node hash failed. [" << cnode_ptr_->fullname_with_scope() << "]";
  }
  kernel_hash_name = json_creator->GetJsonName();
  return true;
}

bool TbeKernelSelect::GetKernelBuildInfoFromCache() {
  // Note: kFormatAgnosticPattern need select, like cast ...
  if (op_info_->op_pattern() == kFormatAgnosticPattern) {
    return false;
  }
  auto iter = select_cache_.find(kernel_hash_name);
  if (iter == select_cache_.end()) {
    return false;
  }
  for (const auto &cache_info : iter->second) {
    auto builder = KernelBuildInfo::KernelBuildInfoBuilder(cache_info);
    (void)kernel_info_list_->emplace_back(builder.Build());
  }
  MS_LOG(DEBUG) << "Select kernel cache hit " << kernel_hash_name << " for node " << cnode_ptr_->fullname_with_scope();
  return true;
}

void TbeKernelSelect::GenerateKernelBuildInfo(const SupportFormatDType &support_format_dtype) {
  auto dyn_input_sizes = GetNodeDynamicInputs();
  // get real input/output num
  size_t real_input_num = common::AnfAlgo::GetInputTensorNum(cnode_ptr_);
  size_t real_output_num = common::AnfAlgo::GetOutputTensorNum(cnode_ptr_);
  auto op_info_input_num = support_format_dtype.input_dtypes.size();
  auto op_info_output_num = support_format_dtype.output_dtypes.size();
  if (op_info_output_num == 0 && op_info_input_num == 0) {
    MS_LOG(EXCEPTION) << "input and output is null, please check, " << full_name_;
  }

  auto select_support_num = support_format_dtype.output_dtypes.at(0).size();
  for (size_t support_index = 0; support_index < select_support_num; ++support_index) {
    KernelBuildInfoItem input_kernel_build_info;
    KernelBuildInfoItem output_kernel_build_info;
    for (size_t io_index = 0, real_put_index = 0; real_put_index < real_input_num && io_index < op_info_input_num;) {
      auto op_io_info = op_info_->inputs_ptr().at(io_index);
      auto support_dtype = support_format_dtype.input_dtypes.at(io_index).at(support_index);
      auto support_format = support_format_dtype.input_formats.at(io_index).at(support_index);
      int64_t dynamic_input_num = kDynamicInvalidNum;
      if (!dyn_input_sizes.empty()) {
        if (io_index >= dyn_input_sizes.size()) {
          MS_LOG(EXCEPTION) << "Io index should be less than the dynamic input's size, node name: " << full_name_;
        } else {
          dynamic_input_num = dyn_input_sizes[io_index] > 0 ? dyn_input_sizes[io_index] : 1;
        }
      }
      ConstructIOKernelBuildInfo(op_io_info, support_dtype, support_format, dynamic_input_num, &input_kernel_build_info,
                                 &io_index, &real_put_index);
    }
    for (size_t io_index = 0, real_put_index = 0; real_put_index < real_output_num && io_index < op_info_output_num;) {
      auto op_io_info = op_info_->outputs_ptr().at(io_index);
      auto support_dtype = support_format_dtype.output_dtypes.at(io_index).at(support_index);
      auto support_format = support_format_dtype.output_formats.at(io_index).at(support_index);
      int64_t dynamic_output_num = kDynamicInvalidNum;
      if (op_io_info->param_type() == kParamTypeDynamic) {
        if (op_info_->outputs_ptr().size() != 1) {
          MS_LOG(EXCEPTION) << "Dynamic output num only support 1, output name: " << op_io_info->name()
                            << ", node name: " << full_name_;
        }
        dynamic_output_num = SizeToLong(real_output_num);
      }
      ConstructIOKernelBuildInfo(op_io_info, support_dtype, support_format, dynamic_output_num,
                                 &output_kernel_build_info, &io_index, &real_put_index);
    }
    ConstructKernelBuildInfo(input_kernel_build_info, output_kernel_build_info);
  }
}

void TbeKernelSelect::ConstructKernelBuildInfo(const KernelBuildInfoItem &input_kernel_build_info,
                                               const KernelBuildInfoItem &output_kernel_build_info) {
  auto builder = KernelBuildInfo::KernelBuildInfoBuilder();
  builder.SetProcessor(AICORE);
  std::string fusion_name = op_info_->fusion_type();
  auto fusion_type = GetFusionTypeByName(fusion_name);
  if (fusion_type != UNKNOWN_FUSION_TYPE) {
    builder.SetFusionType(fusion_type);
  }
  builder.SetOpPattern(op_info_->op_pattern());
  builder.SetKernelType(TBE_KERNEL);
  builder.SetInputsDeviceType(input_kernel_build_info.device_types);
  builder.SetInputsFormat(input_kernel_build_info.formats);
  builder.SetInputsReshapeType(input_kernel_build_info.reshape_types);
  builder.SetInputsValueDepend(input_kernel_build_info.value_depends);
  builder.SetOutputsDeviceType(output_kernel_build_info.device_types);
  builder.SetOutputsFormat(output_kernel_build_info.formats);
  builder.SetOutputsReshapeType(output_kernel_build_info.reshape_types);
  (void)kernel_info_list_->emplace_back(builder.Build());
}

void TbeKernelSelect::ConstructIOKernelBuildInfo(const OpIOInfoPtr &op_io_info, const std::string &support_dtype,
                                                 const std::string &support_format, int64_t dynamic_num,
                                                 KernelBuildInfoItem *kernel_build_info_item, size_t *io_index,
                                                 size_t *real_put_index) const {
  MS_EXCEPTION_IF_NULL(kernel_build_info_item);
  MS_EXCEPTION_IF_NULL(io_index);
  MS_EXCEPTION_IF_NULL(real_put_index);
  if (op_io_info->param_type() == kParamTypeDynamic) {
    if (dynamic_num == kDynamicInvalidNum) {
      MS_LOG(EXCEPTION) << "Get node dynamic inputs num failed, node name: " << full_name_;
    }
    for (int64_t i = 0; i < dynamic_num; ++i) {
      (void)kernel_build_info_item->formats.emplace_back(support_format);
      (void)kernel_build_info_item->device_types.emplace_back(tbe::DtypeToTypeId(support_dtype));
      (void)kernel_build_info_item->reshape_types.emplace_back(op_io_info->reshape_type());
      (void)kernel_build_info_item->value_depends.emplace_back(op_io_info->value_depend());
    }
    (*real_put_index) += LongToSize(dynamic_num);
  } else {
    (void)kernel_build_info_item->formats.emplace_back(support_format);
    (void)kernel_build_info_item->device_types.emplace_back(tbe::DtypeToTypeId(support_dtype));
    (void)kernel_build_info_item->reshape_types.emplace_back(op_io_info->reshape_type());
    (void)kernel_build_info_item->value_depends.emplace_back(op_io_info->value_depend());
    (*real_put_index) += 1;
  }
  (*io_index) += 1;
}
}  // namespace mindspore::kernel
