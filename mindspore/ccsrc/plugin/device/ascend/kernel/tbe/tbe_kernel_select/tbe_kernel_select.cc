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
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_select/tbe_property_checker.h"
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_select/tbe_selector_creator.h"
#include "backend/common/optimizer/helper.h"

namespace mindspore::kernel {
constexpr int64_t kDynamicInvalidNum = -1;
constexpr size_t kDynamicFirstInputIndex = 0;
constexpr size_t kMatMulInputSize = 3;

bool IsSkipStaticImplCheck(const std::string &op_name) {
  const std::set<std::string> only_has_dynamic_impl = {kUnsortedSegmentSumOpName};
  return (only_has_dynamic_impl.count(op_name) != 0);
}

void TbeMetadataInfo(const CNodePtr &kernel_node, std::vector<std::shared_ptr<KernelBuildInfo>> *kernel_info_list) {
  auto tbe_selector = TbeKernelSelect(kernel_node, kernel_info_list);
  auto node_name = common::AnfAlgo::GetCNodeName(kernel_node);
  auto op_info_ptr = tbe::TbeDynamicShapeUtil::FindOp(node_name, kernel_node);
  if (op_info_ptr == nullptr) {
    return;
  }

  if (common::AnfAlgo::HasDynamicTupleInput(kernel_node)) {
    return;
  }

  if (IsKernelDynamicImpl(kernel_node)) {
    common::AnfAlgo::SetNodeAttr(kAttrIsKernelDynamicImpl, MakeValue(true), kernel_node);
    if (tbe_selector.CheckOpSupported()) {
      return;
    }
  }
  if (IsSkipStaticImplCheck(node_name)) {
    return;
  }
  MS_EXCEPTION_IF_NULL(kernel_info_list);
  kernel_info_list->clear();
  common::AnfAlgo::SetNodeAttr(kAttrIsKernelDynamicImpl, MakeValue(false), kernel_node);
  tbe_selector.CheckOpSupported();
}

bool TbeCheckIsSupportedSpec(const CNodePtr &kernel_node, const KernelBuildInfoPtr &select_kernel_build_info) {
  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> kernel_info_list;
  auto tbe_selector = TbeKernelSelect(kernel_node, &kernel_info_list);
  return tbe_selector.TbeCheckIsSupportedSpec(kernel_node, select_kernel_build_info);
}

bool TbeCheckIsSupportedAny(const CNodePtr &kernel_node) {
  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> kernel_info_list;
  auto tbe_selector = TbeKernelSelect(kernel_node, &kernel_info_list);
  return tbe_selector.TbeCheckIsSupportedAny(kernel_node);
}

TbeKernelSelect::TbeKernelSelect(CNodePtr kernel_node, std::vector<std::shared_ptr<KernelBuildInfo>> *kernel_info_list)
    : cnode_ptr_(std::move(kernel_node)), kernel_info_list_(kernel_info_list) {}

bool TbeKernelSelect::CheckOpSupported() {
  MS_EXCEPTION_IF_NULL(kernel_info_list_);
  (void)GetSupportFormatDTypes();
  if (common::AnfAlgo::IsDtypeFormatSensitiveOp(cnode_ptr_)) {
    kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
    auto input_format = AnfAlgo::GetPrevNodeOutputFormat(cnode_ptr_, 0);
    auto input_type = common::AnfAlgo::GetPrevNodeOutputInferDataType(cnode_ptr_, 0);
    auto output_type = common::AnfAlgo::GetOutputInferDataType(cnode_ptr_, 0);
    builder.SetInputsFormat({input_format});
    builder.SetOutputsFormat({input_format});
    builder.SetInputsDeviceType({input_type});
    builder.SetOutputsDeviceType({output_type});
    builder.SetProcessor(kernel::Processor::AICORE);
    auto kernel_build_info = builder.Build();
    return std::any_of(kernel_info_list_->cbegin(), kernel_info_list_->cend(),
                       [&kernel_build_info](const auto &kernel_info) { return *kernel_info == *kernel_build_info; });
  }
  return (!kernel_info_list_->empty());
}

std::vector<std::shared_ptr<KernelBuildInfo>> TbeKernelSelect::GetSupportFormatDTypes() {
  // Step1: Initialize, get node info
  if (!Initialize()) {
    MS_LOG(DEBUG) << "Initialize failed, node name: " << full_name_;
    return {};
  }
  // Step2: if kernel build info in cache, use cache return
  if (GetKernelBuildInfoFromCache()) {
    MS_EXCEPTION_IF_NULL(kernel_info_list_);
    return *kernel_info_list_;
  }
  // Step3:
  auto get_support_format_dtype_func = GetSelectorFunc(cnode_ptr_);
  if (!get_support_format_dtype_func) {
    MS_LOG(ERROR) << "Get selector func failed, node name: " << full_name_;
    return {};
  }
  SupportFormatDType support_format_dtype;
  get_support_format_dtype_func(cnode_ptr_, &support_format_dtype);
  PrintSupportedFormatDtype(support_format_dtype);
  if (IsSupportFormatDTypeValid(support_format_dtype)) {
    GenerateKernelBuildInfo(support_format_dtype);
  }
  FilterInvalidKernelInfo();
  AddKernelBuildInfoToCache();
  MS_EXCEPTION_IF_NULL(kernel_info_list_);
  return *kernel_info_list_;
}

bool TbeKernelSelect::TbeCheckIsSupportedSpec(const CNodePtr &kernel_node,
                                              const KernelBuildInfoPtr &select_kernel_build_info) {
  MS_EXCEPTION_IF_NULL(select_kernel_build_info);
  if (IsOpSupportDynamicImpl(kernel_node)) {
    common::AnfAlgo::SetNodeAttr(kAttrIsKernelDynamicImpl, MakeValue(true), kernel_node);
    auto format_dtypes = GetSupportFormatDTypes();
    auto exist = std::any_of(format_dtypes.cbegin(), format_dtypes.cend(), [&](const auto &item) {
      MS_EXCEPTION_IF_NULL(item);
      return (*item == *select_kernel_build_info);
    });
    if (exist) {
      return true;
    }
  }
  if (IsSkipStaticImplCheck(node_name_)) {
    common::AnfAlgo::EraseNodeAttr(kAttrIsKernelDynamicImpl, kernel_node);
    return false;
  }
  common::AnfAlgo::SetNodeAttr(kAttrIsKernelDynamicImpl, MakeValue(false), kernel_node);
  auto format_dtypes = GetSupportFormatDTypes();
  auto exist = std::any_of(format_dtypes.cbegin(), format_dtypes.cend(), [&](const auto &item) {
    MS_EXCEPTION_IF_NULL(item);
    return (*item == *select_kernel_build_info);
  });
  if (exist) {
    return true;
  }
  common::AnfAlgo::EraseNodeAttr(kAttrIsKernelDynamicImpl, kernel_node);
  return false;
}

bool TbeKernelSelect::TbeCheckIsSupportedAny(const CNodePtr &kernel_node) {
  if (IsOpSupportDynamicImpl(kernel_node)) {
    common::AnfAlgo::SetNodeAttr(kAttrIsKernelDynamicImpl, MakeValue(true), kernel_node);
    auto format_dtypes_filter = GetSupportFormatDTypesWithFilter();
    if (!format_dtypes_filter.empty()) {
      return true;
    }
  }
  if (IsSkipStaticImplCheck(node_name_)) {
    common::AnfAlgo::EraseNodeAttr(kAttrIsKernelDynamicImpl, kernel_node);
    return false;
  }
  common::AnfAlgo::SetNodeAttr(kAttrIsKernelDynamicImpl, MakeValue(false), kernel_node);
  auto format_dtypes_filter = GetSupportFormatDTypesWithFilter();
  if (!format_dtypes_filter.empty()) {
    return true;
  }
  common::AnfAlgo::EraseNodeAttr(kAttrIsKernelDynamicImpl, kernel_node);
  return false;
}

std::vector<std::shared_ptr<kernel::KernelBuildInfo>> TbeKernelSelect::GetSupportFormatDTypesWithFilter() {
  auto format_dtypes = GetSupportFormatDTypes();
  const std::set<std::string> skip_filter_nodes = {kCastOpName, kTransDataOpName};
  if (skip_filter_nodes.count(node_name_) != 0 || common::AnfAlgo::HasNodeAttr(kAttrMutableKernel, cnode_ptr_)) {
    return format_dtypes;
  }
  bool reduce_flag = false;
  auto format_dtypes_filter = FilterRaisedOrReducePrecisionMatchedKernelInfo(cnode_ptr_, format_dtypes, &reduce_flag);
  return format_dtypes_filter;
}

void TbeKernelSelect::AddKernelBuildInfoToCache() {
  MS_EXCEPTION_IF_NULL(op_info_);
  if (op_info_->op_pattern() == kFormatAgnosticPattern) {
    return;
  }

  std::vector<std::shared_ptr<KernelBuildInfo>> cache_kernel_list;
  MS_EXCEPTION_IF_NULL(kernel_info_list_);
  if (kernel_info_list_->empty()) {
    cache_kernel_list = {};
  } else {
    cache_kernel_list = *kernel_info_list_;
  }
  select_cache_[kernel_hash_name_] = cache_kernel_list;
  MS_EXCEPTION_IF_NULL(cnode_ptr_);
  MS_LOG(INFO) << "Add select kernel cache " << kernel_hash_name_ << " from node " << cnode_ptr_->fullname_with_scope()
               << ", kernel info size: " << cache_kernel_list.size() << ", cache size: " << select_cache_.size();
}

void TbeKernelSelect::FilterInvalidKernelInfo() {
  MS_EXCEPTION_IF_NULL(kernel_info_list_);
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
    if (!FilterUnspportedMatMul(kernel_build_info)) {
      continue;
    }
    // Skip check for ACL op.
    if (!common::AnfAlgo::HasNodeAttr(kAttrMutableKernel, cnode_ptr_)) {
      if (!TbeCheckSupported(kernel_build_info)) {
        continue;
      }
    }
    (void)kernel_info_list.emplace_back(kernel_build_info);
  }
  if (kernel_info_list.empty()) {
    MS_LOG(DEBUG) << "After tbe check supported, all valid AI CORE kernel infos were filtered out. Node:" << full_name_;
  }
  (*kernel_info_list_).swap(kernel_info_list);
}

bool TbeKernelSelect::FilterUnspportedMatMul(const KernelBuildInfoPtr &kernel_build_info) {
  // A MatMul op is unsupported if it has a bias and bias is fp32
  // we need to filter it out or it will cause compile error.
  if (common::AnfAlgo::GetCNodeName(cnode_ptr_) != prim::kPrimMatMul->name() ||
      !common::AnfAlgo::IsDynamicShape(cnode_ptr_)) {
    return true;
  }
  const auto &input_dtypes = kernel_build_info->GetAllInputDeviceTypes();
  if (input_dtypes.size() < kMatMulInputSize) {
    return true;
  }
  const auto bias_dtype = input_dtypes[kMatMulInputSize - 1];
  return !(bias_dtype == TypeId::kNumberTypeFloat32 || bias_dtype == TypeId::kNumberTypeFloat);
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
  // only transdata supported, no need in graph
  if (IsOneOfNotSupportedTransFormat(format)) {
    return false;
  }
  // if format is default, it means support all format
  if (common::AnfAlgo::GetCNodeName(cnode_ptr_) == prim::kPrimBNTrainingReduce->name() ||
      common::AnfAlgo::GetCNodeName(cnode_ptr_) == prim::kPrimBNTrainingUpdate->name()) {
    if ((format == kOpFormat_DEFAULT) && common::AnfAlgo::IsDynamicShape(cnode_ptr_)) {
      return false;
    }
  }
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
  MS_EXCEPTION_IF_NULL(op_info);
  if (!op_info->need_check_support()) {
    return true;
  }
  MS_EXCEPTION_IF_NULL(kernel_build_info);
  // replace kernel_info with current kernel info
  auto kernel_build_info_tmp = AnfAlgo::GetSelectKernelBuildInfo(cnode_ptr_);
  AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info, cnode_ptr_.get());
  auto &build_manager = kernel::ascend::TbeKernelCompileManager::GetInstance();
  auto ret =
    HostCheck::CheckValidDeviceShape(cnode_ptr_) && build_manager.TbeOpCheckSupported(cnode_ptr_, &kernel_json_);
  AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info_tmp, cnode_ptr_.get());
  return ret;
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

bool IsSupportFormatDTypeValidItem(const std::vector<std::vector<std::string>> &item, size_t *support_item_num) {
  MS_EXCEPTION_IF_NULL(support_item_num);
  for (const auto &put : item) {
    if (*support_item_num == UINT64_MAX) {
      *support_item_num = put.size();
    }
    if (*support_item_num != put.size()) {
      return false;
    }
  }
  return true;
}

bool TbeKernelSelect::IsSupportFormatDTypeValid(const SupportFormatDType &support_format_dtype) {
  auto support_item_num = UINT64_MAX;
  if (!IsSupportFormatDTypeValidItem(support_format_dtype.input_dtypes, &support_item_num) ||
      !IsSupportFormatDTypeValidItem(support_format_dtype.input_formats, &support_item_num) ||
      !IsSupportFormatDTypeValidItem(support_format_dtype.output_dtypes, &support_item_num) ||
      !IsSupportFormatDTypeValidItem(support_format_dtype.output_formats, &support_item_num)) {
    PrintSupportedFormatDtype(support_format_dtype);
    return false;
  }
  return true;
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
  auto ret = json_creator->GenJson(cnode_ptr_, &kernel_json_);
  if (!ret) {
    MS_LOG(EXCEPTION) << "Gen node hash failed. [" << cnode_ptr_->fullname_with_scope() << "]";
  }
  kernel_hash_name_ = json_creator->GetJsonName();
  return true;
}

bool TbeKernelSelect::GetKernelBuildInfoFromCache() {
  // Note: kFormatAgnosticPattern need select, like cast ...
  MS_EXCEPTION_IF_NULL(op_info_);
  if (op_info_->op_pattern() == kFormatAgnosticPattern) {
    return false;
  }
  auto iter = select_cache_.find(kernel_hash_name_);
  if (iter == select_cache_.end()) {
    return false;
  }
  MS_EXCEPTION_IF_NULL(kernel_info_list_);
  for (const auto &cache_info : iter->second) {
    auto builder = KernelBuildInfo::KernelBuildInfoBuilder(cache_info);
    (void)kernel_info_list_->emplace_back(builder.Build());
  }
  MS_EXCEPTION_IF_NULL(cnode_ptr_);
  MS_LOG(DEBUG) << "Select kernel cache hit " << kernel_hash_name_ << " for node " << cnode_ptr_->fullname_with_scope();
  return true;
}

void TbeKernelSelect::GenerateKernelBuildInfo(const SupportFormatDType &support_format_dtype) {
  auto dyn_input_sizes = GetNodeDynamicInputs();
  // get real input/output num
  size_t real_input_num = AnfAlgo::GetInputElementNum(cnode_ptr_);
  size_t real_output_num = AnfAlgo::GetOutputElementNum(cnode_ptr_);
  auto op_info_input_num = support_format_dtype.input_dtypes.size();
  auto op_info_output_num = support_format_dtype.output_dtypes.size();
  if (op_info_output_num == 0 && op_info_input_num == 0) {
    MS_LOG(EXCEPTION) << "input and output is null, please check, " << full_name_;
  }

  auto select_support_num = support_format_dtype.output_dtypes.at(0).size();
  for (size_t support_index = 0; support_index < select_support_num; ++support_index) {
    KernelBuildInfoItem input_kernel_build_info;
    KernelBuildInfoItem output_kernel_build_info;
    int64_t dynamic_input_num = kDynamicInvalidNum;
    for (size_t io_index = 0, real_put_index = 0; real_put_index < real_input_num && io_index < op_info_input_num;) {
      auto op_io_info = op_info_->inputs_ptr().at(io_index);
      auto support_dtype = support_format_dtype.input_dtypes.at(io_index).at(support_index);
      auto support_format = support_format_dtype.input_formats.at(io_index).at(support_index);
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
      int64_t dynamic_output_num = dynamic_input_num;
      if (op_io_info->param_type() == kDynamic) {
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
  builder.SetOpPattern(op_info_->op_pattern());
  builder.SetKernelType(TBE_KERNEL);
  builder.SetInputsDeviceType(input_kernel_build_info.device_types);
  builder.SetInputsFormat(input_kernel_build_info.formats);
  builder.SetInputsKernelObjectType(
    std::vector<KernelObjectType>(input_kernel_build_info.formats.size(), KernelObjectType::TENSOR));
  builder.SetInputsReshapeType(input_kernel_build_info.reshape_types);
  builder.SetOutputsDeviceType(output_kernel_build_info.device_types);
  builder.SetOutputsFormat(output_kernel_build_info.formats);
  builder.SetOutputsReshapeType(output_kernel_build_info.reshape_types);
  builder.SetOutputsKernelObjectType(
    std::vector<KernelObjectType>(output_kernel_build_info.formats.size(), KernelObjectType::TENSOR));
  (void)kernel_info_list_->emplace_back(builder.Build());
}

void TbeKernelSelect::ConstructIOKernelBuildInfo(const OpIOInfoPtr &op_io_info, const std::string &support_dtype,
                                                 const std::string &support_format, int64_t dynamic_num,
                                                 KernelBuildInfoItem *kernel_build_info_item, size_t *io_index,
                                                 size_t *real_put_index) const {
  MS_EXCEPTION_IF_NULL(op_io_info);
  MS_EXCEPTION_IF_NULL(kernel_build_info_item);
  MS_EXCEPTION_IF_NULL(io_index);
  MS_EXCEPTION_IF_NULL(real_put_index);
  if (op_io_info->param_type() == kDynamic) {
    if (dynamic_num == kDynamicInvalidNum) {
      MS_LOG(EXCEPTION) << "Get node dynamic inputs num failed, node name: " << full_name_;
    }
    for (int64_t i = 0; i < dynamic_num; ++i) {
      (void)kernel_build_info_item->formats.emplace_back(support_format);
      (void)kernel_build_info_item->device_types.emplace_back(tbe::DtypeToTypeId(support_dtype));
      (void)kernel_build_info_item->reshape_types.emplace_back(op_io_info->reshape_type());
    }
    (*real_put_index) += LongToSize(dynamic_num);
  } else {
    (void)kernel_build_info_item->formats.emplace_back(support_format);
    (void)kernel_build_info_item->device_types.emplace_back(tbe::DtypeToTypeId(support_dtype));
    (void)kernel_build_info_item->reshape_types.emplace_back(op_io_info->reshape_type());
    (*real_put_index) += 1;
  }
  (*io_index) += 1;
}
}  // namespace mindspore::kernel
