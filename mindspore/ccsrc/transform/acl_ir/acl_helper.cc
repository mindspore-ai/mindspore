/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "transform/acl_ir/acl_helper.h"
#include <set>
#include <map>
#include <unordered_map>
#include <string>
#include "include/api/data_type.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "include/transform/graph_ir/types.h"
#include "ops/nn_ops.h"
#include "ops/array_ops.h"
#include "ops/conv_pool_ops.h"
#include "ops/structure_ops.h"
#include "ops/ascend_op_name.h"
#include "ops/image_op_name.h"
#include "ops/math_op_name.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include "transform/acl_ir/acl_adapter_info.h"
#include "transform/acl_ir/ge_adapter_info.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace transform {
namespace {
#define GET_DEFAULT_FORMAT(shape) (shape.size() == kDim4 ? kOpFormat_NCHW : kOpFormat_DEFAULT)
static const std::set<std::string> kDefaultOutputNode = {
  // Dynamic output shape kernel.
  kUniqueOpName, kMaskedSelectOpName, kNonMaxSuppressionV3OpName,
  // Dropout
  kDropoutGenMaskOpName, kDropoutGenMaskV3OpName, kStatelessDropOutGenMaskOpName, kDropoutDoMaskOpName,
  kDropoutDoMaskV3OpName, kDropoutOpName, kDropoutGradOpName, kDropout2DOpName, kDropout3DOpName,
  // Special Op
  kAffineGridOpName, kRangeOpName, kBernoulliOpName};

static const std::set<std::string> kHcomOps = {
  kHcomOpTypeAllReduce, kHcomOpTypeReduce,        kHcomOpTypeAllGather, kHcomOpTypeBroadcast, kHcomOpTypeSend,
  kHcomOpTypeReceive,   kHcomOpTypeReduceScatter, kHcomOpTypeAllToAllV, kHcomOpTypeBarrier,   kHcomOpTypeScatter,
  kHcomOpTypeGather,    kHcomOpTypeBatchSendRecv, kHcomOpTypeAlltoAllV};

static const HashMap<GeDataType, TypeId> kGeTypeToMsType = {{GeDataType::DT_BOOL, kNumberTypeBool},
                                                            {GeDataType::DT_INT8, kNumberTypeInt8},
                                                            {GeDataType::DT_INT16, kNumberTypeInt16},
                                                            {GeDataType::DT_INT32, kNumberTypeInt32},
                                                            {GeDataType::DT_INT64, kNumberTypeInt64},
                                                            {GeDataType::DT_UINT8, kNumberTypeUInt8},
                                                            {GeDataType::DT_UINT16, kNumberTypeUInt16},
                                                            {GeDataType::DT_UINT32, kNumberTypeUInt32},
                                                            {GeDataType::DT_UINT64, kNumberTypeUInt64},
                                                            {GeDataType::DT_FLOAT16, kNumberTypeFloat16},
                                                            {GeDataType::DT_FLOAT, kNumberTypeFloat32},
                                                            {GeDataType::DT_DOUBLE, kNumberTypeFloat64},
                                                            {GeDataType::DT_STRING, kObjectTypeString},
                                                            {GeDataType::DT_COMPLEX64, kNumberTypeComplex64},
                                                            {GeDataType::DT_COMPLEX128, kNumberTypeComplex128},
                                                            {GeDataType::DT_BF16, kNumberTypeBFloat16}};

TypeId ConvertGeType(GeDataType type) {
  if (kGeTypeToMsType.count(type) != 0) {
    return kGeTypeToMsType.at(type);
  }
  return kTypeUnknown;
}

bool GLogIsDebug() {
  const std::string &glog = common::GetEnv("GLOG_v");
  auto is_debug = !glog.empty() && glog[0] == '0';

  auto submodule = common::GetEnv("MS_SUBMODULE_LOG_v");
  bool is_submodule_debug = false;
  constexpr std::string_view kKernelSub = "KERNEL";
  constexpr size_t kKernelPos = 7;
  if (!submodule.empty() && submodule.find(kKernelSub) != std::string::npos) {
    auto start_pos = submodule.find(kKernelSub) + kKernelPos;
    is_submodule_debug = submodule[start_pos] == '0';
  }
  return is_debug || is_submodule_debug;
}

bool NeedNDInput(const CNodePtr &cnode, const AnfNodePtr &input_node, const std::string &new_format,
                 std::string *input_format, bool *input_special_flag) {
  if (AclHelper::IsNopNode(cnode) && !AclHelper::CheckDefaultSupportFormat(*input_format)) {
    *input_special_flag = true;
    return true;
  }

  auto input_cnode = input_node->cast<CNodePtr>();
  if (input_cnode != nullptr && common::AnfAlgo::HasNodeAttr(kAttrAclSpecialFormat, input_cnode)) {
    return true;
  }

  if (!AclHelper::CheckDefaultSupportFormat(*input_format) || AclHelper::CheckDefaultSupportFormat(new_format)) {
    return false;
  }

  SetParameterFormat(input_node, new_format, input_format);
  return false;
}

bool NeedNDOutput(const CNodePtr &cnode, const size_t input_num, const size_t output_num,
                  const std::vector<std::string> &input_formats) {
  auto name = GetCNodeFuncName(cnode);
  if (kDefaultOutputNode.count(name) != 0) {
    return true;
  }

  if (input_num != output_num) {
    if (output_num != 1 || input_formats.empty() ||
        !std::all_of(input_formats.begin(), input_formats.end(),
                     [&input_formats](const std::string &format) { return format == input_formats[0]; })) {
      return true;
    }
  }

  for (size_t i = 0; i < output_num; ++i) {
    const auto &shape = common::AnfAlgo::GetOutputInferShape(cnode, i);
    if (shape.size() <= 1) {
      return true;
    }
  }

  return false;
}

void GetInputBuildInfo(const AnfNodePtr &node, const size_t input_num, const AclAdapterInfo &acl_info,
                       const GeAdapterInfoPtr &ge_info, std::vector<std::string> *input_formats,
                       std::vector<std::string> *input_reshape_types) {
  auto input_info = acl_info.inputs();
  static bool default_format = device::ascend::GetFormatMode() == "1";
  std::vector<size_t> special_inputs;
  for (size_t i = 0; i < input_num; ++i) {
    auto kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(node, i);
    bool input_special_flag = false;
    std::string input_format = AnfAlgo::GetOutputFormat(kernel_with_index.first, kernel_with_index.second);
    auto prev_shape = common::AnfAlgo::GetOutputInferShape(kernel_with_index.first, kernel_with_index.second);
    auto cnode = node->cast<CNodePtr>();
    auto new_format = input_format;
    if (!default_format && acl_info.input_selector().count(i) != 0) {
      auto func = acl_info.input_selector().at(i);
      auto prev_dtype = common::AnfAlgo::GetOutputInferDataType(kernel_with_index.first, kernel_with_index.second);
      new_format = func(prev_dtype, {prev_shape});
    }
    input_format = NeedNDInput(cnode, kernel_with_index.first, new_format, &input_format, &input_special_flag)
                     ? GET_DEFAULT_FORMAT(prev_shape)
                     : input_format;

    (void)input_formats->emplace_back(input_format);
    if (input_special_flag) {
      (void)special_inputs.emplace_back(i);
    }

    if (i >= input_info.size()) {
      continue;
    }
    // Get reshape type.
    auto ge_idx = ge_info->GetGeInputByMsInputIndex(i).index;
    if (ge_idx >= input_info.size()) {
      continue;
    }
    auto special_info = input_info.at(ge_idx);
    if (!special_info.reshape_type.empty()) {
      input_reshape_types->at(i) = special_info.reshape_type;
    }
  }
  if (!special_inputs.empty()) {
    common::AnfAlgo::SetNodeAttr(kAttrAclSpecialInputFormat, MakeValue(special_inputs), node);
  }
}

void GetOutputBuildInfo(const AnfNodePtr &node, const size_t output_num, const AclAdapterInfo &acl_info,
                        const std::vector<std::string> &input_formats, std::vector<std::string> *output_formats) {
  // First use output func.
  auto input_num = common::AnfAlgo::GetInputTensorNum(node);
  static bool default_format = device::ascend::GetFormatMode() == "1";
  if (!default_format && acl_info.output_selector() != nullptr) {
    auto data_type = common::AnfAlgo::GetOutputInferDataType(node, 0);
    std::vector<ShapeVector> input_shapes;
    for (size_t i = 0; i < input_num; ++i) {
      (void)input_shapes.emplace_back(common::AnfAlgo::GetPrevNodeOutputInferShape(node, i));
    }
    auto func = acl_info.output_selector();
    for (size_t i = 0; i < output_num; ++i) {
      const auto &format = func(data_type, input_shapes);
      (void)output_formats->emplace_back(format);
    }
    return;
  }

  // Second use output format.
  if (!acl_info.no_special_outputs()) {
    for (size_t i = 0; i < output_num; ++i) {
      (void)output_formats->emplace_back(acl_info.output_format(i, input_formats));
    }
    return;
  }

  for (size_t i = 0; i < output_num; ++i) {
    auto shape = common::AnfAlgo::GetOutputInferShape(node, i);
    (void)output_formats->emplace_back(GET_DEFAULT_FORMAT(shape));
  }
}

void SetOutputIdentityFlag(const AnfNodePtr &node, const std::vector<std::string> &output_formats) {
  if (device::ascend::GetFormatMode() == "1" && AclHelper::NeedIdentityFlag(output_formats)) {
    common::AnfAlgo::SetNodeAttr(kAttrAclSpecialFormat, MakeValue(true), node);
  }
}

void RefreshRefFormat(const std::unordered_map<size_t, size_t> &ref_map, const std::vector<std::string> &input_formats,
                      std::vector<std::string> *output_formats) {
  if (ref_map.empty()) {
    return;
  }

  for (auto [out_idx, in_idx] : ref_map) {
    if (out_idx >= output_formats->size()) {
      MS_LOG(EXCEPTION) << "Error output index:" << out_idx << " for refresh!";
    }
    if (in_idx >= input_formats.size()) {
      MS_LOG(EXCEPTION) << "Error input index:" << in_idx << " for refresh!";
    }
    output_formats->at(out_idx) = input_formats[in_idx];
  }
}
}  // namespace

void SetParameterFormat(const AnfNodePtr &node, const std::string &format, std::string *old_foramt) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<Parameter>()) {
    if (IsPrimitiveCNode(node, prim::kPrimCast)) {
      auto kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(node, 0);
      if (kernel_with_index.first->isa<Parameter>()) {
        SetParameterFormat(kernel_with_index.first, format, old_foramt);
      } else {
        return;
      }
      auto kernel_info = std::dynamic_pointer_cast<device::KernelInfo>(node->kernel_info_ptr());
      MS_EXCEPTION_IF_NULL(kernel_info);
      auto build_info = kernel_info->GetMutableSelectKernelBuildInfo();
      MS_EXCEPTION_IF_NULL(build_info);
      build_info->SetInputsFormat({format});
      build_info->SetOutputsFormat({format});
      kernel_info->set_select_kernel_build_info(build_info);
    }
    return;
  }
  const auto &output_with_indexs = common::AnfAlgo::GetAllOutputWithIndex(node);
  std::vector<std::string> output_formats{output_with_indexs.size(), format};
  auto kernel_info = std::dynamic_pointer_cast<device::KernelInfo>(node->kernel_info_ptr());
  if (kernel_info == nullptr) {
    kernel_info = std::make_shared<device::KernelInfo>();
    node->set_kernel_info(kernel_info);
  }
  MS_EXCEPTION_IF_NULL(kernel_info);

  auto build_info = kernel_info->GetMutableSelectKernelBuildInfo();
  if (build_info == nullptr) {
    auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
    build_info = builder->Build();
  }
  MS_EXCEPTION_IF_NULL(build_info);
  build_info->SetOutputsFormat(output_formats);
  kernel_info->set_select_kernel_build_info(build_info);
  *old_foramt = format;
}

bool AclHelper::IsPrintDebugString() {
  static bool is_debug = GLogIsDebug();
  return is_debug;
}

bool AclHelper::CheckDefaultSupportFormat(const string &format) {
  static std::set<std::string> default_support = {kOpFormat_DEFAULT, kOpFormat_ND,    kOpFormat_NCHW,
                                                  kOpFormat_NHWC,    kOpFormat_NDHWC, kOpFormat_NCDHW};
  return default_support.find(format) != default_support.end();
}

bool AclHelper::GetMoreDataTypeSupported(TypeId data_type, const std::string &op_type) {
  if (!AclAdapterManager::GetInstance().CheckAclAdapter(op_type)) {
    return false;
  }
  auto acl_info = AclAdapterManager::GetInstance().GetOpInfo(op_type);
  if (acl_info.precision_mode() == FORCE_FP32) {
    if (data_type != kNumberTypeFloat32 && data_type != kNumberTypeFloat) {
      return false;
    }
    return true;
  }
  if (!acl_info.extra_supported_datatype().empty()) {
    if (std::any_of(acl_info.extra_supported_datatype().begin(), acl_info.extra_supported_datatype().end(),
                    [data_type](GeDataType ge_type) { return ConvertGeType(ge_type) == data_type; })) {
      return true;
    }
  }
  return false;
}

KernelType AclHelper::GetKernelInfoByInputs(const CNodePtr &cnode, const std::shared_ptr<GeAdapterInfo> &info) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(info);
  auto input_supported_dtypes = info->input_supported_dtypes();
  size_t num_real_inputs = common::AnfAlgo::GetInputTensorNum(cnode);
  size_t ms_real_idx = 0;  // index of actual input argument
  auto value_depend_indices = ops::GetInputDependValueList(common::AnfAlgo::GetCNodePrimitive(cnode));

  std::vector<int64_t> dyn_input_sizes = {};
  if (common::AnfAlgo::HasNodeAttr(kAttrDynInputSizes, cnode)) {
    dyn_input_sizes = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(cnode, kAttrDynInputSizes);
  }

  for (size_t ms_proto_idx = 0; ms_proto_idx < info->GetNumInputsOfMsOpProto(); ++ms_proto_idx) {
    MS_LOG(DEBUG) << "ms_proto_idx=" << ms_proto_idx << ", ms_real_idx=" << ms_real_idx
                  << ", num_real_inputs=" << num_real_inputs;
    // skip attribute converted input
    if (NeedCheckAttrToInput(cnode, info->attr_input_map(), ms_proto_idx)) {
      MS_LOG(DEBUG) << "Op prototype input idx:" << ms_proto_idx << " is attr to input, skip check";
      continue;
    }

    if (ms_real_idx >= num_real_inputs) {
      break;
    }

    auto opt_ge_input_info = info->GetOptGeInputByMsInputIndex(ms_proto_idx);
    // skip input which will be converted to attribute, or some extra inputs defined by mindspore, such as AvgPoolGrad
    if (!opt_ge_input_info.has_value()) {
      MS_LOG(DEBUG) << "Unsupported op prototype input idx:" << ms_proto_idx
                    << " of node:" << cnode->fullname_with_scope();
      ms_real_idx += 1;
      continue;
    }

    auto &ge_input_info = opt_ge_input_info.value();
    auto base_type = common::AnfAlgo::GetPrevNodeOutputInferDataType(cnode, ms_real_idx);
    bool is_value_depend = value_depend_indices.find(static_cast<int64_t>(ms_real_idx)) != value_depend_indices.end();
    if (is_value_depend) {
      // if the input is value_depend,  verification is performed in the launch and type conversion if necessary
      MS_LOG(DEBUG) << "When input is value_depend, skip it." << cnode->fullname_with_scope();
      ms_real_idx += 1;
      continue;
    }

    if (!std::any_of(
          input_supported_dtypes[ms_proto_idx].begin(), input_supported_dtypes[ms_proto_idx].end(),
          [base_type, ge_input_info](const ::ge::DataType ge_type) { return ConvertGeType(ge_type) == base_type; })) {
      if (base_type == kMetaTypeNone && ge_input_info.type == Ms2GeParamInfo::OPTIONAL) {
        MS_LOG(DEBUG) << "Input is a placeholder, continue!";
        ms_real_idx += 1;
        continue;
      }
      if (GetMoreDataTypeSupported(base_type, info->op_type())) {
        MS_LOG(DEBUG) << "More data type is supported, continue!";
        ms_real_idx += 1;
        continue;
      }
      MS_LOG(DEBUG) << "Unsupported input dtype:" << TypeIdLabel(base_type)
                    << " in ACL, node:" << cnode->fullname_with_scope();
      return UNKNOWN_KERNEL_TYPE;
    }

    if (ge_input_info.type == Ms2GeParamInfo::DYNAMIC) {
      if (dyn_input_sizes.empty()) {
        auto input_node = common::AnfAlgo::GetPrevNodeOutput(cnode, ms_real_idx);
        auto abstract = input_node.first->abstract();
        MS_EXCEPTION_IF_NULL(abstract);
        if (abstract->isa<abstract::AbstractTuple>() || abstract->isa<abstract::AbstractList>()) {
          ms_real_idx += 1;
          continue;
        }
      }
      if (ms_proto_idx >= dyn_input_sizes.size()) {
        MS_LOG(EXCEPTION) << "Attribute " << kAttrDynInputSizes << " of " << cnode->fullname_with_scope() << " is "
                          << dyn_input_sizes << ", of which size is less than " << ms_proto_idx;
      }
      ms_real_idx += dyn_input_sizes[ms_proto_idx];
    } else {
      ms_real_idx += 1;
    }
  }

  return ACL_KERNEL;
}

KernelType AclHelper::GetKernelInfoByOutputs(const AnfNodePtr &node, const std::shared_ptr<GeAdapterInfo> &info) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(info);
  auto output_supported_dtypes = info->output_supported_dtypes();
  auto output_flags = info->GetOutputMappingFlags();
  size_t output_num = ((output_flags & GeTensorInfo::kDynamicParam) == 0) ? info->GetNumOutputsOfMsOpProto()
                                                                          : AnfAlgo::GetOutputTensorNum(node);

  auto is_support = [&node, &output_supported_dtypes](size_t i) {
    auto base_type = common::AnfAlgo::GetOutputInferDataType(node, i);
    if (!std::any_of(output_supported_dtypes[i].begin(), output_supported_dtypes[i].end(),
                     [base_type](const ::ge::DataType ge_type) { return ConvertGeType(ge_type) == base_type; })) {
      MS_LOG(DEBUG) << "Unsupported output dtype:" << TypeIdLabel(base_type)
                    << " in ACL, node:" << node->fullname_with_scope();
      return false;
    }
    return true;
  };

  // operator has dynamic output
  if ((info->GetOutputMappingFlags() & GeTensorInfo::kDynamicParam) != 0) {
    if (info->GetNumOutputsOfMsOpProto() == 1) {
      return is_support(0) ? ACL_KERNEL : UNKNOWN_KERNEL_TYPE;
    } else {
      MS_LOG(EXCEPTION)
        << "Now not support operator containing dynamic output mixed with other outputs, the failed not is "
        << node->fullname_with_scope();
    }
  }

  // operator does not have dynamic output
  for (size_t i = 0; i < output_num; ++i) {
    if (!is_support(i)) {
      return UNKNOWN_KERNEL_TYPE;
    }
  }

  return ACL_KERNEL;
}

KernelType AclHelper::GetKernelInfoFromGe(const AnfNodePtr &node, ErrorAclType *err_type) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  std::string name = GetCNodeFuncName(cnode);
  if (common::AnfAlgo::IsCommunicationOp(node)) {
    *err_type = kNormalOp;
    return HCCL_KERNEL;
  }

  auto info = GeAdapterManager::GetInstance().GetInfo(name, true);
  if (info == nullptr) {
    *err_type = kUnknownOp;
    MS_LOG(DEBUG) << "Unsupported op type on acl, node name: " << node->fullname_with_scope();
    return UNKNOWN_KERNEL_TYPE;
  }

  // check whether all inputs are matched
  if (GetKernelInfoByInputs(cnode, info) == UNKNOWN_KERNEL_TYPE) {
    *err_type = kInValidType;
    return UNKNOWN_KERNEL_TYPE;
  }

  *err_type = kNormalOp;
  return ACL_KERNEL;
}

bool AclHelper::IsInputDtypeSupport(const std::string &kernel_name, TypeId base_type, size_t idx) {
  auto info = GeAdapterManager::GetInstance().GetInfo(kernel_name, true);
  MS_EXCEPTION_IF_NULL(info);
  auto input_supported_dtypes = info->input_supported_dtypes();
  if (idx >= info->GetNumInputsOfMsOpProto()) {
    // this branch represent input_attr_map, didn't need check
    return true;
  }
  if (!std::any_of(input_supported_dtypes[idx].begin(), input_supported_dtypes[idx].end(),
                   [base_type](const ::ge::DataType ge_type) { return ConvertGeType(ge_type) == base_type; })) {
    return false;
  }
  return true;
}

void AclHelper::GetValidKernelBuildInfo(const AnfNodePtr &node, std::vector<std::string> *input_formats,
                                        std::vector<std::string> *output_formats,
                                        std::vector<std::string> *input_reshape_types,
                                        std::vector<std::string> *output_reshape_types) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(input_formats);
  MS_EXCEPTION_IF_NULL(output_formats);
  MS_EXCEPTION_IF_NULL(input_reshape_types);
  MS_EXCEPTION_IF_NULL(output_reshape_types);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  std::string name = GetCNodeFuncName(cnode);
  auto info = GeAdapterManager::GetInstance().GetInfo(name, true);
  auto op_type = info->op_type();

  input_formats->clear();
  output_formats->clear();
  input_reshape_types->clear();
  output_reshape_types->clear();
  size_t input_num = common::AnfAlgo::GetInputTensorNum(node);
  size_t output_num = AnfUtils::GetOutputTensorNum(node);
  input_reshape_types->assign(input_num, "");
  output_reshape_types->assign(output_num, "");

  if (!AclAdapterManager::GetInstance().CheckAclAdapter(op_type)) {
    std::vector<size_t> special_inputs;
    for (size_t i = 0; i < input_num; ++i) {
      auto kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(node, i);
      bool input_special_flag = false;
      auto input_format = AnfAlgo::GetOutputFormat(kernel_with_index.first, kernel_with_index.second);
      auto prev_shape = common::AnfAlgo::GetOutputInferShape(kernel_with_index.first, kernel_with_index.second);
      input_format = NeedNDInput(cnode, kernel_with_index.first, input_format, &input_format, &input_special_flag)
                       ? GET_DEFAULT_FORMAT(prev_shape)
                       : input_format;
      (void)input_formats->emplace_back(input_format);
      if (input_special_flag) {
        (void)special_inputs.emplace_back(i);
      }
    }
    // Input and output number same's op forward.
    if (NeedNDOutput(cnode, input_num, output_num, *input_formats)) {
      for (size_t i = 0; i < output_num; ++i) {
        auto shape = common::AnfAlgo::GetOutputInferShape(node, i);
        (void)output_formats->emplace_back(GET_DEFAULT_FORMAT(shape));
      }
    } else {
      if (output_num == 1) {
        output_formats->emplace_back(input_formats->at(0));
      } else {
        output_formats->assign(input_formats->begin(), input_formats->end());
      }
      SetOutputIdentityFlag(node, *output_formats);
    }

    if (!special_inputs.empty()) {
      common::AnfAlgo::SetNodeAttr(kAttrAclSpecialInputFormat, MakeValue(special_inputs), node);
    }
    RefreshRefFormat(info->GetRefMappingInfo(), *input_formats, output_formats);
    return;
  }

  auto acl_info = AclAdapterManager::GetInstance().GetOpInfo(op_type);
  GetInputBuildInfo(node, input_num, acl_info, info, input_formats, input_reshape_types);
  GetOutputBuildInfo(node, output_num, acl_info, *input_formats, output_formats);
  SetOutputIdentityFlag(node, *output_formats);
  RefreshRefFormat(info->GetRefMappingInfo(), *input_formats, output_formats);
}

void AclHelper::PaddingOriShape(const std::string &name, size_t idx, const std::string &format, ShapeVector *shape) {
  MS_EXCEPTION_IF_NULL(shape);
  auto info = GeAdapterManager::GetInstance().GetInfo(name, true);
  auto op_type = info->op_type();
  if (!AclAdapterManager::GetInstance().CheckAclAdapter(op_type)) {
    return;
  }
  auto acl_info = AclAdapterManager::GetInstance().GetOpInfo(op_type);
  auto info_list = acl_info.inputs();
  if (info_list.empty() || idx >= info_list.size()) {
    return;
  }
  auto ge_idx = info->GetGeInputByMsInputIndex(idx).index;
  auto special_iter = info_list.find(ge_idx);
  if (special_iter == info_list.end() || special_iter->second.ori_format.empty()) {
    return;
  }
  if (!special_iter->second.ori_format.empty() && format == kOpFormat_NCHW && shape->size() < kDim4) {
    *shape = trans::PaddingShape(*shape, kOpFormat_NCHW, special_iter->second.reshape_type);
  }
}

std::string AclHelper::ConvertOriginShapeAndFormat(const std::string &name, size_t idx, const std::string &dev_format,
                                                   ShapeVector *shape) {
  MS_EXCEPTION_IF_NULL(shape);
  auto info = GeAdapterManager::GetInstance().GetInfo(name, true);
  auto op_type = info->op_type();
  std::string ret_format = (shape->size() == kDim4) ? kOpFormat_NCHW : kOpFormat_DEFAULT;
  // case0: normal
  if (!AclAdapterManager::GetInstance().CheckAclAdapter(op_type)) {
    return ret_format;
  }
  // case1: 3d operator
  auto acl_info = AclAdapterManager::GetInstance().GetOpInfo(op_type);
  if (acl_info.is_3d()) {
    *shape = trans::PaddingShape(*shape, kOpFormat_NCDHW);
    return kOpFormat_NCDHW;
  }
  if (acl_info.is_need_pad_no_shape() && shape->empty()) {
    shape->push_back(1);
  }
  // case2: no special config
  auto info_list = acl_info.inputs();
  if (info_list.empty() || idx >= info_list.size()) {
    return ret_format;
  }
  auto ge_idx = info->GetGeInputByMsInputIndex(idx).index;
  auto special_iter = info_list.find(ge_idx);
  if (special_iter == info_list.end() || special_iter->second.ori_format.empty()) {
    return ret_format;
  }
  // case3: if config input ori format or dev_format is special
  if (!special_iter->second.ori_format.empty() || !CheckDefaultSupportFormat(dev_format)) {
    if (special_iter->second.ori_format[0] == kOpFormat_ND) {
      return kOpFormat_ND;
    }
    if (ret_format == kOpFormat_DEFAULT && shape->size() < kDim4) {
      *shape = trans::PaddingShape(*shape, kOpFormat_NCHW, special_iter->second.reshape_type);
      ret_format = kOpFormat_NCHW;
    }
  }
  return ret_format;
}

bool AclHelper::NeedCheckAttrToInput(const CNodePtr &node,
                                     const mindspore::HashMap<size_t, std::string> &attr_input_map, size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  if (attr_input_map.count(index) == 0) {
    return false;
  }

  const auto &attr_name = attr_input_map.at(index);
  if (common::AnfAlgo::HasNodeAttr(attr_name, node)) {
    return true;
  }
  return false;
}

std::string AclHelper::GetFormatFromAttr(const PrimitivePtr &primitive) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto &attrs = primitive->attrs();
  std::string format;
  if (attrs.count("format") != 0) {
    auto attr_value = attrs.at("format");
    if (attr_value->isa<StringImm>()) {
      format = GetValue<std::string>(attr_value);
    } else {
      MS_LOG(DEBUG) << "The attr format is not a valid value.";
    }
  }
  return format;
}

bool AclHelper::GetDefaultFormatFlagFromAttr(const PrimitivePtr &primitive, bool is_input) {
  MS_EXCEPTION_IF_NULL(primitive);
  bool is_default = true;
  auto key = is_input ? kAttrInputDefaultFormat : kAttrOutputDefaultFormat;
  auto attrs = primitive->attrs();
  if (attrs.count(key) != 0) {
    auto attr_value = attrs.at(key);
    if (attr_value->isa<BoolImm>()) {
      is_default = GetValue<bool>(attr_value);
    } else {
      MS_LOG(DEBUG) << "The attr: " << key << " is not a valid value.";
    }
  }
  return is_default;
}

int64_t AclHelper::GetFracZGroupFromAttr(const PrimitivePtr &primitive) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto attrs = primitive->attrs();
  int64_t fracz_group = 1;
  if (attrs.count(kAttrFracZGroup) != 0) {
    auto attr_value = attrs.at(kAttrFracZGroup);
    if (attr_value->isa<Int64Imm>()) {
      fracz_group = GetValue<int64_t>(attr_value);
    } else {
      MS_LOG(DEBUG) << "The FracZGroup attr is not a valid value.";
    }
  }
  return fracz_group;
}

bool AclHelper::IsNopNode(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  static mindspore::HashSet<std::string> nop_nodes = {prim::kPrimReshape->name(), prim::kPrimExpandDims->name(),
                                                      prim::kPrimSqueeze->name(), prim::kPrimFlatten->name(),
                                                      prim::kPrimFlattenGrad->name()};
  auto op_name = common::AnfAlgo::GetCNodeName(node);
  return (nop_nodes.find(op_name) != nop_nodes.end());
}

bool AclHelper::NeedIdentityFlag(const std::vector<std::string> &formats) {
  return std::any_of(formats.begin(), formats.end(),
                     [](const auto &format) { return !AclHelper::CheckDefaultSupportFormat(format); });
}
}  // namespace transform
}  // namespace mindspore
