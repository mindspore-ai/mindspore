/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/internal/internal_helper.h"

#include <vector>
#include <set>
#include <unordered_map>
#include "primitive/math_op_name.h"
#include "primitive/nn_optimizer_op_name.h"
#include "mindspore/ops/ops_utils/op_constants.h"
#include "include/utils/anfalgo.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel_info.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel_build_info.h"
#include "mindapi/base/type_id.h"
#include "utils/log_adapter.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"

namespace mindspore {
namespace kernel {
std::string TransInternalOpName(const std::string &ms_op_name) {
  auto internal_name = NameMapper::GetInstance().GetInternalName(ms_op_name);
  if (internal_name.empty()) {
    MS_LOG(EXCEPTION) << "Op " << ms_op_name << " is supported in Internal, but the name is not mapped";
  }
  return internal_name;
}

internal::DataType TransInternalDataType(TypeId ms_type) {
  static const std::unordered_map<TypeId, internal::DataType> kMSTypeToInternalType = {
    {kNumberTypeFloat16, internal::DataType::kTypeFloat16},
    {kNumberTypeBFloat16, internal::DataType::kTypeBF16},
    {kNumberTypeFloat32, internal::DataType::kTypeFloat32},
    {kNumberTypeDouble, internal::DataType::kTypeFloat64},
    {kNumberTypeInt32, internal::DataType::kTypeInt32},
    {kNumberTypeUInt32, internal::DataType::kTypeUint32},
    {kNumberTypeInt16, internal::DataType::kTypeInt16},
    {kNumberTypeUInt16, internal::DataType::kTypeUint16},
    {kNumberTypeInt8, internal::DataType::kTypeInt8},
    {kNumberTypeUInt8, internal::DataType::kTypeUint8},
    {kNumberTypeInt64, internal::DataType::kTypeInt64},
    {kNumberTypeUInt64, internal::DataType::kTypeUint64},
    {kNumberTypeComplex64, internal::DataType::kTypeComplex64},
    {kNumberTypeComplex128, internal::DataType::kTypeComplex128},
    {kNumberTypeBool, internal::DataType::kTypeBool},
    {kMetaTypeNone, internal::DataType::kTypeNone},
  };

  auto iter = kMSTypeToInternalType.find(ms_type);
  if (iter == kMSTypeToInternalType.end()) {
    MS_LOG(INFO) << "Type " << ms_type << " is not supported in Internal";
    return internal::DataType::kTypeUnknown;
  }

  return iter->second;
}

internal::TensorFormat TransInternalFormat(Format format) {
  static const std::unordered_map<Format, internal::TensorFormat> kMSFormatToInternalFormat = {
    {DEFAULT_FORMAT, internal::TensorFormat::kFormatND},
    {NCHW, internal::TensorFormat::kFormatNCHW},
    {NHWC, internal::TensorFormat::kFormatNHWC},
    {ND, internal::TensorFormat::kFormatND},
    {NC1HWC0, internal::TensorFormat::kFormatNC1HWC0},
    {FRACTAL_Z, internal::TensorFormat::kFormatFRACTAL_Z},
    {NC1HWC0_C04, internal::TensorFormat::kFormatNC1HWC0_C04},
    {HWCN, internal::TensorFormat::kFormatHWCN},
    {NDHWC, internal::TensorFormat::kFormatNDHWC},
    {FRACTAL_NZ, internal::TensorFormat::kFormatFRACTAL_NZ},
    {NCDHW, internal::TensorFormat::kFormatNCDHW},
    {NDC1HWC0, internal::TensorFormat::kFormatNDC1HWC0},
    {FRACTAL_Z_3D, internal::TensorFormat::kFormatFRACTAL_Z_3D},
  };

  auto iter = kMSFormatToInternalFormat.find(format);
  if (iter == kMSFormatToInternalFormat.end()) {
    MS_LOG(EXCEPTION) << "Format " << format << " is not supported in Internal";
  }

  switch (format) {
    case NCHW:
    case NHWC:
    case NDHWC:
    case NCDHW:
      // some op not support NCHW, NHWC, ... format, current return ND format
      return internal::TensorFormat::kFormatND;
    default:
      return iter->second;
  }
}

bool CheckDefaultSupportFormat(const std::string &format) {
  static std::set<std::string> default_support = {kOpFormat_DEFAULT, kOpFormat_ND,    kOpFormat_NCHW,
                                                  kOpFormat_NHWC,    kOpFormat_NDHWC, kOpFormat_NCDHW};
  return default_support.find(format) != default_support.end();
}

void SetParameterFormat(const AnfNodePtr &node, const std::string &format, std::string *old_format) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<Parameter>()) {
    if (IsPrimitiveCNode(node, prim::kPrimCast)) {
      auto kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(node, 0);
      if (kernel_with_index.first->isa<Parameter>()) {
        SetParameterFormat(kernel_with_index.first, format, old_format);
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
  auto parameter = node->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(parameter);
  bool is_weight = common::AnfAlgo::IsParameterWeight(parameter);
  if (!is_weight) {
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
  MS_EXCEPTION_IF_NULL(old_format);
  *old_format = format;
}
}  // namespace kernel
}  // namespace mindspore
