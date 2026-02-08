/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/symmetric_memory/symmetric_memory_helper.h"

#include <string>
#include <vector>
#include <set>
#include <unordered_map>
#include "mindspore/ops/op_def/math_op_name.h"
#include "mindspore/ops/op_def/nn_optimizer_op_name.h"
#include "mindspore/ops/ops_utils/op_constants.h"
#include "include/utils/anfalgo.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel_info.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel_build_info.h"
#include "mindapi/base/type_id.h"
#include "utils/log_adapter.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_c.h"

namespace mindspore {
namespace kernel {
std::string TransSymmetricMemoryOpName(const std::string &ms_op_name) {
  auto symmetric_memory_name = NameMapper::GetInstance().GetSymmetricMemoryName(ms_op_name);
  if (symmetric_memory_name.empty()) {
    MS_LOG(EXCEPTION) << "Op " << ms_op_name << " is supported in SymmetricMemory, but the name is not mapped";
  }
  return symmetric_memory_name;
}

symmetricmemory::DataType TransSymmetricMemoryDataType(TypeId ms_type) {
  static const std::unordered_map<TypeId, symmetricmemory::DataType> kMSTypeToSymmetricMemoryType = {
    {kNumberTypeFloat16, symmetricmemory::DataType::kTypeFloat16},
    {kNumberTypeBFloat16, symmetricmemory::DataType::kTypeBF16},
    {kNumberTypeFloat32, symmetricmemory::DataType::kTypeFloat32},
    {kNumberTypeDouble, symmetricmemory::DataType::kTypeFloat64},
    {kNumberTypeInt32, symmetricmemory::DataType::kTypeInt32},
    {kNumberTypeUInt32, symmetricmemory::DataType::kTypeUint32},
    {kNumberTypeInt16, symmetricmemory::DataType::kTypeInt16},
    {kNumberTypeUInt16, symmetricmemory::DataType::kTypeUint16},
    {kNumberTypeInt8, symmetricmemory::DataType::kTypeInt8},
    {kNumberTypeUInt8, symmetricmemory::DataType::kTypeUint8},
    {kNumberTypeInt64, symmetricmemory::DataType::kTypeInt64},
    {kNumberTypeUInt64, symmetricmemory::DataType::kTypeUint64},
    {kNumberTypeComplex64, symmetricmemory::DataType::kTypeComplex64},
    {kNumberTypeComplex128, symmetricmemory::DataType::kTypeComplex128},
    {kNumberTypeBool, symmetricmemory::DataType::kTypeBool},
    {kMetaTypeNone, symmetricmemory::DataType::kTypeNone},
  };

  auto iter = kMSTypeToSymmetricMemoryType.find(ms_type);
  if (iter == kMSTypeToSymmetricMemoryType.end()) {
    MS_LOG(INFO) << "Type " << ms_type << " is not supported in SymmetricMemory";
    return symmetricmemory::DataType::kTypeUnknown;
  }

  return iter->second;
}

symmetricmemory::TensorFormat TransSymmetricMemoryFormat(Format format) {
  static const std::unordered_map<Format, symmetricmemory::TensorFormat> kMSFormatToSymmetricMemoryFormat = {
    {DEFAULT_FORMAT, symmetricmemory::TensorFormat::kFormatND},
    {NCHW, symmetricmemory::TensorFormat::kFormatNCHW},
    {NHWC, symmetricmemory::TensorFormat::kFormatNHWC},
    {ND, symmetricmemory::TensorFormat::kFormatND},
    {NC1HWC0, symmetricmemory::TensorFormat::kFormatNC1HWC0},
    {FRACTAL_Z, symmetricmemory::TensorFormat::kFormatFRACTAL_Z},
    {NC1HWC0_C04, symmetricmemory::TensorFormat::kFormatNC1HWC0_C04},
    {HWCN, symmetricmemory::TensorFormat::kFormatHWCN},
    {NDHWC, symmetricmemory::TensorFormat::kFormatNDHWC},
    {FRACTAL_NZ, symmetricmemory::TensorFormat::kFormatFRACTAL_NZ},
    {NCDHW, symmetricmemory::TensorFormat::kFormatNCDHW},
    {NDC1HWC0, symmetricmemory::TensorFormat::kFormatNDC1HWC0},
    {FRACTAL_Z_3D, symmetricmemory::TensorFormat::kFormatFRACTAL_Z_3D},
  };

  auto iter = kMSFormatToSymmetricMemoryFormat.find(format);
  if (iter == kMSFormatToSymmetricMemoryFormat.end()) {
    MS_LOG(EXCEPTION) << "Format " << format << " is not supported in SymmetricMemory";
  }

  switch (format) {
    case NCHW:
    case NHWC:
    case NDHWC:
    case NCDHW:
      // some op not support NCHW, NHWC, ... format, current return ND format
      return symmetricmemory::TensorFormat::kFormatND;
    default:
      return iter->second;
  }
}

bool CheckDefaultSupportFormat(const std::string &format) {
  static std::set<std::string> default_support = {kOpFormat_DEFAULT, kOpFormat_ND,    kOpFormat_NCHW,
                                                  kOpFormat_NHWC,    kOpFormat_NDHWC, kOpFormat_NCDHW};
  return default_support.find(format) != default_support.end();
}

}  // namespace kernel
}  // namespace mindspore
