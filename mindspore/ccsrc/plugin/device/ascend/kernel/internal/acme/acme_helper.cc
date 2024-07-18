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

#include "plugin/device/ascend/kernel/internal/acme/acme_helper.h"

#include <unordered_map>
#include "ops/math_op_name.h"
#include "ops/nn_optimizer_op_name.h"
#include "mindapi/base/type_id.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace kernel {
std::string TransAcmeOpName(const std::string &ms_op_name) {
  auto acme_name = NameMapper::GetInstance().GetAcmeName(ms_op_name);
  if (acme_name.empty()) {
    MS_LOG(EXCEPTION) << "Op " << ms_op_name << " is supported in Acme, but the name is not mapped";
  }
  return acme_name;
}

acme::DataType TransAcmeDataType(TypeId ms_type) {
  static const std::unordered_map<TypeId, acme::DataType> kMSTypeToAcmeType = {
    {kNumberTypeFloat16, acme::DataType::kTypeFloat16},     {kNumberTypeBFloat16, acme::DataType::kTypeBF16},
    {kNumberTypeFloat32, acme::DataType::kTypeFloat32},     {kNumberTypeDouble, acme::DataType::kTypeFloat64},
    {kNumberTypeInt32, acme::DataType::kTypeInt32},         {kNumberTypeUInt32, acme::DataType::kTypeUint32},
    {kNumberTypeInt16, acme::DataType::kTypeInt16},         {kNumberTypeUInt16, acme::DataType::kTypeUint16},
    {kNumberTypeInt8, acme::DataType::kTypeInt8},           {kNumberTypeUInt8, acme::DataType::kTypeUint8},
    {kNumberTypeInt64, acme::DataType::kTypeInt64},         {kNumberTypeUInt64, acme::DataType::kTypeUint64},
    {kNumberTypeComplex64, acme::DataType::kTypeComplex64}, {kNumberTypeComplex128, acme::DataType::kTypeComplex128},
    {kNumberTypeBool, acme::DataType::kTypeBool},
  };

  auto iter = kMSTypeToAcmeType.find(ms_type);
  if (iter == kMSTypeToAcmeType.end()) {
    MS_LOG(EXCEPTION) << "Type " << ms_type << " is not supported in Acme";
  }

  return iter->second;
}

acme::TensorFormat TransAcmeFormat(Format format) {
  static const std::unordered_map<Format, acme::TensorFormat> kMSFormatToAcmeFormat = {
    {DEFAULT_FORMAT, acme::TensorFormat::kFormatND},
    {NCHW, acme::TensorFormat::kFormatNCHW},
    {NHWC, acme::TensorFormat::kFormatNHWC},
    {ND, acme::TensorFormat::kFormatND},
    {NC1HWC0, acme::TensorFormat::kFormatNC1HWC0},
    {FRACTAL_Z, acme::TensorFormat::kFormatFRACTAL_Z},
    {NC1HWC0_C04, acme::TensorFormat::kFormatNC1HWC0_C04},
    {HWCN, acme::TensorFormat::kFormatHWCN},
    {NDHWC, acme::TensorFormat::kFormatNDHWC},
    {FRACTAL_NZ, acme::TensorFormat::kFormatFRACTAL_NZ},
    {NCDHW, acme::TensorFormat::kFormatNCDHW},
    {NDC1HWC0, acme::TensorFormat::kFormatNDC1HWC0},
    {FRACTAL_Z_3D, acme::TensorFormat::kFormatFRACTAL_Z_3D},
  };

  auto iter = kMSFormatToAcmeFormat.find(format);
  if (iter == kMSFormatToAcmeFormat.end()) {
    MS_LOG(EXCEPTION) << "Format " << format << " is not supported in Acme";
  }

  return iter->second;
}

}  // namespace kernel
}  // namespace mindspore
