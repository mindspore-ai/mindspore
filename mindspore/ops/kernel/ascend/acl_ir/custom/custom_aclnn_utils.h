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

#ifndef MINDSPORE_OPS_KERNEL_ASCEND_ACL_IR_CUSTOM_CUSTOM_ACLNN_UTILS_H
#define MINDSPORE_OPS_KERNEL_ASCEND_ACL_IR_CUSTOM_CUSTOM_ACLNN_UTILS_H

#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "hccl/hccl.h"
#include "acl/acl_op_compiler.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "ir/anf.h"
#include "kernel/ascend/visible.h"

namespace mindspore::kernel::custom {
constexpr auto kTypeAclNNTensor = "aclTensor*";
constexpr auto kTypeAclNNTensorList = "aclTensorList*";
constexpr auto kTypeAclNNScalar = "aclScalar*";
constexpr auto kTypeAclNNIntArray = "aclIntArray*";
constexpr auto kTypeAclNNBoolArray = "aclBoolArray*";
constexpr auto kTypeAclNNFloatArray = "aclFloatArray*";
constexpr auto kTypeAclNNDType = "aclDataType";
constexpr auto kTypeAclNNFloat = "float";
constexpr auto kTypeAclNNInt = "int64_t";
constexpr auto kTypeAclNNUInt = "uint64_t";
constexpr auto kTypeAclNNDouble = "double";
constexpr auto kTypeAclNNBool = "bool";
constexpr auto kTypeAclNNString = "char*";

enum CustomSupportType : int8_t {
  UNKNOWN = -1,
  kTypeTensor,
  kTypeTensorList,
  kTypeDType,
  kTypeScalar,
  kTypeBool,
  kTypeBoolArray,
  kTypeInt,
  kTypeIntArray,
  kTypeFloat,
  kTypeFloatArray,
  kTypeUInt,
  kTypeDouble,
  kTypeString
};

const std::unordered_map<CustomSupportType, std::string> custom_supported_type_to_string = {
  {CustomSupportType::kTypeTensor, kTypeAclNNTensor},
  {CustomSupportType::kTypeTensorList, kTypeAclNNTensorList},
  {CustomSupportType::kTypeScalar, kTypeAclNNScalar},
  {CustomSupportType::kTypeIntArray, kTypeAclNNIntArray},
  {CustomSupportType::kTypeBoolArray, kTypeAclNNBoolArray},
  {CustomSupportType::kTypeFloatArray, kTypeAclNNFloatArray},
  {CustomSupportType::kTypeFloat, kTypeAclNNFloat},
  {CustomSupportType::kTypeDouble, kTypeAclNNDouble},
  {CustomSupportType::kTypeInt, kTypeAclNNInt},
  {CustomSupportType::kTypeUInt, kTypeAclNNUInt},
  {CustomSupportType::kTypeBool, kTypeAclNNBool},
  {CustomSupportType::kTypeString, kTypeAclNNString},
  {CustomSupportType::kTypeDType, kTypeAclNNDType}};

const std::map<std::string, CustomSupportType> string_to_custom_supported_type = {
  {kTypeAclNNTensor, CustomSupportType::kTypeTensor},
  {kTypeAclNNTensorList, CustomSupportType::kTypeTensorList},
  {kTypeAclNNScalar, CustomSupportType::kTypeScalar},
  {kTypeAclNNIntArray, CustomSupportType::kTypeIntArray},
  {kTypeAclNNBoolArray, CustomSupportType::kTypeBoolArray},
  {kTypeAclNNFloatArray, CustomSupportType::kTypeFloatArray},
  {kTypeAclNNFloat, CustomSupportType::kTypeFloat},
  {kTypeAclNNDouble, CustomSupportType::kTypeDouble},
  {kTypeAclNNInt, CustomSupportType::kTypeInt},
  {kTypeAclNNUInt, CustomSupportType::kTypeInt},
  {kTypeAclNNBool, CustomSupportType::kTypeBool},
  {kTypeAclNNString, CustomSupportType::kTypeString},
  {kTypeAclNNDType, CustomSupportType::kTypeDType}};

}  // namespace mindspore::kernel::custom
#endif  // MINDSPORE_OPS_KERNEL_ASCEND_ACL_IR_CUSTOM_CUSTOM_ACLNN_UTILS_H
