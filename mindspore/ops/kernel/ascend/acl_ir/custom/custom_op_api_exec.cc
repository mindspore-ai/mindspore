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

#include "kernel/ascend/acl_ir/custom/custom_op_api_exec.h"

namespace mindspore::device::ascend {
namespace {
void CheckParamsSize(size_t params_size, size_t input_output_types_size) {
  if (params_size < kSizeTwo) {
    MS_LOG(EXCEPTION) << "Params number is invalid, it should greater than 2, but got " << params_size;
  }
  if (params_size - kSizeTwo != input_output_types_size) {
    MS_LOG(EXCEPTION) << "Params size " << (params_size - kSizeTwo) << " is not equal to input_output_types_ size "
                      << input_output_types_size;
  }
}
void ReleaseConvertTypesImpl(const std::vector<void *> &params,
                             const std::vector<CustomSupportType> &input_output_types) {
  CheckParamsSize(params.size(), input_output_types.size());
  for (size_t i = 0; i < params.size() - kIndex2; i++) {
    auto type = input_output_types[i];
    MS_LOG(DEBUG) << "Release convert types input-" << i
                  << ", type: " << mindspore::kernel::custom::custom_supported_type_to_string.at(type);
    MS_VLOG(VL_CUSTOM_OP) << "Release convert types input-" << i
                          << ", type: " << mindspore::kernel::custom::custom_supported_type_to_string.at(type);
    switch (type) {
      case CustomSupportType::kTypeTensor: {
        const auto &acl_tensor = static_cast<aclTensor *>(params[i]);
        MS_EXCEPTION_IF_NULL(acl_tensor);
        Release(acl_tensor);
        break;
      }
      case CustomSupportType::kTypeTensorList: {
        const auto &acl_tensor_list = static_cast<aclTensorList *>(params[i]);
        MS_EXCEPTION_IF_NULL(acl_tensor_list);
        Release(acl_tensor_list);
        break;
      }
      case CustomSupportType::kTypeScalar: {
        const auto &acl_scalar = static_cast<aclScalar *>(params[i]);
        MS_EXCEPTION_IF_NULL(acl_scalar);
        Release(acl_scalar);
        break;
      }
      case CustomSupportType::kTypeIntArray: {
        const auto acl_int_array = static_cast<aclIntArray *>(params[i]);
        if (acl_int_array != nullptr) {
          Release(acl_int_array);
        }
        break;
      }
      case CustomSupportType::kTypeBoolArray: {
        const auto acl_bool_array = static_cast<aclBoolArray *>(params[i]);
        if (acl_bool_array != nullptr) {
          Release(acl_bool_array);
        }
        break;
      }
      case CustomSupportType::kTypeFloatArray: {
        const auto acl_float_array = static_cast<aclFloatArray *>(params[i]);
        if (acl_float_array != nullptr) {
          Release(acl_float_array);
        }
        break;
      }
      default:
        break;
    }
  }
  Release(params[params.size() - kIndex2]);
  Release(params[params.size() - kIndex1]);
}
}  // namespace

std::vector<ShapeVector> CustomGraphCache::FillShapeListFromTuple(const std::vector<void *> &params) {
  CheckParamsSize(params.size(), input_output_types_.size());
  std::vector<ShapeVector> shape_list;
  for (size_t i = 0; i < params.size(); i++) {
    auto type = input_output_types_[i];
    if (type == CustomSupportType::kTypeTensor) {
      const auto &acl_tensor = static_cast<aclTensor *>(params[i]);
      MS_EXCEPTION_IF_NULL(acl_tensor);
      GetShape(acl_tensor, &shape_list);
    } else {
      (void)shape_list.emplace_back(ShapeVector());
    }
  }
  return shape_list;
}

void CustomGraphCache::ReleaseConvertTypes(const std::vector<void *> &params) {
  ReleaseConvertTypesImpl(params, input_output_types_);
}

void CustomGraphCache::UpdateAddressForTensor(aclOpExecutor *executor,
                                              const std::vector<std::vector<void *>> &address_list,
                                              const std::vector<void *> &params) {
  CheckParamsSize(params.size(), input_output_types_.size());
  if (address_list.size() != params.size() - kSizeTwo) {
    MS_LOG(EXCEPTION) << "The size of address_list " << address_list.size()
                      << " is not equal to params size: " << (params.size() - kSizeTwo);
  }
  size_t valid_index = 0;
  for (size_t i = 0; i < params.size() - kIndex2; i++) {
    auto type = input_output_types_[i];
    MS_LOG(DEBUG) << "Update tensor address for input-" << i
                  << ", type: " << mindspore::kernel::custom::custom_supported_type_to_string.at(type);
    MS_VLOG(VL_CUSTOM_OP) << "Update tensor address for input-" << i
                          << ", type: " << mindspore::kernel::custom::custom_supported_type_to_string.at(type);
    switch (type) {
      case CustomSupportType::kTypeTensor: {
        const auto &acl_tensor = static_cast<aclTensor *>(params[i]);
        MS_EXCEPTION_IF_NULL(acl_tensor);
        UpdateAddress(executor, acl_tensor, address_list[i], &valid_index);
        break;
      }
      case CustomSupportType::kTypeTensorList: {
        const auto &acl_tensor_list = static_cast<aclTensorList *>(params[i]);
        MS_EXCEPTION_IF_NULL(acl_tensor_list);
        UpdateAddress(executor, acl_tensor_list, address_list[i], &valid_index);
        break;
      }
      case CustomSupportType::kTypeScalar: {
        const auto &acl_scalar = static_cast<aclScalar *>(params[i]);
        MS_EXCEPTION_IF_NULL(acl_scalar);
        UpdateAddress(executor, acl_scalar, address_list[i], &valid_index);
        break;
      }
      case CustomSupportType::kTypeIntArray: {
        auto acl_int_array = static_cast<aclIntArray *>(params[i]);
        UpdateAddress(executor, acl_int_array, address_list[i], &valid_index);
        break;
      }
      case CustomSupportType::kTypeBoolArray: {
        auto acl_bool_array = static_cast<aclBoolArray *>(params[i]);
        UpdateAddress(executor, acl_bool_array, address_list[i], &valid_index);
        break;
      }
      case CustomSupportType::kTypeFloatArray: {
        auto acl_float_array = static_cast<aclFloatArray *>(params[i]);
        UpdateAddress(executor, acl_float_array, address_list[i], &valid_index);
        break;
      }
      default:
        break;
    }
  }
}

void CustomReleaseCall::ReleaseConvertTypes(std::vector<void *> params) {
  ReleaseConvertTypesImpl(params, input_output_types_);
}

}  // namespace mindspore::device::ascend
