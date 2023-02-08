/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "c_api/include/attribute.h"
#include <memory>
#include "c_api/src/helper.h"
#include "c_api/src/common.h"
#include "ir/tensor.h"

PrimitivePtr GetOpPrim(ResMgrHandle res_mgr, NodeHandle node) {
  auto src_node = GetSrcPtr<CNodePtr>(res_mgr, node);
  auto node_input = src_node->input(0);
  if (node_input == nullptr) {
    MS_LOG(ERROR) << "The node's input is nullptr.";
    return nullptr;
  }
  auto prim_node = node_input->cast<ValueNodePtr>();
  if (prim_node == nullptr) {
    MS_LOG(ERROR) << "The node's input is with invalid type.";
    return nullptr;
  }
  auto node_value = prim_node->value();
  if (node_value == nullptr) {
    MS_LOG(ERROR) << "The node's value is nullptr.";
    return nullptr;
  }
  auto prim = node_value->cast<PrimitivePtr>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "The node's value is with invalid type.";
    return nullptr;
  }
  return prim;
}

STATUS MSOpSetScalarAttrFloat32(ResMgrHandle res_mgr, NodeHandle op, const char *attr_name, float value) {
  if (res_mgr == nullptr || op == nullptr || attr_name == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [op] or [attr_name] is nullptr.";
    return RET_NULL_PTR;
  }
  auto prim = GetOpPrim(res_mgr, op);
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Get primitive node failed";
    return RET_NULL_PTR;
  }
  prim->set_attr(attr_name, mindspore::MakeValue(value));
  return RET_OK;
}

STATUS MSOpSetScalarAttrBool(ResMgrHandle res_mgr, NodeHandle op, const char *attr_name, bool value) {
  if (res_mgr == nullptr || op == nullptr || attr_name == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [op] or [attr_name] is nullptr.";
    return RET_NULL_PTR;
  }
  auto prim = GetOpPrim(res_mgr, op);
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Get primitive node failed";
    return RET_NULL_PTR;
  }
  prim->set_attr(attr_name, mindspore::MakeValue(value));
  return RET_OK;
}

STATUS MSOpSetScalarAttrInt32(ResMgrHandle res_mgr, NodeHandle op, const char *attr_name, int32_t value) {
  if (res_mgr == nullptr || op == nullptr || attr_name == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [op] or [attr_name] is nullptr.";
    return RET_NULL_PTR;
  }
  auto prim = GetOpPrim(res_mgr, op);
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Get primitive node failed";
    return RET_NULL_PTR;
  }
  prim->set_attr(attr_name, mindspore::MakeValue(value));
  return RET_OK;
}

STATUS MSOpSetScalarAttrInt64(ResMgrHandle res_mgr, NodeHandle op, const char *attr_name, int64_t value) {
  if (res_mgr == nullptr || op == nullptr || attr_name == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [op] or [attr_name] is nullptr.";
    return RET_NULL_PTR;
  }
  auto prim = GetOpPrim(res_mgr, op);
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Get primitive node failed";
    return RET_NULL_PTR;
  }
  prim->set_attr(attr_name, mindspore::MakeValue(value));
  return RET_OK;
}

STATUS MSOpSetAttrType(ResMgrHandle res_mgr, NodeHandle op, const char *attr_name, TypeId value) {
  if (res_mgr == nullptr || op == nullptr || attr_name == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [op] or [attr_name] is nullptr.";
    return RET_NULL_PTR;
  }
  auto prim = GetOpPrim(res_mgr, op);
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Get primitive node failed";
    return RET_NULL_PTR;
  }
  auto cxx_type = mindspore::TypeId(value);
  prim->set_attr(attr_name, mindspore::TypeIdToType(cxx_type));
  return RET_OK;
}

STATUS MSOpSetAttrTypeArray(ResMgrHandle res_mgr, NodeHandle op, const char *attr_name, TypeId value[],
                            size_t vec_size) {
  if (res_mgr == nullptr || op == nullptr || attr_name == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [op] or [attr_name] is nullptr.";
    return RET_NULL_PTR;
  }
  auto prim = GetOpPrim(res_mgr, op);
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Get primitive node failed";
    return RET_NULL_PTR;
  }
  std::vector<mindspore::ValuePtr> vec_value;
  mindspore::TypeId cxx_type;
  for (size_t i = 0; i < vec_size; i++) {
    cxx_type = mindspore::TypeId(value[i]);
    vec_value.push_back(mindspore::TypeIdToType(cxx_type));
  }
  prim->set_attr(attr_name, mindspore::MakeValue(vec_value));
  return RET_OK;
}

STATUS MSOpSetAttrArray(ResMgrHandle res_mgr, NodeHandle op, const char *attr_name, void *value, size_t vec_size,
                        TypeId dataType) {
  if (res_mgr == nullptr || op == nullptr || attr_name == nullptr || value == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [op] or [attr_name] or [value_vec] is nullptr.";
    return RET_NULL_PTR;
  }
  auto prim = GetOpPrim(res_mgr, op);
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Get primitive node failed";
    return RET_NULL_PTR;
  }

  switch (dataType) {
    case TypeId::kNumberTypeBool: {
      std::vector<bool> vec_value(reinterpret_cast<bool *>(value), reinterpret_cast<bool *>(value) + vec_size);
      prim->set_attr(attr_name, mindspore::MakeValue(vec_value));
      break;
    }
    case TypeId::kNumberTypeInt32: {
      std::vector<int32_t> vec_value(reinterpret_cast<int32_t *>(value), reinterpret_cast<int32_t *>(value) + vec_size);
      prim->set_attr(attr_name, mindspore::MakeValue(vec_value));
      break;
    }
    case TypeId::kNumberTypeInt64: {
      std::vector<int64_t> vec_value(reinterpret_cast<int64_t *>(value), reinterpret_cast<int64_t *>(value) + vec_size);
      prim->set_attr(attr_name, mindspore::MakeValue(vec_value));
      break;
    }
    case TypeId::kNumberTypeFloat32: {
      std::vector<float> vec_value(reinterpret_cast<float *>(value), reinterpret_cast<float *>(value) + vec_size);
      prim->set_attr(attr_name, mindspore::MakeValue(vec_value));
      break;
    }
    default:
      MS_LOG(ERROR) << "Unrecognized datatype w/ TypeId: " << dataType << " , Attribute name: " << attr_name
                    << std::endl;
      return RET_ERROR;
  }
  return RET_OK;
}

STATUS MSOpSetAttrStringArray(ResMgrHandle res_mgr, NodeHandle op, const char *attr_name, const char *value[],
                              size_t vec_size) {
  if (res_mgr == nullptr || op == nullptr || attr_name == nullptr || value == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [op] or [attr_name] or [value_vec] is nullptr.";
    return RET_NULL_PTR;
  }
  auto prim = GetOpPrim(res_mgr, op);
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Get primitive node failed";
    return RET_NULL_PTR;
  }

  std::vector<mindspore::ValuePtr> vec_value;
  for (size_t i = 0; i < vec_size; i++) {
    vec_value.push_back(mindspore::MakeValue(value[i]));
  }
  prim->set_attr(attr_name, std::make_shared<mindspore::ValueList>(vec_value));
  return RET_OK;
}

STATUS MSOpSetAttrString(ResMgrHandle res_mgr, NodeHandle op, const char *attr_name, const char *value) {
  if (res_mgr == nullptr || op == nullptr || attr_name == nullptr || value == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [op] or [attr_name] or [value_vec] is nullptr.";
    return RET_NULL_PTR;
  }
  auto prim = GetOpPrim(res_mgr, op);
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Get primitive node failed";
    return RET_NULL_PTR;
  }
  std::string value_str(value);
  prim->set_attr(attr_name, mindspore::MakeValue(value_str));
  return RET_OK;
}

int64_t MSOpGetScalarAttrInt64(ResMgrHandle res_mgr, NodeHandle op, const char *attr_name, STATUS *error) {
  if (error == nullptr) {
    MS_LOG(ERROR) << "Input status flag [error] is nullptr.";
    return 0;
  }
  if (res_mgr == nullptr || op == nullptr || attr_name == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [op] or [attr_name] is nullptr.";
    *error = RET_NULL_PTR;
    return 0;
  }
  std::string attr_name_str(attr_name);
  try {
    auto prim = GetOpPrim(res_mgr, op);
    MS_EXCEPTION_IF_NULL(prim);
    auto value = prim->GetAttr(attr_name_str);
    auto value_int64 = value->cast<Int64ImmPtr>();
    MS_EXCEPTION_IF_NULL(value_int64);
    auto ret_val = value_int64->value();
    *error = RET_OK;
    return ret_val;
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << " Get Attribute failed. Error info: " << e.what();
    *error = RET_ERROR;
    return 0;
  }
}

STATUS MSOpGetAttrArrayInt64(ResMgrHandle res_mgr, NodeHandle op, const char *attr_name, int64_t values[],
                             size_t value_num) {
  if (res_mgr == nullptr || op == nullptr || attr_name == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [op] or [attr_name] is nullptr.";
    return RET_NULL_PTR;
  }
  std::string attr_name_str(attr_name);
  try {
    auto prim = GetOpPrim(res_mgr, op);
    MS_EXCEPTION_IF_NULL(prim);
    auto value = prim->GetAttr(attr_name_str);
    MS_EXCEPTION_IF_NULL(value);
    auto value_tuple = value->cast<ValueTuplePtr>();
    MS_EXCEPTION_IF_NULL(value_tuple);
    auto value_list = value_tuple->value();
    if (value_list.size() != value_num) {
      MS_LOG(ERROR) << "Invalid input vector length, it should be: " << value_list.size() << ", but got: " << value_num;
      return RET_ERROR;
    }
    for (size_t i = 0; i < value_num; i++) {
      auto val_imm = value_list[i]->cast<Int64ImmPtr>();
      values[i] = val_imm->value();
    }
    return RET_OK;
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Get Attribute failed. Error info: " << e.what();
    return RET_ERROR;
  }
}

AttrHandle MSNewAttrInt64(ResMgrHandle res_mgr, const int64_t v) {
  if (res_mgr == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] is nullptr.";
    return nullptr;
  }
  auto value = std::make_shared<Int64ImmImpl>(v);
  return GetRawPtr(res_mgr, value);
}

AttrHandle MSNewAttrFloat32(ResMgrHandle res_mgr, const float v) {
  if (res_mgr == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] is nullptr.";
    return nullptr;
  }
  auto value = std::make_shared<Float32ImmImpl>(v);
  return GetRawPtr(res_mgr, value);
}

AttrHandle MSNewAttrBool(ResMgrHandle res_mgr, const bool v) {
  if (res_mgr == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] is nullptr.";
    return nullptr;
  }
  auto value = std::make_shared<BoolImmImpl>(v);
  return GetRawPtr(res_mgr, value);
}

AttrHandle MSOpNewAttrs(ResMgrHandle res_mgr, void *value, size_t vec_size, TypeId data_type) {
  if (res_mgr == nullptr || value == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [value_vec] is nullptr.";
    return nullptr;
  }
  mindspore::ValuePtr value_node;

  switch (data_type) {
    case TypeId::kNumberTypeBool: {
      std::vector<bool> vec_value(reinterpret_cast<bool *>(value), reinterpret_cast<bool *>(value) + vec_size);
      value_node = mindspore::MakeValue(vec_value);
      break;
    }
    case TypeId::kNumberTypeInt32: {
      std::vector<int32_t> vec_value(reinterpret_cast<int32_t *>(value), reinterpret_cast<int32_t *>(value) + vec_size);
      value_node = mindspore::MakeValue(vec_value);
      break;
    }
    case TypeId::kNumberTypeInt64: {
      std::vector<int64_t> vec_value(reinterpret_cast<int64_t *>(value), reinterpret_cast<int64_t *>(value) + vec_size);
      value_node = mindspore::MakeValue(vec_value);
      break;
    }
    case TypeId::kNumberTypeFloat32: {
      std::vector<float> vec_value(reinterpret_cast<float *>(value), reinterpret_cast<float *>(value) + vec_size);
      value_node = mindspore::MakeValue(vec_value);
      break;
    }
    default:
      MS_LOG(ERROR) << "Unrecognized datatype w/ TypeId: " << data_type << std::endl;
      return nullptr;
  }
  return GetRawPtr(res_mgr, value_node);
}
