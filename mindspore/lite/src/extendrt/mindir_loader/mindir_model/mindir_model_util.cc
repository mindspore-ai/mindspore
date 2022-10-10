/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include <vector>
#include <memory>
#include <string>

#include "extendrt/mindir_loader/mindir_model/mindir_model_util.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "include/errorcode.h"
#include "nnacl/op_base.h"
#include "src/common/common.h"

namespace mindspore::infer::mindir {
static mindspore::HashMap<int, TypeId> kDefaultValueSwitchMap{
  {mind_ir::TensorProto_DataType_BOOL, kNumberTypeBool},
  {mind_ir::TensorProto_DataType_INT8, kNumberTypeInt8},
  {mind_ir::TensorProto_DataType_INT16, kNumberTypeInt16},
  {mind_ir::TensorProto_DataType_INT32, kNumberTypeInt32},
  {mind_ir::TensorProto_DataType_INT64, kNumberTypeInt64},
  {mind_ir::TensorProto_DataType_UINT8, kNumberTypeUInt8},
  {mind_ir::TensorProto_DataType_UINT16, kNumberTypeUInt16},
  {mind_ir::TensorProto_DataType_UINT32, kNumberTypeUInt32},
  {mind_ir::TensorProto_DataType_UINT64, kNumberTypeUInt64},
  {mind_ir::TensorProto_DataType_FLOAT16, kNumberTypeFloat16},
  {mind_ir::TensorProto_DataType_FLOAT, kNumberTypeFloat32},
  {mind_ir::TensorProto_DataType_FLOAT64, kNumberTypeFloat64},
  {mind_ir::TensorProto_DataType_DOUBLE, kNumberTypeFloat64},
  {mind_ir::TensorProto_DataType_STRING, kObjectTypeString},
  {mind_ir::TensorProto_DataType_COMPLEX64, kNumberTypeComplex64},
  {mind_ir::TensorProto_DataType_COMPLEX128, kNumberTypeComplex128}};

mindspore::ValuePtr MindirModelUtil::MakeValueFromAttribute(const mind_ir::AttributeProto &attr_proto) {
  switch (attr_proto.type()) {
    case mind_ir::AttributeProto_AttributeType_TENSORS: {
      // embed tensor attribute
      return MindirModelUtil::MakeValueFromTensorOrTypeAttribute(attr_proto);
    }
    case mind_ir::AttributeProto_AttributeType_TUPLE:
    case mind_ir::AttributeProto_AttributeType_LIST: {
      // list attribute
      return MindirModelUtil::MakeValueFromListAttribute(attr_proto);
    }
    default: {
      // base scalar attribute
      return MindirModelUtil::MakeValueFromScalarAttribute(attr_proto);
    }
  }
}

mindspore::ValuePtr MindirModelUtil::MakeValueFromTensorOrTypeAttribute(const mind_ir::AttributeProto &attr_proto) {
  auto tensor_proto = attr_proto.tensors(0);
  if (tensor_proto.has_raw_data()) {
    // For real tensor
    return MindirModelUtil::MakeValueFromTensorAttribute(tensor_proto);
  } else {
    // for data type
    const int attr_tensor_type = tensor_proto.data_type();
    auto iter = kDefaultValueSwitchMap.find(attr_tensor_type);
    MS_CHECK_TRUE_MSG(iter == kDefaultValueSwitchMap.end(), nullptr,
                      "MindirModelUtil: Generate value ptr failed, cannot find attr tensor type " << attr_tensor_type);
    return TypeIdToType(iter->second);
  }
}

mindspore::ValuePtr MindirModelUtil::MakeValueFromTensorAttribute(const mind_ir::TensorProto &tensor_proto,
                                                                  bool need_load_data) {
  ShapeVector shape;
  auto attr_tensor_type = tensor_proto.data_type();
  for (int i = 0; i < tensor_proto.dims_size(); i++) {
    shape.push_back(tensor_proto.dims(i));
  }
  tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(kDefaultValueSwitchMap[attr_tensor_type], shape);

  MS_EXCEPTION_IF_NULL(tensor);
  const std::string &tensor_buf = tensor_proto.raw_data();
  if (tensor_proto.has_raw_data()) {
    auto *tensor_data_buf = reinterpret_cast<uint8_t *>(tensor->data_c());
    auto ret = memcpy_s(tensor_data_buf, tensor->data().nbytes(), tensor_buf.data(), tensor_buf.size());
    MS_CHECK_TRUE_MSG(
      ret != mindspore::lite::RET_OK, nullptr,
      "MindirModelUtil: Generate tensor ptr from tensor proto failed, failed to get tensor from tensor proto.");
  } else {
    MS_CHECK_TRUE_MSG(
      need_load_data, nullptr,
      "MindirModelUtil: Generate tensor ptr from tensor proto failed, failed to get tensor from tensor proto.");
  }
  return tensor;
}

mindspore::ValuePtr MindirModelUtil::MakeValueFromListAttribute(const mind_ir::AttributeProto &attr_proto) {
  std::vector<mindspore::ValuePtr> vec;
  for (int i = 0; i < attr_proto.values_size(); i++) {
    mind_ir::AttributeProto elem_attr_proto = attr_proto.values(i);
    mindspore::ValuePtr value_ptr = MindirModelUtil::MakeValueFromAttribute(elem_attr_proto);
    vec.emplace_back(value_ptr);
  }
  auto type = attr_proto.type();
  mindspore::ValuePtr value_sequence;
  switch (type) {
    case mind_ir::AttributeProto_AttributeType_TUPLE: {
      return std::make_shared<mindspore::ValueTuple>(vec);
    }
    case mind_ir::AttributeProto_AttributeType_LIST: {
      return std::make_shared<mindspore::ValueList>(vec);
    }
    default: {
      MS_LOG(ERROR)
        << "MindirModelUtil: Obtain value in sequence form failed, the attribute type should be tuple or list";
      return nullptr;
    }
  }
}

mindspore::ValuePtr MindirModelUtil::MakeValueFromScalarAttribute(const mind_ir::AttributeProto &attr_proto) {
  auto attr_proto_type = static_cast<int>(attr_proto.type());
  switch (attr_proto_type) {
    case mind_ir::AttributeProto_AttributeType_STRING: {
      auto value = static_cast<std::string>(attr_proto.s());
      return MakeValue<std::string>(value);
    }
    case mind_ir::AttributeProto_AttributeType_INT8: {
      auto value = static_cast<int8_t>(attr_proto.i());
      return MakeValue<int8_t>(value);
    }
    case mind_ir::AttributeProto_AttributeType_INT16: {
      auto value = static_cast<int16_t>(attr_proto.i());
      return MakeValue<int16_t>(value);
    }
    case mind_ir::AttributeProto_AttributeType_INT32: {
      auto value = static_cast<int32_t>(attr_proto.i());
      return MakeValue<int32_t>(value);
    }
    case mind_ir::AttributeProto_AttributeType_INT64: {
      auto value = static_cast<int64_t>(attr_proto.i());
      return MakeValue<int64_t>(value);
    }
    case mind_ir::AttributeProto_AttributeType_UINT8: {
      auto value = static_cast<uint8_t>(attr_proto.i());
      return MakeValue<uint8_t>(value);
    }
    case mind_ir::AttributeProto_AttributeType_UINT16: {
      auto value = static_cast<uint16_t>(attr_proto.i());
      return MakeValue<uint16_t>(value);
    }
    case mind_ir::AttributeProto_AttributeType_UINT32: {
      auto value = static_cast<uint32_t>(attr_proto.i());
      return MakeValue<uint32_t>(value);
    }
    case mind_ir::AttributeProto_AttributeType_UINT64: {
      auto value = static_cast<uint64_t>(attr_proto.i());
      return MakeValue<uint64_t>(value);
    }
    case mind_ir::AttributeProto_AttributeType_FLOAT: {
      auto value = static_cast<float>(attr_proto.f());
      return MakeValue<float>(value);
    }
    case mind_ir::AttributeProto_AttributeType_DOUBLE: {
      auto value = static_cast<double>(attr_proto.d());
      return MakeValue<double>(value);
    }
    case mind_ir::AttributeProto_AttributeType_BOOL: {
      auto value = static_cast<int32_t>(attr_proto.i());
      return MakeValue<bool>(value);
    }
    default: {
      MS_LOG(ERROR) << "MindirModelUtil: Obtain cnode attr in single scalar form failed, attr type " << attr_proto_type
                    << " is xinot supported ";
      return nullptr;
    }
  }
}

mindspore::TypeId MindirModelUtil::ProtoTypeToTypeId(int32_t proto_type) {
  auto it = kDefaultValueSwitchMap.find(proto_type);
  if (it == kDefaultValueSwitchMap.end()) {
    return kTypeUnknown;
  }
  return it->second;
}

bool MindirModelUtil::NeedRuntimeConvert(const void *model_data, size_t data_size,
                                         const std::shared_ptr<mindspore::Context> &context) {
  auto device_list = context->MutableDeviceInfo();
  for (const auto &device_info : device_list) {
    if (device_info == nullptr) {
      continue;
    }
    if (device_info->GetDeviceType() == DeviceType::kAscend && device_info->GetProvider() == "ge") {
      return false;
    }
  }
  bool need_runtime_convert = true;
  mind_ir::ModelProto model_proto;
  std::string str(static_cast<const char *>(model_data), data_size);
  if (model_proto.ParseFromString(str)) {
    mind_ir::GraphProto *graph_proto = model_proto.mutable_graph();
    if (graph_proto != nullptr) {
      for (int i = 0; i < graph_proto->attribute_size(); ++i) {
        const mind_ir::AttributeProto &attr_proto = graph_proto->attribute(i);
        if (attr_proto.has_name() && attr_proto.name() == lite::kIsOptimized) {
          const int attr_type = static_cast<int>(attr_proto.type());
          if (attr_type != mind_ir::AttributeProto_AttributeType_BOOL) {
            MS_LOG(ERROR) << "The type of attr optimized value must be bool.";
            return true;
          }
          if (static_cast<bool>(attr_proto.i())) {
            need_runtime_convert = false;
            MS_LOG(DEBUG) << "No need to online infer.";
          }
          break;
        }
      }
    }
  } else {
    MS_LOG(WARNING) << "Not mindir model";
    need_runtime_convert = false;
  }
  return need_runtime_convert;
}
}  // namespace mindspore::infer::mindir
