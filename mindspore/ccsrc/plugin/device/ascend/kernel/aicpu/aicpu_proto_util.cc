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

#include "plugin/device/ascend/kernel/aicpu/aicpu_proto_util.h"

#include "proto/tensor.pb.h"
#include "proto/tensor_shape.pb.h"
#include "proto/attr.pb.h"
#include "proto/node_def.pb.h"

#include "ops/array_ops.h"
#include "kernel/oplib/oplib.h"
#include "transform/graph_ir/transform_util.h"
#include "plugin/device/ascend/kernel/aicpu/aicpu_util.h"

namespace mindspore {
namespace kernel {

template <typename T>
void GetListValue(const std::string &attr_name, const mindspore::ValuePtr &value,
                  ::google::protobuf::Map<::std::string, ::mindspore::AttrValue> *node_attr) {
  MS_EXCEPTION_IF_NULL(value);
  MS_EXCEPTION_IF_NULL(node_attr);
  std::vector<T> attr_value;
  auto value_type = value->type();
  MS_EXCEPTION_IF_NULL(value_type);
  auto value_type_str = value_type->ToString();
  if (value_type_str == "string" || value_type_str == "float" || value_type_str == "Int64") {
    auto data = GetValue<T>(value);
    attr_value.push_back(data);
  } else {
    attr_value = GetValue<std::vector<T>>(value);
  }
  mindspore::AttrValue input_shape_attr;
  mindspore::AttrValue_ArrayValue *input_shape_attr_list = input_shape_attr.mutable_array();
  MS_EXCEPTION_IF_NULL(input_shape_attr_list);
  for (const auto shape : attr_value) {
    if constexpr (std::is_same_v<T, std::string>) {
      input_shape_attr_list->add_s(shape);
    } else if constexpr (std::is_same_v<T, float>) {
      input_shape_attr_list->add_f(shape);
    } else if constexpr (std::is_same_v<T, int64_t>) {
      input_shape_attr_list->add_i(shape);
    } else {
      MS_LOG(EXCEPTION) << "Unsupported type " << value_type_str;
    }
  }
  (*node_attr)[attr_name] = input_shape_attr;
}

void ParseAttrValue(const std::string &type, const std::string &attr_name, const mindspore::ValuePtr &value,
                    ::google::protobuf::Map<::std::string, ::mindspore::AttrValue> *node_attr) {
  MS_EXCEPTION_IF_NULL(node_attr);
  MS_EXCEPTION_IF_NULL(value);
  if (type == "int") {
    auto attr_value = value->isa<Int32Imm>() ? GetValue<int>(value) : GetValue<int64_t>(value);
    (*node_attr)[attr_name].set_i(attr_value);
  } else if (type == "str") {
    auto attr_value = GetValue<std::string>(value);
    (*node_attr)[attr_name].set_s(attr_value);
  } else if (type == "bool") {
    auto attr_value = GetValue<bool>(value);
    (*node_attr)[attr_name].set_b(attr_value);
  } else if (type == "float") {
    auto attr_value = GetValue<float>(value);
    (*node_attr)[attr_name].set_f(attr_value);
  } else if (type == "Type") {
    auto attr_value = GetValue<TypePtr>(value);
    auto type_value = static_cast<mindspore::DataType>(kernel::AicpuOpUtil::MsTypeToProtoType(attr_value->type_id()));
    (*node_attr)[attr_name].set_type(type_value);
  } else if (type == "listInt") {
    GetListValue<int64_t>(attr_name, value, node_attr);
  } else if (type == "listFloat") {
    GetListValue<float>(attr_name, value, node_attr);
  } else if (type == "listStr") {
    GetListValue<std::string>(attr_name, value, node_attr);
  } else {
    MS_LOG(EXCEPTION) << "type: " << type << "not support";
  }
}

void SetNodeAttr(const PrimitivePtr &primitive, mindspore::NodeDef *proto) {
  MS_EXCEPTION_IF_NULL(primitive);
  MS_EXCEPTION_IF_NULL(proto);
  std::string op_name = primitive->name();
  if (op_name == kInitDataSetQueue) {
    op_name = kInitData;
  }
  if (op_name == kPrint) {
    return;
  }

  auto op_info_ptr = mindspore::kernel::OpLib::FindOp(op_name, OpImplyType::kImplyAICPU);
  MS_EXCEPTION_IF_NULL(op_info_ptr);
  auto attrs_ptr = op_info_ptr->attrs_ptr();
  MS_EXCEPTION_IF_NULL(primitive);
  ::google::protobuf::Map<::std::string, ::mindspore::AttrValue> *node_attr = proto->mutable_attrs();
  for (const auto &attr_ptr : attrs_ptr) {
    MS_EXCEPTION_IF_NULL(attr_ptr);
    std::string attr_name = attr_ptr->name();
    auto value = primitive->GetAttr(attr_name);
    if (value != nullptr) {
      if (attr_name == kQueueName || attr_name == kSharedName) {
        attr_name = kChannelName;
      } else if (attr_name == kSeed0) {
        attr_name = kSeed;
      } else if (attr_name == kSeed1) {
        attr_name = kSeed2;
      } else if (attr_name == kFormat) {
        attr_name = kDataFormat;
      }
      std::string type = attr_ptr->type();
      ParseAttrValue(type, attr_name, value, node_attr);
    }
  }
}

void SetNodeInputs(const PrimitivePtr &primitive, const std::vector<KernelTensor *> &inputs,
                   mindspore::NodeDef *proto) {
  MS_EXCEPTION_IF_NULL(proto);
  MS_EXCEPTION_IF_NULL(primitive);
  size_t input_num = inputs.size();
  if (input_num == 0) {
    MS_LOG(INFO) << "Node [" << primitive->name() << "] does not have input.";
    return;
  }

  for (size_t input_index = 0; input_index < input_num; input_index++) {
    ::mindspore::Tensor *node_inputs = proto->add_inputs();
    MS_EXCEPTION_IF_NULL(node_inputs);
    const std::vector<int64_t> &input_shape = inputs[input_index]->GetDeviceShapeVector();
    int32_t input_data_type = AicpuOpUtil::MsTypeToProtoType(inputs[input_index]->dtype_id());

    mindspore::TensorShape *tensor_shape = node_inputs->mutable_tensor_shape();
    MS_EXCEPTION_IF_NULL(tensor_shape);
    // todo: delete when tansdata in libcpu_kernel.so is fixed
    if (IsPrimitiveEquals(primitive, prim::kPrimTransData)) {
      auto fmt = transform::TransformUtil::ConvertFormat(inputs[input_index]->GetStringFormat(), input_shape.size());
      tensor_shape->set_data_format(static_cast<::google::protobuf::int32>(fmt));
    }
    for (auto item : input_shape) {
      mindspore::TensorShape_Dim *dim = tensor_shape->add_dim();
      dim->set_size((::google::protobuf::int64)item);
    }
    node_inputs->set_tensor_type(input_data_type);
    node_inputs->set_mem_device("HBM");
    node_inputs->set_data_size(inputs[input_index]->size());
  }
}

void SetNodeOutputs(const PrimitivePtr &primitive, const std::vector<KernelTensor *> &outputs,
                    mindspore::NodeDef *proto) {
  MS_EXCEPTION_IF_NULL(proto);
  MS_EXCEPTION_IF_NULL(primitive);
  size_t output_num = outputs.size();
  if (output_num == 0) {
    MS_LOG(INFO) << "Node [" << primitive->name() << "] does not have output. ";
    return;
  }

  for (size_t output_index = 0; output_index < output_num; output_index++) {
    ::mindspore::Tensor *node_outputs = proto->add_outputs();
    MS_EXCEPTION_IF_NULL(node_outputs);
    mindspore::TensorShape *tensor_shape = node_outputs->mutable_tensor_shape();
    MS_EXCEPTION_IF_NULL(tensor_shape);
    // todo: delete when tansdata in libcpu_kernel.so is fixed
    if (IsPrimitiveEquals(primitive, prim::kPrimTransData)) {
      auto fmt = transform::TransformUtil::ConvertFormat(outputs[output_index]->GetStringFormat(),
                                                         outputs[output_index]->GetDeviceShapeVector().size());
      tensor_shape->set_data_format(static_cast<::google::protobuf::int32>(fmt));
    }
    for (auto item : outputs[output_index]->GetDeviceShapeVector()) {
      mindspore::TensorShape_Dim *dim = tensor_shape->add_dim();
      MS_EXCEPTION_IF_NULL(dim);
      dim->set_size((::google::protobuf::int64)item);
    }
    int32_t output_data_type = AicpuOpUtil::MsTypeToProtoType(outputs[output_index]->dtype_id());

    node_outputs->set_tensor_type(output_data_type);
    node_outputs->set_mem_device("HBM");
    node_outputs->set_data_size(outputs[output_index]->size());
  }
}

void SetNodedefProto(const PrimitivePtr &primitive, const std::vector<KernelTensor *> &inputs,
                     const std::vector<KernelTensor *> &outputs, mindspore::NodeDef *proto) {
  MS_EXCEPTION_IF_NULL(primitive);
  MS_EXCEPTION_IF_NULL(proto);
  std::string op_name = primitive->name();
  if (op_name == kInitDataSetQueue) {
    op_name = kInitData;
  }
  // when op_name is different in mindspore and aicpu
  if (auto iter = kOpNameToAicpuOpNameMap.find(op_name); iter != kOpNameToAicpuOpNameMap.end()) {
    op_name = iter->second;
  }
  // set op name
  proto->set_op(op_name);
  // set inputs tensor
  SetNodeInputs(primitive, inputs, proto);
  // set outputs tensor
  SetNodeOutputs(primitive, outputs, proto);
  // set node attr
  SetNodeAttr(primitive, proto);
}

bool CreateNodeDefBytes(const PrimitivePtr &primitive, const std::vector<KernelTensor *> &inputs,
                        const std::vector<KernelTensor *> &outputs, std::string *proto_str) {
  MS_EXCEPTION_IF_NULL(primitive);
  MS_EXCEPTION_IF_NULL(proto_str);

  mindspore::NodeDef proto;
  SetNodedefProto(primitive, inputs, outputs, &proto);
  if (!proto.SerializeToString(proto_str)) {
    MS_LOG(ERROR) << "Serialize nodeDef to string failed.";
    return false;
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
