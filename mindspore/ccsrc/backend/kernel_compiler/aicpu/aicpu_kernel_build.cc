/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "backend/kernel_compiler/aicpu/aicpu_kernel_build.h"
#include <google/protobuf/text_format.h>
#include <utility>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include "runtime/device/kernel_runtime.h"
#include "backend/kernel_compiler/aicpu/aicpu_kernel_mod.h"
#include "backend/kernel_compiler/akg/akg_kernel_build.h"
#include "proto/tensor.pb.h"
#include "proto/tensor_shape.pb.h"
#include "proto/attr.pb.h"
#include "proto/node_def.pb.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/kernel_compiler/aicpu/aicpu_util.h"
#include "backend/session/kernel_graph.h"
#include "backend/kernel_compiler/common_utils.h"

namespace mindspore {
namespace kernel {
using FNodeAttrHandle = std::function<void(const std::shared_ptr<AnfNode> &anf_node, mindspore::NodeDef *proto)>;

bool SetIOIputSize(const std::shared_ptr<AnfNode> &anf_node, const size_t &input_num,
                   std::vector<size_t> *input_size_list) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(input_size_list);
  for (size_t i = 0; i < input_num; i++) {
    std::vector<size_t> shape_i = AnfAlgo::GetInputDeviceShape(anf_node, i);
    if (AnfAlgo::GetInputDeviceDataType(anf_node, i) == kObjectTypeString) {
      if (!anf_node->isa<CNode>()) {
        MS_LOG(EXCEPTION) << "anf_node is not CNode.";
      }
      auto cnode = anf_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      if (cnode->inputs().size() < (i + 1)) {
        MS_LOG(ERROR) << "cnode inputs size " << cnode->inputs().size() << " is smaller than " << i + 1;
        return false;
      }
      auto input_node = cnode->inputs()[i + 1];
      MS_EXCEPTION_IF_NULL(input_node);
      if (input_node->isa<ValueNode>()) {
        auto value_ptr = GetValueNode(input_node);
        auto value = GetValue<std::string>(value_ptr);
        input_size_list->push_back(value.size());
      }
    } else {
      auto type_ptr = TypeIdToType(AnfAlgo::GetInputDeviceDataType(anf_node, i));
      MS_EXCEPTION_IF_NULL(type_ptr);
      int64_t size_i = 1;
      for (size_t j = 0; j < shape_i.size(); j++) {
        size_i = LongMulWithOverflowCheck(size_i, static_cast<int>(shape_i[j]));
      }
      size_t type_byte = GetTypeByte(type_ptr);
      if (type_byte == 0) {
        return false;
      }
      size_i = LongMulWithOverflowCheck(size_i, SizeToInt(type_byte));
      input_size_list->push_back(LongToSize(size_i));
    }
  }
  return true;
}

bool SetIOSize(const std::shared_ptr<AnfNode> &anf_node, const std::shared_ptr<AicpuOpKernelMod> &kernel_mod_ptr) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
  std::vector<size_t> input_size_list;
  std::vector<size_t> output_size_list;
  size_t input_num = AnfAlgo::GetInputTensorNum(anf_node);
  size_t output_num = AnfAlgo::GetOutputTensorNum(anf_node);

  if (!SetIOIputSize(anf_node, input_num, &input_size_list)) {
    return false;
  }
  kernel_mod_ptr->SetInputSizeList(input_size_list);

  for (size_t i = 0; i < output_num; i++) {
    std::vector<size_t> shape_i = AnfAlgo::GetOutputDeviceShape(anf_node, i);
    TypePtr type_ptr = TypeIdToType(AnfAlgo::GetOutputDeviceDataType(anf_node, i));
    MS_EXCEPTION_IF_NULL(type_ptr);
    int64_t size_i = 1;
    for (size_t j = 0; j < shape_i.size(); j++) {
      size_i = LongMulWithOverflowCheck(size_i, static_cast<int>(shape_i[j]));
    }
    size_t type_byte = GetTypeByte(type_ptr);
    if (type_byte == 0) {
      return false;
    }
    size_i = LongMulWithOverflowCheck(size_i, SizeToInt(type_byte));
    output_size_list.push_back(LongToSize(size_i));
  }
  kernel_mod_ptr->SetOutputSizeList(output_size_list);
  return true;
}

void ParseAttrValue(const std::string &type, const std::string &attr_name, const mindspore::ValuePtr &value,
                    ::google::protobuf::Map<::std::string, ::mindspore::AttrValue> *node_attr) {
  MS_EXCEPTION_IF_NULL(node_attr);
  MS_EXCEPTION_IF_NULL(value);
  if (type == "int") {
    auto attr_value = GetValue<int>(value);
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
  } else if (type == "listInt") {
    std::vector<int> attr_value;
    auto value_type = value->type();
    MS_EXCEPTION_IF_NULL(value_type);
    auto value_type_str = value_type->ToString();
    if (value_type_str == "Int32") {
      int data = GetValue<int>(value);
      attr_value.push_back(data);
    } else {
      attr_value = GetValue<std::vector<int>>(value);
    }
    mindspore::AttrValue input_shape_attr;
    mindspore::AttrValue_ArrayValue *input_shape_attr_list = input_shape_attr.mutable_array();
    MS_EXCEPTION_IF_NULL(input_shape_attr_list);
    for (const auto shape : attr_value) {
      input_shape_attr_list->add_i(shape);
    }
    (*node_attr)[attr_name] = input_shape_attr;
  } else {
    MS_LOG(EXCEPTION) << "type: " << type << "not support";
  }
}

void SetNodeAttr(const std::shared_ptr<AnfNode> &anf_node, mindspore::NodeDef *proto) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(proto);
  std::string op_name = AnfAlgo::GetCNodeName(anf_node);
  if (op_name == kInitDataSetQueue) {
    op_name = kInitData;
  }
  if (op_name == kPrint) {
    return;
  }

  auto op_info_ptr = mindspore::kernel::OpLib::FindOp(op_name, OpImplyType::kAICPU);
  MS_EXCEPTION_IF_NULL(op_info_ptr);
  auto attrs_ptr = op_info_ptr->attrs_ptr();
  auto primitive = AnfAlgo::GetCNodePrimitive(anf_node);
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
      }
      std::string type = attr_ptr->type();
      ParseAttrValue(type, attr_name, value, node_attr);
    }
  }
  MS_LOG(INFO) << "Set node attr end!";
}

void SetNodeInputs(const std::shared_ptr<AnfNode> &anf_node, mindspore::NodeDef *proto) {
  MS_EXCEPTION_IF_NULL(proto);
  MS_EXCEPTION_IF_NULL(anf_node);
  size_t input_num = AnfAlgo::GetInputTensorNum(anf_node);
  if (input_num == 0) {
    MS_LOG(INFO) << "Node [" << AnfAlgo::GetCNodeName(anf_node) << "] does not have input.";
    return;
  }

  for (size_t input_index = 0; input_index < input_num; input_index++) {
    ::mindspore::Tensor *node_inputs = proto->add_inputs();
    MS_EXCEPTION_IF_NULL(node_inputs);
    TypeId input_type = AnfAlgo::GetInputDeviceDataType(anf_node, input_index);
    std::vector<size_t> input_shape;
    int32_t input_data_type;
    if (input_type == kObjectTypeString) {
      auto cnode = anf_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      auto input_node = cnode->inputs()[input_index + 1];
      auto value_ptr = GetValueNode(input_node);
      auto value = GetValue<std::string>(value_ptr);
      input_shape.push_back(1);
      input_shape.push_back(value.size());
      input_data_type = AicpuOpUtil::MsTypeToProtoType(kTypeUnknown);
    } else {
      input_shape = AnfAlgo::GetInputDeviceShape(anf_node, input_index);
      input_data_type = AicpuOpUtil::MsTypeToProtoType(input_type);
    }

    mindspore::TensorShape *tensorShape = node_inputs->mutable_tensor_shape();
    for (auto item : input_shape) {
      mindspore::TensorShape_Dim *dim = tensorShape->add_dim();
      dim->set_size((::google::protobuf::int64)item);
    }
    node_inputs->set_tensor_type((mindspore::DataType)input_data_type);
    node_inputs->set_mem_device("HBM");
  }
}

void SetNodeOutputs(const std::shared_ptr<AnfNode> &anf_node, mindspore::NodeDef *proto) {
  MS_EXCEPTION_IF_NULL(proto);
  MS_EXCEPTION_IF_NULL(anf_node);
  size_t output_num = AnfAlgo::GetOutputTensorNum(anf_node);
  if (output_num == 0) {
    MS_LOG(INFO) << "Node [" << AnfAlgo::GetCNodeName(anf_node) << "] does not have output. ";
    return;
  }

  for (size_t output_index = 0; output_index < output_num; output_index++) {
    ::mindspore::Tensor *node_outputs = proto->add_outputs();
    MS_EXCEPTION_IF_NULL(node_outputs);
    std::vector<size_t> output_shape = AnfAlgo::GetOutputDeviceShape(anf_node, output_index);
    mindspore::TensorShape *tensorShape = node_outputs->mutable_tensor_shape();
    MS_EXCEPTION_IF_NULL(tensorShape);
    for (auto item : output_shape) {
      mindspore::TensorShape_Dim *dim = tensorShape->add_dim();
      MS_EXCEPTION_IF_NULL(dim);
      dim->set_size((::google::protobuf::int64)item);
    }
    TypeId output_type = AnfAlgo::GetOutputDeviceDataType(anf_node, output_index);
    int32_t output_data_type = AicpuOpUtil::MsTypeToProtoType(output_type);
    node_outputs->set_tensor_type((mindspore::DataType)output_data_type);
    node_outputs->set_mem_device("HBM");
  }
}

void SetNodedefProto(const std::shared_ptr<AnfNode> &anf_node, mindspore::NodeDef *proto) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(proto);
  MS_LOG(INFO) << "SetNodedefProto entry";
  std::string op_name = AnfAlgo::GetCNodeName(anf_node);
  if (op_name == kInitDataSetQueue) {
    op_name = kInitData;
  }
  // set op name
  proto->set_op(op_name);
  // set inputs tensor
  SetNodeInputs(anf_node, proto);
  // set outputs tensor
  SetNodeOutputs(anf_node, proto);
  // set node attr
  SetNodeAttr(anf_node, proto);
  MS_LOG(INFO) << "SetNodedefProto end!";
}

bool CreateNodeDefBytes(const std::shared_ptr<AnfNode> &anf_node,
                        const std::shared_ptr<AicpuOpKernelMod> &kernel_mod_ptr) {
  MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_LOG(INFO) << "CreateNodeDefBytes entry";

  mindspore::NodeDef proto;
  SetNodedefProto(anf_node, &proto);
  std::string nodeDefStr;
  if (!proto.SerializeToString(&nodeDefStr)) {
    MS_LOG(ERROR) << "Serialize nodeDef to string failed.";
    return false;
  }
  kernel_mod_ptr->SetNodeDef(nodeDefStr);
  MS_LOG(INFO) << "CreateNodeDefBytes end!";
  return true;
}

KernelModPtr AicpuOpBuild(const std::shared_ptr<AnfNode> &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  std::string op_name = AnfAlgo::GetCNodeName(anf_node);
  if (op_name == kInitDataSetQueue) {
    op_name = kInitData;
  }
  auto kernel_mod_ptr = std::make_shared<AicpuOpKernelMod>();
  MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
  kernel_mod_ptr->SetAnfNode(anf_node);
  kernel_mod_ptr->SetNodeName(op_name);
  if (!CreateNodeDefBytes(anf_node, kernel_mod_ptr)) {
    MS_LOG(EXCEPTION) << "Create nodeDefBytes faild!";
  }
  if (!SetIOSize(anf_node, kernel_mod_ptr)) {
    MS_LOG(EXCEPTION) << "Set input output size list failed.";
  }
  return kernel_mod_ptr;
}
}  // namespace kernel
}  // namespace mindspore
