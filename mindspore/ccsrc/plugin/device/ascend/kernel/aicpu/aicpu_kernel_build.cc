/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/kernel/aicpu/aicpu_kernel_build.h"

#include <utility>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <climits>

#include "include/common/utils/utils.h"
#include "plugin/device/ascend/kernel/aicpu/aicpu_kernel_mod.h"
#include "proto/tensor.pb.h"
#include "proto/tensor_shape.pb.h"
#include "proto/attr.pb.h"
#include "proto/node_def.pb.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/device/ascend/kernel/aicpu/aicpu_util.h"
#include "plugin/device/ascend/kernel/aicpu/aicpu_kernel_load.h"
#include "kernel/common_utils.h"
#include "kernel/oplib/oplib.h"
#include "cce/fwk_adpt_struct.h"
#include "external/graph/types.h"
#include "transform/graph_ir/transform_util.h"
#include "cce/aicpu_engine_struct.h"

namespace mindspore {
namespace kernel {
namespace {
static uint64_t g_aicpu_kernel_id = 0;
static uint64_t g_aicpu_session_id = 0;
}  // namespace
using FNodeAttrHandle = std::function<void(const std::shared_ptr<AnfNode> &anf_node, mindspore::NodeDef *proto)>;

bool SetIOIputSize(const std::shared_ptr<AnfNode> &anf_node, const size_t &input_num,
                   std::vector<size_t> *input_size_list) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(input_size_list);
  for (size_t i = 0; i < input_num; i++) {
    auto shape_i = AnfAlgo::GetInputDeviceShape(anf_node, i);
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
      int64_t size_i = 1;
      if (!GetShapeSize(shape_i, type_ptr, &size_i)) {
        return false;
      }
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
  size_t input_num = common::AnfAlgo::GetInputTensorNum(anf_node);
  size_t output_num = AnfAlgo::GetOutputTensorNum(anf_node);

  if (!SetIOIputSize(anf_node, input_num, &input_size_list)) {
    return false;
  }
  kernel_mod_ptr->SetInputSizeList(input_size_list);
  if (output_num == 1 && HasAbstractMonad(anf_node)) {
    output_num = 0;
  }
  for (size_t i = 0; i < output_num; i++) {
    auto shape_i = AnfAlgo::GetOutputDeviceShape(anf_node, i);
    TypePtr type_ptr = TypeIdToType(AnfAlgo::GetOutputDeviceDataType(anf_node, i));
    int64_t size_i = 1;
    if (!GetShapeSize(shape_i, type_ptr, &size_i)) {
      return false;
    }
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
    std::vector<int64_t> attr_value;
    auto value_type = value->type();
    MS_EXCEPTION_IF_NULL(value_type);
    auto value_type_str = value_type->ToString();
    if (value_type_str == "Int64") {
      auto data = GetValue<int64_t>(value);
      attr_value.push_back(data);
    } else {
      attr_value = GetValue<std::vector<int64_t>>(value);
    }
    mindspore::AttrValue input_shape_attr;
    mindspore::AttrValue_ArrayValue *input_shape_attr_list = input_shape_attr.mutable_array();
    MS_EXCEPTION_IF_NULL(input_shape_attr_list);
    for (const auto shape : attr_value) {
      input_shape_attr_list->add_i(shape);
    }
    (*node_attr)[attr_name] = input_shape_attr;
  } else if (type == "listFloat") {
    std::vector<float> attr_value;
    auto value_type = value->type();
    MS_EXCEPTION_IF_NULL(value_type);
    auto value_type_str = value_type->ToString();
    if (value_type_str == "float") {
      auto data = GetValue<float>(value);
      attr_value.push_back(data);
    } else {
      attr_value = GetValue<std::vector<float>>(value);
    }
    mindspore::AttrValue input_shape_attr;
    mindspore::AttrValue_ArrayValue *input_shape_attr_list = input_shape_attr.mutable_array();
    MS_EXCEPTION_IF_NULL(input_shape_attr_list);
    for (const auto shape : attr_value) {
      input_shape_attr_list->add_f(shape);
    }
    (*node_attr)[attr_name] = input_shape_attr;
  } else if (type == "listStr") {
    std::vector<std::string> attr_value;
    auto value_type = value->type();
    MS_EXCEPTION_IF_NULL(value_type);
    auto value_type_str = value_type->ToString();
    if (value_type_str == "string") {
      auto data = GetValue<std::string>(value);
      attr_value.push_back(data);
    } else {
      attr_value = GetValue<std::vector<std::string>>(value);
    }
    mindspore::AttrValue input_shape_attr;
    mindspore::AttrValue_ArrayValue *input_shape_attr_list = input_shape_attr.mutable_array();
    MS_EXCEPTION_IF_NULL(input_shape_attr_list);
    for (const auto shape : attr_value) {
      input_shape_attr_list->add_s(shape);
    }
    (*node_attr)[attr_name] = input_shape_attr;
  } else {
    MS_LOG(EXCEPTION) << "type: " << type << "not support";
  }
}

void SetNodeAttr(const std::shared_ptr<AnfNode> &anf_node, mindspore::NodeDef *proto) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(proto);
  std::string op_name = common::AnfAlgo::GetCNodeName(anf_node);
  if (op_name == kInitDataSetQueue) {
    op_name = kInitData;
  }
  if (op_name == kPrint) {
    return;
  }

  auto op_info_ptr = mindspore::kernel::OpLib::FindOp(op_name, OpImplyType::kImplyAICPU);
  MS_EXCEPTION_IF_NULL(op_info_ptr);
  auto attrs_ptr = op_info_ptr->attrs_ptr();
  auto primitive = common::AnfAlgo::GetCNodePrimitive(anf_node);
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

void SetNodeInputs(const std::shared_ptr<AnfNode> &anf_node, mindspore::NodeDef *proto) {
  MS_EXCEPTION_IF_NULL(proto);
  MS_EXCEPTION_IF_NULL(anf_node);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(anf_node);
  if (input_num == 0) {
    MS_LOG(INFO) << "Node [" << common::AnfAlgo::GetCNodeName(anf_node) << "] does not have input.";
    return;
  }

  std::vector<size_t> input_size_list;
  if (!SetIOIputSize(anf_node, input_num, &input_size_list)) {
    MS_LOG(ERROR) << "Node [" << common::AnfAlgo::GetCNodeName(anf_node) << "] get input size list failed.";
    return;
  }

  for (size_t input_index = 0; input_index < input_num; input_index++) {
    ::mindspore::Tensor *node_inputs = proto->add_inputs();
    MS_EXCEPTION_IF_NULL(node_inputs);
    TypeId input_type = AnfAlgo::GetInputDeviceDataType(anf_node, input_index);
    std::vector<int64_t> input_shape;
    int32_t input_data_type;
    if (input_type == kObjectTypeString) {
      auto cnode = anf_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      auto input_node = cnode->inputs()[input_index + 1];
      auto value_ptr = GetValueNode(input_node);
      MS_EXCEPTION_IF_NULL(value_ptr);
      auto value = GetValue<std::string>(value_ptr);
      input_shape.push_back(1);
      input_shape.push_back(static_cast<int64_t>(value.size()));
      input_data_type = AicpuOpUtil::MsTypeToProtoType(kObjectTypeString);
    } else {
      input_shape = AnfAlgo::GetInputDeviceShape(anf_node, input_index);
      input_data_type = AicpuOpUtil::MsTypeToProtoType(input_type);
    }

    mindspore::TensorShape *tensor_shape = node_inputs->mutable_tensor_shape();
    MS_EXCEPTION_IF_NULL(tensor_shape);
    // todo: delete when tansdata in libcpu_kernel.so is fixed
    if (IsPrimitiveCNode(anf_node, prim::kPrimTransData)) {
      auto format = AnfAlgo::GetInputFormat(anf_node, input_index);
      tensor_shape->set_data_format(
        static_cast<::google::protobuf::int32>(transform::TransformUtil::ConvertFormat(format, input_shape.size())));
    }
    for (auto item : input_shape) {
      mindspore::TensorShape_Dim *dim = tensor_shape->add_dim();
      dim->set_size((::google::protobuf::int64)item);
    }
    node_inputs->set_tensor_type(input_data_type);
    node_inputs->set_mem_device("HBM");
    node_inputs->set_data_size(input_size_list[input_index]);
  }
}

void SetNodeOutputs(const std::shared_ptr<AnfNode> &anf_node, mindspore::NodeDef *proto) {
  MS_EXCEPTION_IF_NULL(proto);
  MS_EXCEPTION_IF_NULL(anf_node);
  size_t output_num = AnfAlgo::GetOutputTensorNum(anf_node);
  if (output_num == 1 && HasAbstractMonad(anf_node)) {
    output_num = 0;
  }
  if (output_num == 0) {
    MS_LOG(INFO) << "Node [" << common::AnfAlgo::GetCNodeName(anf_node) << "] does not have output. ";
    return;
  }

  for (size_t output_index = 0; output_index < output_num; output_index++) {
    ::mindspore::Tensor *node_outputs = proto->add_outputs();
    MS_EXCEPTION_IF_NULL(node_outputs);
    auto output_shape = AnfAlgo::GetOutputDeviceShape(anf_node, output_index);
    mindspore::TensorShape *tensor_shape = node_outputs->mutable_tensor_shape();
    MS_EXCEPTION_IF_NULL(tensor_shape);
    // todo: delete when tansdata in libcpu_kernel.so is fixed
    if (IsPrimitiveCNode(anf_node, prim::kPrimTransData)) {
      auto format = AnfAlgo::GetOutputFormat(anf_node, output_index);
      tensor_shape->set_data_format(
        static_cast<::google::protobuf::int32>(transform::TransformUtil::ConvertFormat(format, output_shape.size())));
    }
    for (auto item : output_shape) {
      mindspore::TensorShape_Dim *dim = tensor_shape->add_dim();
      MS_EXCEPTION_IF_NULL(dim);
      dim->set_size((::google::protobuf::int64)item);
    }
    TypeId output_type = AnfAlgo::GetOutputDeviceDataType(anf_node, output_index);
    int32_t output_data_type = AicpuOpUtil::MsTypeToProtoType(output_type);

    int64_t data_size = 1;
    if (!GetShapeSize(output_shape, TypeIdToType(output_type), &data_size)) {
      MS_LOG(ERROR) << "Node [" << common::AnfAlgo::GetCNodeName(anf_node) << "] get output size failed for output "
                    << output_index;
      return;
    }

    node_outputs->set_tensor_type(output_data_type);
    node_outputs->set_mem_device("HBM");
    node_outputs->set_data_size(LongToSize(data_size));
  }
}

void SetNodedefProto(const std::shared_ptr<AnfNode> &anf_node, mindspore::NodeDef *proto) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(proto);
  std::string op_name = common::AnfAlgo::GetCNodeName(anf_node);
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
  SetNodeInputs(anf_node, proto);
  // set outputs tensor
  SetNodeOutputs(anf_node, proto);
  // set node attr
  SetNodeAttr(anf_node, proto);
}

bool CreateNodeDefBytes(const std::shared_ptr<AnfNode> &anf_node,
                        const std::shared_ptr<AicpuOpKernelMod> &kernel_mod_ptr) {
  MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
  MS_EXCEPTION_IF_NULL(anf_node);

  mindspore::NodeDef proto;
  SetNodedefProto(anf_node, &proto);
  std::string nodeDefStr;
  if (!proto.SerializeToString(&nodeDefStr)) {
    MS_LOG(ERROR) << "Serialize nodeDef to string failed.";
    return false;
  }
  kernel_mod_ptr->SetNodeDef(nodeDefStr);
  return true;
}

uint64_t SetExtInfoShapeType(char *ext_info_buf, uint64_t ext_info_offset, ::ge::UnknowShapeOpType type) {
  // deal1: unknown shape type
  auto *info = reinterpret_cast<aicpu::FWKAdapter::ExtInfo *>(ext_info_buf + ext_info_offset);
  info->infoType = static_cast<int32_t>(aicpu::FWKAdapter::FWK_ADPT_EXT_SHAPE_TYPE);
  info->infoLen = sizeof(int32_t);
  ext_info_offset += aicpu::FWKAdapter::kExtInfoHeadSize;
  auto *shape_type = reinterpret_cast<int32_t *>(ext_info_buf + ext_info_offset);
  *shape_type = static_cast<int32_t>(type);
  ext_info_offset += info->infoLen;
  return ext_info_offset;
}

uint64_t SetExtInfoInputShapeType(char *ext_info_buf, uint64_t ext_info_offset,
                                  const std::shared_ptr<AnfNode> &anf_node, size_t input_num) {
  // deal2:input ShapeAndType
  auto *info = reinterpret_cast<aicpu::FWKAdapter::ExtInfo *>(ext_info_buf + ext_info_offset);
  info->infoType = static_cast<int32_t>(aicpu::FWKAdapter::FWK_ADPT_EXT_INPUT_SHAPE);
  info->infoLen = SizeToUint(input_num * sizeof(aicpu::FWKAdapter::ShapeAndType));
  ext_info_offset += aicpu::FWKAdapter::kExtInfoHeadSize;

  auto *inputs = reinterpret_cast<aicpu::FWKAdapter::ShapeAndType *>(ext_info_buf + ext_info_offset);
  for (size_t input_index = 0; input_index < input_num; input_index++) {
    TypeId input_type = AnfAlgo::GetInputDeviceDataType(anf_node, input_index);
    std::vector<int64_t> input_shape;
    int32_t input_data_type;
    if (input_type == kObjectTypeString) {
      auto cnode = anf_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      auto input_node = cnode->inputs()[input_index + 1];
      auto value_ptr = GetValueNode(input_node);
      auto value = GetValue<std::string>(value_ptr);
      input_shape.push_back(1);
      input_shape.push_back(static_cast<int64_t>(value.size()));
      input_data_type = AicpuOpUtil::MsTypeToProtoType(kObjectTypeString);
    } else {
      input_shape = AnfAlgo::GetInputDeviceShape(anf_node, input_index);
      input_data_type = AicpuOpUtil::MsTypeToProtoType(input_type);
    }
    inputs[input_index].type = input_data_type;

    size_t input_shape_index = 0;
    for (; input_shape_index < input_shape.size(); input_shape_index++) {
      inputs[input_index].dims[input_shape_index] = input_shape[input_shape_index];
    }
    if (input_shape.size() < aicpu::FWKAdapter::kMaxShapeDims) {
      inputs[input_index].dims[input_shape_index] = LLONG_MIN;
    }
  }
  ext_info_offset += info->infoLen;
  return ext_info_offset;
}

uint64_t SetExtInfoOutputShapeType(char *ext_info_buf, uint64_t ext_info_offset,
                                   const std::shared_ptr<AnfNode> &anf_node, size_t output_num) {
  // deal3:output ShapeAndType
  auto *info = reinterpret_cast<aicpu::FWKAdapter::ExtInfo *>(ext_info_buf + ext_info_offset);
  info->infoType = static_cast<int32_t>(aicpu::FWKAdapter::FWK_ADPT_EXT_OUTPUT_SHAPE);
  info->infoLen = SizeToUint(output_num * sizeof(aicpu::FWKAdapter::ShapeAndType));
  ext_info_offset += aicpu::FWKAdapter::kExtInfoHeadSize;

  auto *outputs = reinterpret_cast<aicpu::FWKAdapter::ShapeAndType *>(ext_info_buf + ext_info_offset);
  for (size_t output_index = 0; output_index < output_num; output_index++) {
    auto output_shape = AnfAlgo::GetOutputDeviceShape(anf_node, output_index);
    TypeId output_type = AnfAlgo::GetOutputDeviceDataType(anf_node, output_index);
    int32_t output_data_type = AicpuOpUtil::MsTypeToProtoType(output_type);
    outputs[output_index].type = output_data_type;

    size_t output_shape_index = 0;
    for (; output_shape_index < output_shape.size(); output_shape_index++) {
      outputs[output_index].dims[output_shape_index] = output_shape[output_shape_index];
    }
    if (output_shape_index < aicpu::FWKAdapter::kMaxShapeDims) {
      outputs[output_index].dims[output_shape_index] = LLONG_MIN;
    }
  }

  ext_info_offset += info->infoLen;
  return ext_info_offset;
}

uint64_t SetExtInfoAsyncWait(char *ext_info_buf, uint64_t ext_info_offset) {
  // deal5: async wait
  auto *info = reinterpret_cast<aicpu::FWKAdapter::ExtInfo *>(ext_info_buf + ext_info_offset);
  info->infoType = static_cast<int32_t>(aicpu::FWKAdapter::FWK_ADPT_EXT_ASYNCWAIT);
  info->infoLen = sizeof(aicpu::FWKAdapter::AsyncWait);
  ext_info_offset += aicpu::FWKAdapter::kExtInfoHeadSize;
  aicpu::FWKAdapter::AsyncWait *wait_info =
    reinterpret_cast<aicpu::FWKAdapter::AsyncWait *>(ext_info_buf + ext_info_offset);
  wait_info->waitType = aicpu::FWKAdapter::FWK_ADPT_WAIT_TYPE_NULL;
  wait_info->waitId = 0;
  wait_info->timeOut = 0;
  wait_info->reserved = 0;
  ext_info_offset += info->infoLen;
  return ext_info_offset;
}

uint64_t SetExtInfoBitMap(char *ext_info_buf, uint64_t ext_info_offset, uint64_t bitmap) {
  // deal2: bit map
  auto *info = reinterpret_cast<aicpu::FWKAdapter::ExtInfo *>(ext_info_buf + ext_info_offset);
  info->infoType = static_cast<int32_t>(aicpu::FWKAdapter::FWK_ADPT_EXT_BITMAP);
  info->infoLen = sizeof(uint64_t);
  ext_info_offset += aicpu::FWKAdapter::kExtInfoHeadSize;
  uint64_t *bit_map = reinterpret_cast<uint64_t *>(ext_info_buf + ext_info_offset);
  *bit_map = bitmap;
  ext_info_offset += info->infoLen;
  return ext_info_offset;
}

uint64_t GenerateUniqueKernelId() {
  if (g_aicpu_kernel_id == ULLONG_MAX) {
    g_aicpu_kernel_id = 0;
  }
  return g_aicpu_kernel_id++;
}

uint64_t GenerateUniqueSessionId() {
  if (g_aicpu_session_id == ULLONG_MAX) {
    g_aicpu_session_id = 0;
  }
  return g_aicpu_session_id++;
}

uint64_t SetExtInfoSessionInfo(char *ext_info_buf, uint64_t ext_info_offset) {
  // deal5: async wait
  auto *info = reinterpret_cast<aicpu::FWKAdapter::ExtInfo *>(ext_info_buf + ext_info_offset);
  info->infoType = static_cast<int32_t>(aicpu::FWKAdapter::FWK_ADPT_EXT_SESSION_INFO);
  info->infoLen = sizeof(SessionInfo);
  ext_info_offset += aicpu::FWKAdapter::kExtInfoHeadSize;
  SessionInfo *session_info = reinterpret_cast<SessionInfo *>(ext_info_buf + ext_info_offset);
  session_info->sessionId = GenerateUniqueSessionId();
  session_info->kernelId = GenerateUniqueKernelId();
  session_info->sessFlag = false;
  ext_info_offset += info->infoLen;
  return ext_info_offset;
}

void CreateExtInfo(const std::shared_ptr<AnfNode> &anf_node, const std::shared_ptr<AicpuOpKernelMod> &kernel_mod_ptr) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
  auto cnode = anf_node->cast<CNodePtr>();
  if (cnode == nullptr) {
    return;
  }

  auto op_name = common::AnfAlgo::GetCNodeName(anf_node);
  if (!common::AnfAlgo::IsDynamicShape(anf_node) && op_name != kGetNextOpName) {
    return;
  }

  uint64_t ext_info_head_len = aicpu::FWKAdapter::kExtInfoHeadSize;
  std::string ext_info;
  size_t input_num = common::AnfAlgo::GetInputTensorNum(anf_node);
  size_t output_num = AnfAlgo::GetOutputTensorNum(anf_node);

  // 1.addr:unknown shape type
  uint64_t ext_info_len = ext_info.size();
  ext_info_len += ext_info_head_len + sizeof(int32_t);

  // 2.addr:bitmap, value: uint64_t
  ext_info_len += ext_info_head_len + sizeof(uint64_t);

  // 3.addr:input ShapeAndType
  ext_info_len += ext_info_head_len + input_num * sizeof(aicpu::FWKAdapter::ShapeAndType);

  // 4.addr:output ShapeAndType
  ext_info_len += ext_info_head_len + output_num * sizeof(aicpu::FWKAdapter::ShapeAndType);

  // 5.addr:session info
  ext_info_len += ext_info_head_len + sizeof(SessionInfo);

  // 5.addr:getnext async wait
  if (op_name == kGetNextOpName) {
    ext_info_len += (ext_info_head_len + sizeof(aicpu::FWKAdapter::AsyncWait));
  }

  uint64_t ext_info_offset = ext_info.size();
  ext_info.resize(ext_info_len, 0);
  char *ext_info_buf = ext_info.data();

  ::ge::UnknowShapeOpType shape_type = ::ge::UnknowShapeOpType::DEPEND_IN_SHAPE;
  if (IsOneOfComputeDepend(op_name)) {
    shape_type = ::ge::UnknowShapeOpType::DEPEND_COMPUTE;
  }
  ext_info_offset = SetExtInfoShapeType(ext_info_buf, ext_info_offset, shape_type);
  // if bitmap = 1, means static_shape, if bitmap = 0, means dynamic_shape
  uint64_t bitmap = 1;
  if (common::AnfAlgo::IsDynamicShape(anf_node)) {
    bitmap = 0;
  }
  ext_info_offset = SetExtInfoBitMap(ext_info_buf, ext_info_offset, bitmap);
  ext_info_offset = SetExtInfoInputShapeType(ext_info_buf, ext_info_offset, anf_node, input_num);
  ext_info_offset = SetExtInfoOutputShapeType(ext_info_buf, ext_info_offset, anf_node, output_num);
  ext_info_offset = SetExtInfoSessionInfo(ext_info_buf, ext_info_offset);
  if (op_name == kGetNextOpName) {
    ext_info_offset = SetExtInfoAsyncWait(ext_info_buf, ext_info_offset);
  }

  MS_LOG(INFO) << "Check ext_info_len:" << ext_info_len << " ext_info_offset:" << ext_info_offset;
  // set ext info
  kernel_mod_ptr->SetExtInfo(ext_info);
}

KernelModPtr AicpuOpBuild(const std::shared_ptr<AnfNode> &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  std::string op_name = common::AnfAlgo::GetCNodeName(anf_node);
  if (op_name == kInitDataSetQueue) {
    op_name = kInitData;
  }
  std::shared_ptr<AicpuOpKernelMod> kernel_mod_ptr;

  kernel_mod_ptr = std::make_shared<AicpuOpKernelMod>(anf_node);

  MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
  kernel_mod_ptr->SetNodeName(op_name);
  if (!CreateNodeDefBytes(anf_node, kernel_mod_ptr)) {
    MS_LOG(EXCEPTION) << "Create nodeDefBytes failed!";
  }

  CreateExtInfo(anf_node, kernel_mod_ptr);

  if (!SetIOSize(anf_node, kernel_mod_ptr)) {
    MS_LOG(EXCEPTION) << "Set input output size list failed.";
  }

  if (!AicpuOpKernelLoad::GetInstance().LoadAicpuKernelSo(anf_node, kernel_mod_ptr)) {
    MS_LOG(EXCEPTION) << "Aicpu kernel so load failed. task is " << anf_node->fullname_with_scope();
  }

  return kernel_mod_ptr;
}
}  // namespace kernel
}  // namespace mindspore
