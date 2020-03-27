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
#include "kernel/aicpu/aicpu_kernel_build.h"
#include <google/protobuf/text_format.h>
#include <fstream>
#include <utility>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <map>
#include "device/kernel_runtime.h"
#include "kernel/aicpu/aicpu_kernel_mod.h"
#include "kernel/akg/akgkernelbuild.h"
#include "proto/tensor.pb.h"
#include "proto/tensor_shape.pb.h"
#include "proto/attr.pb.h"
#include "proto/node_def.pb.h"
#include "session/anf_runtime_algorithm.h"
#include "common/utils.h"
#include "kernel/aicpu/aicpu_util.h"
#include "session/kernel_graph.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
using FNodeAttrHandle = std::function<void(const std::shared_ptr<AnfNode> &anf_node, mindspore::NodeDef *proto)>;

const std::vector<std::string> local_framework_op_vec = {kInitDataSetQueue, kGetNext, kDropoutGenMask, kPrint};

void InitDataSetQueueAttr(const std::shared_ptr<AnfNode> &anf_node, mindspore::NodeDef *proto) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(proto);

  ::google::protobuf::Map<::std::string, ::mindspore::AttrValue> *node_attr = proto->mutable_attrs();
  MS_EXCEPTION_IF_NULL(node_attr);
  std::string channel_name = AnfAlgo::GetNodeAttr<std::string>(anf_node, kQueueName);
  (*node_attr)[kChannelName].set_s(channel_name);
}

void GetNextAttr(const std::shared_ptr<AnfNode> &anf_node, mindspore::NodeDef *proto) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(proto);

  ::google::protobuf::Map<::std::string, ::mindspore::AttrValue> *node_attr = proto->mutable_attrs();
  MS_EXCEPTION_IF_NULL(node_attr);
  std::string shared_name = AnfAlgo::GetNodeAttr<std::string>(anf_node, kSharedName);
  (*node_attr)[kChannelName].set_s(shared_name);
}

void DropoutGenMaskAttr(const std::shared_ptr<AnfNode> &anf_node, mindspore::NodeDef *proto) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(proto);

  ::google::protobuf::Map<::std::string, ::mindspore::AttrValue> *node_attr = proto->mutable_attrs();
  MS_EXCEPTION_IF_NULL(node_attr);
  int seed = AnfAlgo::GetNodeAttr<int>(anf_node, kSeed);
  int seed2 = AnfAlgo::GetNodeAttr<int>(anf_node, kSeed2);
  (*node_attr)["seed"].set_i(seed);
  (*node_attr)["seed2"].set_i(seed2);
}

void CreateAttrFuncMap(std::map<std::string, FNodeAttrHandle> *mOpAttrFuncMap) {
  (void)mOpAttrFuncMap->emplace(std::pair<std::string, FNodeAttrHandle>(kInitDataSetQueue, InitDataSetQueueAttr));
  (void)mOpAttrFuncMap->emplace(std::pair<std::string, FNodeAttrHandle>(kGetNext, GetNextAttr));
  (void)mOpAttrFuncMap->emplace(std::pair<std::string, FNodeAttrHandle>(kDropoutGenMask, DropoutGenMaskAttr));
}

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
      auto input_node = cnode->inputs()[i + 1];
      if (input_node->isa<ValueNode>()) {
        auto value_ptr = GetValueNode(input_node);
        auto value = GetValue<std::string>(value_ptr);
        input_size_list->push_back(value.size());
      }
    } else {
      auto type_ptr = TypeIdToType(AnfAlgo::GetInputDeviceDataType(anf_node, i));
      MS_EXCEPTION_IF_NULL(type_ptr);
      int size_i = 1;
      for (size_t j = 0; j < shape_i.size(); j++) {
        IntMulWithOverflowCheck(size_i, static_cast<int>(shape_i[j]), &size_i);
      }
      size_t type_byte = GetTypeByte(type_ptr);
      if (type_byte == 0) {
        return false;
      }
      IntMulWithOverflowCheck(size_i, SizeToInt(type_byte), &size_i);
      input_size_list->push_back(IntToSize(size_i));
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
    int size_i = 1;
    for (size_t j = 0; j < shape_i.size(); j++) {
      IntMulWithOverflowCheck(size_i, static_cast<int>(shape_i[j]), &size_i);
    }
    size_t type_byte = GetTypeByte(type_ptr);
    if (type_byte == 0) {
      return false;
    }
    IntMulWithOverflowCheck(size_i, SizeToInt(type_byte), &size_i);
    output_size_list.push_back(IntToSize(size_i));
  }
  kernel_mod_ptr->SetOutputSizeList(output_size_list);

  return true;
}

void SetNodeAttr(const std::shared_ptr<AnfNode> &anf_node, mindspore::NodeDef *proto) {
  std::string op_name = AnfAlgo::GetCNodeName(anf_node);
  if (op_name == "InitDataSetQueue") {
    op_name = "InitData";
  }
  if (op_name == "Print") {
    return;
  }
  std::map<std::string, FNodeAttrHandle> mOpAttrFuncMap;
  CreateAttrFuncMap(&mOpAttrFuncMap);
  FNodeAttrHandle func_ptr = nullptr;
  auto iter = mOpAttrFuncMap.find(op_name);
  if (iter != mOpAttrFuncMap.end()) {
    func_ptr = iter->second;
    MS_EXCEPTION_IF_NULL(func_ptr);
    func_ptr(anf_node, proto);
  } else {
    MS_LOG(ERROR) << "Don't support node [" << op_name << "] to set nodedef of attr";
  }
  MS_LOG(INFO) << "Set node attr end!";
}

void SetNodeInputs(const std::shared_ptr<AnfNode> &anf_node, mindspore::NodeDef *proto) {
  size_t input_num = AnfAlgo::GetInputTensorNum(anf_node);
  if (input_num == 0) {
    MS_LOG(INFO) << "Node [" << AnfAlgo::GetCNodeName(anf_node) << "] does not have input. ";
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
  size_t output_num = AnfAlgo::GetOutputTensorNum(anf_node);
  if (output_num == 0) {
    MS_LOG(INFO) << "Node [" << AnfAlgo::GetCNodeName(anf_node) << "] does not have output. ";
    return;
  }

  for (size_t output_index = 0; output_index < output_num; output_index++) {
    ::mindspore::Tensor *node_outputs = proto->add_outputs();
    std::vector<size_t> output_shape = AnfAlgo::GetOutputDeviceShape(anf_node, output_index);
    mindspore::TensorShape *tensorShape = node_outputs->mutable_tensor_shape();
    for (auto item : output_shape) {
      mindspore::TensorShape_Dim *dim = tensorShape->add_dim();
      dim->set_size((::google::protobuf::int64)item);
    }

    TypeId output_type = AnfAlgo::GetOutputDeviceDataType(anf_node, output_index);

    int32_t output_data_type = AicpuOpUtil::MsTypeToProtoType(output_type);
    node_outputs->set_tensor_type((mindspore::DataType)output_data_type);

    node_outputs->set_mem_device("HBM");
  }
}

void SetNodedefProto(const std::shared_ptr<AnfNode> &anf_node, mindspore::NodeDef *proto) {
  MS_LOG(INFO) << "SetNodedefProto entry";
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(proto);

  std::string op_name = AnfAlgo::GetCNodeName(anf_node);
  if (op_name == "InitDataSetQueue") {
    op_name = "InitData";
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
  MS_LOG(INFO) << "CreateNodeDefBytes entry";
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
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
  if (op_name == "InitDataSetQueue") {
    op_name = "InitData";
  }
  auto kernel_mod_ptr = std::make_shared<AicpuOpKernelMod>();
  MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
  kernel_mod_ptr->SetAnfNode(anf_node);
  kernel_mod_ptr->SetNodeName(op_name);
  auto iter = std::find(local_framework_op_vec.begin(), local_framework_op_vec.end(), op_name);
  if (iter != local_framework_op_vec.end()) {
    if (!CreateNodeDefBytes(anf_node, kernel_mod_ptr)) {
      MS_LOG(EXCEPTION) << "Create nodeDefBytes faild!";
    }
  } else {
    MS_LOG(EXCEPTION) << "Aicpu don't support node [" << op_name << "]";
  }

  if (!SetIOSize(anf_node, kernel_mod_ptr)) {
    MS_LOG(EXCEPTION) << "Set input output size list failed.";
  }

  return kernel_mod_ptr;
}
}  // namespace kernel
}  // namespace mindspore
