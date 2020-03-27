/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "predict/converter/attr_utils/convert_util.h"

namespace mindspore {
namespace predict {
namespace utils {
TypePtr GetTypePtr(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  TypePtr type_ptr = anf_node->Type();
  MS_EXCEPTION_IF_NULL(type_ptr);
  if (type_ptr->isa<TensorType>()) {
    auto tensor_ptr = type_ptr->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(tensor_ptr);
    TypePtr elem = tensor_ptr->element();
    return elem;
  } else if (type_ptr->isa<Tuple>()) {
    auto tuple_ptr = type_ptr->cast<TuplePtr>();
    MS_EXCEPTION_IF_NULL(tuple_ptr);
    auto tuple_i = (*tuple_ptr)[0];
    MS_EXCEPTION_IF_NULL(tuple_i);
    if (tuple_i->isa<TensorType>()) {
      auto tensor_ptr = tuple_i->cast<TensorTypePtr>();
      MS_EXCEPTION_IF_NULL(tensor_ptr);
      TypePtr elem = tensor_ptr->element();
      MS_EXCEPTION_IF_NULL(elem);
      return elem;
    } else if (tuple_i->isa<Number>()) {
      return type_ptr;
    } else {
      MS_LOG(EXCEPTION) << "unsupported type: " << type_ptr->ToString();
    }
  } else if (type_ptr->isa<Number>()) {
    return type_ptr;
  }
  std::string type_name = type_ptr->ToString();
  MS_LOG(EXCEPTION)
    << "The output type of node should be a tensor type a number or a tuple  of tensor type, but this is: "
    << type_name;
}

MsDataType GetMSDataType(TypeId ori_data_type) {
  MsDataType dst_data_type;
  switch (ori_data_type) {
    case kNumberTypeFloat16:
      dst_data_type = mindspore::predict::DataType_DT_FLOAT16;
      return dst_data_type;
    case kNumberTypeFloat32:
      dst_data_type = mindspore::predict::DataType_DT_FLOAT;
      return dst_data_type;
    case kNumberTypeInt8:
      dst_data_type = mindspore::predict::DataType_DT_INT8;
      return dst_data_type;
    case kNumberTypeInt32:
      dst_data_type = mindspore::predict::DataType_DT_INT32;
      return dst_data_type;
    case kNumberTypeUInt8:
      dst_data_type = mindspore::predict::DataType_DT_UINT8;
      return dst_data_type;
    case kNumberTypeUInt32:
      dst_data_type = mindspore::predict::DataType_DT_UINT32;
      return dst_data_type;
    case kTypeUnknown:
      dst_data_type = mindspore::predict::DataType_DT_UNDEFINED;
      return dst_data_type;
    default:
      MS_LOG(EXCEPTION) << "Ms don't support this DataType";
  }
}

MsFormat GetMsFormat(const std::string &format_str) {
  if (format_str == kOpFormat_DEFAULT) {
    MsFormat ms_format = predict::Format_NCHW;
    return ms_format;
  } else {
    // all middle format default to NCHW
    return predict::Format_NCHW;
  }
}

TensorPtr GetParaAscendTensor(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  if (!anf_node->isa<Parameter>()) {
    return nullptr;
  }
  auto device_type_id = AnfAlgo::GetOutputDeviceDataType(anf_node, 0);
  // device type_ptr
  auto device_type_ptr = GetTypePtr(anf_node);
  // device shape
  auto shape = AnfAlgo::GetOutputDeviceShape(anf_node, 0);
  std::vector<int> tensor_shape;
  (void)std::transform(shape.begin(), shape.end(), std::back_inserter(tensor_shape), SizeToInt);
  // device format
  auto format = AnfAlgo::GetOutputFormat(anf_node, 0);
  // device tensor
  TensorPtr device_tensor = std::make_shared<tensor::Tensor>(device_type_id, tensor_shape);
  // device info
  device_tensor->SetDeviceInfo(format, device_type_ptr);
  return device_tensor;
}

TensorPtr GetParaCpuTensor(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  if (!(anf_node->isa<Parameter>())) {
    return nullptr;
  } else {
    auto ori_type_id = AnfAlgo::GetOutputInferDataType(anf_node, 0);
    auto ori_type_ptr = GetTypePtr(anf_node);
    auto ori_shape = AnfAlgo::GetOutputInferShape(anf_node, 0);
    std::vector<int> tensor_shape;
    (void)std::transform(ori_shape.begin(), ori_shape.end(), std::back_inserter(tensor_shape), SizeToInt);
    auto ori_format = AnfAlgo::GetOutputFormat(anf_node, 0);
    TensorPtr cpu_tensor = std::make_shared<tensor::Tensor>(ori_type_id, tensor_shape);
    cpu_tensor->SetDeviceInfo(ori_format, ori_type_ptr);
    return cpu_tensor;
  }
}

TensorPtr GetValueTensor(const ValueNodePtr &const_node) {
  MS_EXCEPTION_IF_NULL(const_node);
  auto value_ptr = const_node->value();
  MS_EXCEPTION_IF_NULL(value_ptr);
  if (!value_ptr->isa<tensor::Tensor>()) {
    return nullptr;
  }
  TensorPtr tensor = value_ptr->cast<TensorPtr>();
  MS_EXCEPTION_IF_NULL(tensor);
  auto data_type = tensor->Dtype();
  MS_EXCEPTION_IF_NULL(data_type);
  auto type_id = data_type->type_id();
  auto shape = tensor->shape();
  TensorPtr tensor_constant = std::make_shared<tensor::Tensor>(type_id, shape);
  tensor_constant->SetDeviceInfo(tensor->device_info().format_, tensor->device_info().data_type_);
  return tensor_constant;
}

TensorPtr GetKernelCpuTensor(const CNodePtr &c_node_ptr, size_t inx) {
  if (c_node_ptr == nullptr || inx >= AnfAlgo::GetOutputTensorNum(c_node_ptr)) {
    MS_LOG(ERROR) << "GetKernelCpuTensor failed";
    return nullptr;
  }
  auto ori_shape = AnfAlgo::GetOutputInferShape(c_node_ptr, inx);
  auto ori_type_id = AnfAlgo::GetOutputInferDataType(c_node_ptr, inx);
  std::vector<int> tensor_shape;
  (void)std::transform(ori_shape.begin(), ori_shape.end(), std::back_inserter(tensor_shape), SizeToInt);
  auto ori_output_type = GetTypePtr(c_node_ptr);
  TensorPtr device_tensor = std::make_shared<tensor::Tensor>(ori_type_id, tensor_shape);
  auto format = AnfAlgo::GetOutputFormat(c_node_ptr, inx);
  device_tensor->SetDeviceInfo(format, ori_output_type);
  return device_tensor;
}

TensorPtr GetKernelAscendTensor(const CNodePtr &c_node_ptr, size_t inx) {
  if (c_node_ptr == nullptr || inx >= AnfAlgo::GetOutputTensorNum(c_node_ptr)) {
    MS_LOG(ERROR) << "GetKernelAscendTensor failed";
    return nullptr;
  }
  auto shape = AnfAlgo::GetOutputDeviceShape(c_node_ptr, inx);
  std::vector<int> tensor_shape;
  (void)std::transform(shape.begin(), shape.end(), std::back_inserter(tensor_shape), SizeToInt);
  auto format = AnfAlgo::GetOutputFormat(c_node_ptr, inx);
  auto type_id = AnfAlgo::GetOutputDeviceDataType(c_node_ptr, inx);
  auto output_type_ptr = GetTypePtr(c_node_ptr);
  TensorPtr device_tensor = std::make_shared<tensor::Tensor>(type_id, tensor_shape);
  device_tensor->SetDeviceInfo(format, output_type_ptr);
  return device_tensor;
}

TensorPtr GetOutputTensor(const AnfNodePtr &out_node, size_t inx) {
  MS_EXCEPTION_IF_NULL(out_node);
  auto shape = AnfAlgo::GetOutputInferShape(out_node, inx);
  std::vector<int> tensor_shape;
  (void)std::transform(shape.begin(), shape.end(), std::back_inserter(tensor_shape), SizeToInt);
  auto type_id = AnfAlgo::GetOutputInferDataType(out_node, inx);
  auto output_type_ptr = GetTypePtr(out_node);
  auto format = AnfAlgo::GetOutputFormat(out_node, inx);
  TensorPtr output_tensor = std::make_shared<tensor::Tensor>(type_id, tensor_shape);
  output_tensor->SetDeviceInfo(format, output_type_ptr);
  return output_tensor;
}

bool FindNodeInMap(const std::unordered_map<MsKernelKey, int> &node_map, const AnfNodePtr &node) {
  return std::any_of(node_map.begin(), node_map.end(),
                     [node](const std::pair<MsKernelKey, int> &kernel_key) { return kernel_key.first == node.get(); });
}

bool SaveDeviceModelUtil(const std::shared_ptr<GraphDefT> &new_ms_graph_ptr, const std::string &save_path_name,
                         SubGraphDefT *sub_graph) {
  MS_EXCEPTION_IF_NULL(new_ms_graph_ptr);
  MS_EXCEPTION_IF_NULL(sub_graph);
  // save mindspore schema to file
  new_ms_graph_ptr->name = "default_graph";
  std::unique_ptr<mindspore::predict::SubGraphDefT> sub_graph_ptr(sub_graph);
  new_ms_graph_ptr->subgraphs.emplace_back(std::move(sub_graph_ptr));
  // get flatbuffer builder
  flatbuffers::FlatBufferBuilder builder(1024);
  auto offset = mindspore::predict::GraphDef::Pack(builder, new_ms_graph_ptr.get());
  builder.Finish(offset);
  auto size = builder.GetSize();
  if (size == 0) {
    MS_LOG(ERROR) << "builder has no size";
    return false;
  }
  auto content = builder.GetBufferPointer();
  std::ofstream output(save_path_name);
  if (!output.is_open()) {
    MS_LOG(EXCEPTION) << "mindspore.mindspoire output failed";
  }
  (void)output.write((const char *)content, size);
  output.close();
  return true;
}
}  // namespace utils
}  // namespace predict
}  // namespace mindspore
