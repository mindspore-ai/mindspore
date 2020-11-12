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

#include "tools/converter/parser/onnx/onnx_slice_parser.h"
#include <memory>
#include <vector>
#include <string>

namespace mindspore {
namespace lite {
STATUS OnnxSliceParser::InsertTensor(const std::vector<int> &onnx_val, const std::string &name,
                                     onnx::NodeProto *onnx_node) {
  std::unique_ptr<schema::TensorT> tensor = std::make_unique<schema::TensorT>();
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "new tensor failed";
    return RET_ERROR;
  }
  tensor->dataType = mindspore::kNumberTypeInt32;
  tensor->dims.push_back(onnx_val.size());
  tensor->format = schema::Format::Format_NCHW;
  tensor->nodeType = schema::NodeType::NodeType_ValueNode;
  int data_size = sizeof(int32_t) * onnx_val.size();
  tensor->data.resize(data_size);
  if (data_size != 0 &&
      memcpy_s(static_cast<void *>(tensor->data.data()), data_size, onnx_val.data(), data_size) != EOK) {
    MS_LOG(ERROR) << "memcpy_s failed";
    return RET_ERROR;
  }
  int tensor_num = OnnxTensorParser::GetInstance()->GetTensorCache()->GetCachedTensor().size();
  std::string tensor_name = name + std::to_string(tensor_num);
  OnnxTensorParser::GetInstance()->GetTensorCache()->AddTensor(tensor_name, tensor.release(), GRAPH_INPUT);
  onnx_node->add_input(tensor_name);
  return RET_OK;
}

STATUS OnnxSliceParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node,
                              schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx SliceParser";
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  std::unique_ptr<schema::StridedSliceT> attr = std::make_unique<schema::StridedSliceT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }

  std::vector<int> starts;
  std::vector<int> ends;
  std::vector<int> axes;
  std::vector<int> steps;
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "starts") {
      const int num = onnx_node_attr.ints_size();
      starts.clear();
      for (int i = 0; i < num; ++i) {
        starts.push_back(static_cast<int>(onnx_node_attr.ints()[i]));
      }
    } else if (attribute_name == "axes") {
      const int num = onnx_node_attr.ints_size();
      axes.clear();
      for (int i = 0; i < num; ++i) {
        axes.push_back(static_cast<int>(onnx_node_attr.ints()[i]));
      }
    } else if (attribute_name == "ends") {
      const int num = onnx_node_attr.ints_size();
      ends.clear();
      for (int i = 0; i < num; ++i) {
        ends.push_back(static_cast<int>(onnx_node_attr.ints()[i]));
      }
    } else if (attribute_name == "steps") {
      const int num = onnx_node_attr.ints_size();
      steps.clear();
      for (int i = 0; i < num; ++i) {
        steps.push_back(static_cast<int>(onnx_node_attr.ints()[i]));
      }
    }
  }
  if (axes.empty()) {
    for (size_t i = 0; i < starts.size(); ++i) {
      axes.push_back(i);
    }
  }
  if (steps.empty()) {
    steps.assign(starts.size(), 1);
  }
  onnx::NodeProto *slice_node = nullptr;
  for (auto &node : onnx_graph.node()) {
    if (&node == &onnx_node) {
      slice_node = const_cast<onnx::NodeProto *>(&node);
    }
  }
  int insert_num = 5 - onnx_node.input_size();
  int status = RET_OK;
  switch (insert_num) {
    case 4: {
      std::string name = "slice/starts/";
      status = InsertTensor(starts, name, slice_node);
    }
    case 3:
      if (status == RET_OK) {
        std::string name = "slice/ends/";
        status = InsertTensor(ends, name, slice_node);
      }
    case 2:
      if (status == RET_OK) {
        std::string name = "slice/axes/";
        status = InsertTensor(axes, name, slice_node);
      }
    case 1:
      if (status == RET_OK) {
        std::string name = "slice/steps/";
        status = InsertTensor(steps, name, slice_node);
      }
    default:
      if (status != RET_OK) {
        MS_LOG(ERROR) << "onnx slice insert tensor failed";
        return RET_ERROR;
      }
  }
  op->primitive->value.type = schema::PrimitiveType_StridedSlice;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

OnnxNodeRegistrar g_onnxSliceParser("Slice", new OnnxSliceParser());
}  // namespace lite
}  // namespace mindspore
