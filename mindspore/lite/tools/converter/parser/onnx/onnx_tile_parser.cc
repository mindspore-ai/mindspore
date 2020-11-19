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

#include "tools/converter/parser/onnx/onnx_tile_parser.h"
#include <memory>
#include <numeric>
#include <vector>
#include "tools/converter/parser/onnx/onnx_tensor_parser.h"

namespace mindspore {
namespace lite {
STATUS OnnxTileParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx TileParser";
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  std::unique_ptr<schema::TileT> attr = std::make_unique<schema::TileT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }
  const auto &onnx_tile_multiple = onnx_node.input(1);
  int index = OnnxTensorParser::GetInstance()->GetTensorCache()->FindTensor(onnx_tile_multiple);
  if (index == -1) {
    MS_LOG(ERROR) << "can not find node: " << onnx_tile_multiple;
    return RET_ERROR;
  }
  auto tile_attr = OnnxTensorParser::GetInstance()->GetTensorCache()->GetCachedTensor()[index];
  if (tile_attr->data.data() == nullptr) {
    MS_LOG(ERROR) << "power's attr pow can't be obtained.";
    return RET_INVALID_OP_ATTR;
  }
  int element_size = std::accumulate(tile_attr->dims.begin(), tile_attr->dims.end(), 1, std::multiplies<int>());
  std::vector<int> multiples;
  std::vector<int> dims;
  for (int i = 0; i < element_size; ++i) {
    multiples.push_back(reinterpret_cast<int *>(tile_attr->data.data())[i]);
    dims.push_back(i);
  }
  attr->multiples = multiples;
  attr->dims = dims;
  op->primitive->value.type = schema::PrimitiveType_Tile;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

OnnxNodeRegistrar g_onnxTileParser("Tile", new OnnxTileParser());
}  // namespace lite
}  // namespace mindspore
