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
#include "tools/converter/parser/tf/tf_tile_parser.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"

namespace mindspore {
namespace lite {
STATUS TFTileParser::Parse(const tensorflow::NodeDef &tf_op,
                           const std::map<string, const tensorflow::NodeDef *> &tf_node_map, PrimitiveC **primitiveC,
                           std::vector<std::string> *inputs, int *output_size) {
  MS_LOG(INFO) << "TF TileParser";
  if (primitiveC == nullptr || output_size == nullptr) {
    MS_LOG(ERROR) << "primitiveC is nullptr";
    return RET_NULL_PTR;
  }

  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "New PrimitiveT failed";
    return RET_NULL_PTR;
  }
  auto attr = std::make_unique<schema::TileT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new attr failed";
    return RET_NULL_PTR;
  }

  *output_size = 1;
  auto status = AddOpInput(tf_op, 0, inputs);

  auto multiplies_node = GetConstInputNode(tf_node_map, tf_op.input(1));
  tensorflow::AttrValue attr_value;
  if (multiplies_node != nullptr && TensorFlowUtils::FindAttrValue(*multiplies_node, "value", &attr_value)) {
    auto tensor_proto = attr_value.tensor();
    if (tensor_proto.int_val_size() > 0) {
      for (int i = 0; i < tensor_proto.int_val_size(); ++i) {
        attr->dims.push_back(i);
        attr->multiples.push_back(tensor_proto.int_val(i));
      }
    } else {
      auto data_num = tensor_proto.tensor_content().size() / sizeof(int32_t);
      auto data = reinterpret_cast<const int32_t *>(tensor_proto.tensor_content().data());
      for (size_t i = 0; i < data_num; ++i) {
        attr->dims.push_back(i);
        attr->multiples.push_back(data[i]);
      }
    }
  } else {
    AddOpInput(tf_op, 1, inputs);
  }

  primitive->value.type = schema::PrimitiveType_Tile;
  primitive->value.value = attr.release();
  *primitiveC = PrimitiveC::Create(primitive.release());
  if (*primitiveC == nullptr) {
    MS_LOG(ERROR) << "primitiveC is nullptr";
    return RET_ERROR;
  }

  return status;
}
TFNodeRegistrar g_tfTileParser("Tile", new TFTileParser());
}  // namespace lite
}  // namespace mindspore
