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
#include "ops/fusion/tile_fusion.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TFTileParser::Parse(const tensorflow::NodeDef &tf_op,
                                     const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                     std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::TileFusion>();

  tensorflow::AttrValue attr_value;
  std::vector<int64_t> dims;
  const auto &tensor_proto = attr_value.tensor();
  if (tensor_proto.int_val_size() > 0) {
    for (int i = 0; i < tensor_proto.int_val_size(); ++i) {
      dims.push_back(i);
    }
  } else {
    auto data_num = tensor_proto.tensor_content().size() / sizeof(int32_t);
    for (size_t i = 0; i < data_num; ++i) {
      dims.push_back(i);
    }
  }
  prim->set_dims(dims);

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }
  return prim.release();
}

TFNodeRegistrar g_tfTileParser("Tile", new TFTileParser());
}  // namespace lite
}  // namespace mindspore
