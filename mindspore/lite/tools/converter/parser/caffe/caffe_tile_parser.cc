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

#include "mindspore/lite/tools/converter/parser/caffe/caffe_tile_parser.h"
#include <memory>
#include <vector>

namespace mindspore {
namespace lite {
STATUS CaffeTileParser::Parse(const caffe::LayerParameter &proto,
                              const caffe::LayerParameter &weight,
                              schema::CNodeT *op,
                              std::vector<schema::TensorT *> *weightVec) {
  MS_LOG(DEBUG) << "parse CaffeTileParser";
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

  const caffe::TileParameter tile_param = proto.tile_param();
  std::vector<int> dims;
  std::vector<int> multiples;
  dims.clear();
  multiples.clear();
  if (tile_param.has_axis()) {
    dims.push_back(tile_param.axis());
  } else {
    dims.push_back(1);
  }
  if (tile_param.has_tiles()) {
    multiples.push_back(tile_param.tiles());
  } else {
    multiples.push_back(1);
  }

  attr->dims = dims;
  attr->multiples = multiples;

  op->primitive->value.type = schema::PrimitiveType_Tile;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

CaffeNodeRegistrar g_caffeTileParser("Tile", new CaffeTileParser());
}  // namespace lite
}  // namespace mindspore

