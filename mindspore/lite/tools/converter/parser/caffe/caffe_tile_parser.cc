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

#include "tools/converter/parser/caffe/caffe_tile_parser.h"
#include <memory>
#include <vector>
#include "ops/fusion/tile_fusion.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr CaffeTileParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_unique<ops::TileFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto prim_c = prim->GetPrim();
  MS_CHECK_TRUE_RET(prim_c != nullptr, nullptr);

  const caffe::TileParameter &tile_param = proto.tile_param();
  std::vector<int64_t> dims;
  dims.clear();
  if (tile_param.has_axis()) {
    dims.push_back(tile_param.axis());
  } else {
    dims.push_back(1);
  }
  prim->set_dims(dims);

  std::vector<int32_t> multiples;
  multiples.clear();
  if (tile_param.has_tiles()) {
    multiples.push_back(tile_param.tiles());
  } else {
    multiples.push_back(1);
  }
  auto value_ptr = MakeValue(multiples);
  MS_CHECK_TRUE_RET(value_ptr != nullptr, nullptr);
  (void)prim_c->AddAttr("multiples", value_ptr);

  return prim->GetPrim();
}

CaffeNodeRegistrar g_caffeTileParser("Tile", new CaffeTileParser());
}  // namespace lite
}  // namespace mindspore
