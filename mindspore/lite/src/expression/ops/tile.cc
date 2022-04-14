/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "src/expression/ops/tile.h"
#include <memory>
#include "src/expression/ops.h"
#include "nnacl/base/tile_base.h"

namespace mindspore {
namespace lite {
TileM::TileM(const std::vector<int> &multiples) : Node() {
  expr()->SetSize(C2NUM);
  TileParameter *param = reinterpret_cast<TileParameter *>(calloc(1, sizeof(TileParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << " cannot allocate ConvParameter";
    return;
  }
  SetOpParam(param);
  set_name(UniqueName("Tile"));
  set_primitive(schema::PrimitiveType_TileFusion);
  Node::CreateConstTensor(C1NUM, {static_cast<int32_t>(multiples.size())}, kNumberTypeInt32, KHWC, "axis",
                          multiples.data());
}

int TileM::UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) {
  auto tile_param = reinterpret_cast<const TileParameter *>(OpParam());
  auto prim = new (std::nothrow) schema::TileFusionT;
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate primitive";
    return RET_ERROR;
  }
  for (size_t i = 0; i < tile_param->dims_size_; i++) {
    prim->dims.push_back(tile_param->dims_[i]);
  }
  cnode->primitive->value.value = prim;
  return RET_OK;
}

namespace NN {
Node *Tile(const std::vector<int> &multiples) {
  auto node = new (std::nothrow) TileM(multiples);
  if (node == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate tile node";
  }
  return node;
}
}  // namespace NN
}  // namespace lite
}  // namespace mindspore
