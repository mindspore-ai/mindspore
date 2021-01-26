/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "src/ops/tile.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/base/tile_base.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateTileParameter(const mindspore::lite::PrimitiveC *primitive) {
  TileParameter *tile_param = reinterpret_cast<TileParameter *>(malloc(sizeof(TileParameter)));
  if (tile_param == nullptr) {
    MS_LOG(ERROR) << "malloc TileParameter failed.";
    return nullptr;
  }
  memset(tile_param, 0, sizeof(TileParameter));
  tile_param->op_parameter_.type_ = primitive->Type();
  auto param = reinterpret_cast<mindspore::lite::Tile *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
#ifdef SUPPORT_TRAIN
  auto multiples = param->GetMultiples();
  tile_param->in_dim_ = multiples.size();
  for (int i = 0; i < tile_param->in_dim_; ++i) {
    tile_param->multiples_[i] = multiples[i];
  }
#else
  auto dims = param->GetDims();
  auto multiples = param->GetMultiples();
  for (size_t i = 0; i < kQuadrupleNum; ++i) {
    tile_param->multiples_[i] = 1;
  }
  if (!dims.empty() && !multiples.empty()) {
    for (size_t i = 0; i < dims.size(); ++i) {
      tile_param->multiples_[dims[i]] = multiples[i];
    }
  }
#endif
  return reinterpret_cast<OpParameter *>(tile_param);
}

Registry TileParameterRegistry(schema::PrimitiveType_Tile, PopulateTileParameter);

}  // namespace lite
}  // namespace mindspore
