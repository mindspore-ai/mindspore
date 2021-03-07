/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "schema/model_v0_generated.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/base/tile_base.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateTileParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  auto tile_prim = primitive->value_as_Tile();
  TileParameter *tile_param = reinterpret_cast<TileParameter *>(malloc(sizeof(TileParameter)));
  if (tile_param == nullptr) {
    MS_LOG(ERROR) << "malloc TileParameter failed.";
    return nullptr;
  }
  memset(tile_param, 0, sizeof(TileParameter));
  tile_param->op_parameter_.type_ = schema::PrimitiveType_TileFusion;
#ifdef SUPPORT_TRAIN
  auto multiples = tile_prim->multiples();
  tile_param->in_dim_ = multiples->size();
  for (int i = 0; i < tile_param->in_dim_; ++i) {
    tile_param->multiples_[i] = *(multiples->begin() + i);
  }
#else
  if (tile_prim->dims() != nullptr) {
    auto dims = tile_prim->dims();
    if (dims != nullptr) {
      for (size_t i = 0; i < dims->size(); i++) {
        tile_param->dims_[i] = static_cast<int>(dims->Get(i));
      }
    }
    tile_param->dims_size_ = tile_prim->dims()->size();
  }

#endif
  return reinterpret_cast<OpParameter *>(tile_param);
}
}  // namespace

Registry g_tileV0ParameterRegistry(schema::v0::PrimitiveType_Tile, PopulateTileParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
