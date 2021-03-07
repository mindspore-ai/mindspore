/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "src/ops/populate/populate_register.h"
#include "nnacl/base/tile_base.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateTileParameter(const void *prim) {
  TileParameter *tile_param = reinterpret_cast<TileParameter *>(malloc(sizeof(TileParameter)));
  if (tile_param == nullptr) {
    MS_LOG(ERROR) << "malloc TileParameter failed.";
    return nullptr;
  }
  memset(tile_param, 0, sizeof(TileParameter));
  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_TileFusion();
  tile_param->op_parameter_.type_ = primitive->value_type();
  auto dims = value->dims();
  if (dims != nullptr) {
    for (size_t i = 0; i < dims->size(); ++i) {
      tile_param->dims_[i] = static_cast<int>(dims->Get(i));
    }
    tile_param->dims_size_ = dims->size();
  }
  return reinterpret_cast<OpParameter *>(tile_param);
}

Registry TileParameterRegistry(schema::PrimitiveType_TileFusion, PopulateTileParameter, SCHEMA_CUR);

}  // namespace lite
}  // namespace mindspore
