/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "src/common/ops/operator_populate/operator_populate_register.h"
#include "nnacl/base/tile_base.h"
#include "ops/tile.h"
#include "ops/fusion/tile_fusion.h"
#include "ops/op_name.h"
using mindspore::ops::kNameTile;
using mindspore::ops::kNameTileFusion;
using mindspore::schema::PrimitiveType_TileFusion;
namespace mindspore {
namespace lite {
OpParameter *PopulateTileOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<TileParameter *>(PopulateOpParameter<TileParameter>(base_operator));
  if (param == nullptr) {
    MS_LOG(ERROR) << "Make OpParameter ptr failed";
    return nullptr;
  }

  auto attr = base_operator->GetPrim()->GetAttr(mindspore::ops::kDims);
  if (attr != nullptr) {
    auto dims = GetValue<std::vector<int64_t>>(attr);
    if (!dims.empty()) {
      if (dims.size() > MAX_SHAPE_SIZE) {
        MS_LOG(ERROR) << "Invalid dims size " << dims.size();
        free(param);
        return nullptr;
      }
      for (size_t i = 0; i < dims.size(); ++i) {
        if (static_cast<int>(dims[i]) < 0) {
          free(param);
          return nullptr;
        }
        param->dims_[i] = static_cast<int>(dims[i]);
      }
      param->dims_size_ = dims.size();
    }
  }
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameTile, PrimitiveType_TileFusion, PopulateTileOpParameter)
REG_OPERATOR_POPULATE(kNameTileFusion, PrimitiveType_TileFusion, PopulateTileOpParameter)
}  // namespace lite
}  // namespace mindspore
