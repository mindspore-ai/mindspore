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

#include "mapper/tile_mapper.h"
#include <memory>
#include <utility>
#include <algorithm>
#include <vector>
#include "common/fetch_content.h"
#include "common/op_attr.h"
#include "common/op_enum.h"
#include "common/anf_util.h"
#include "ops/fusion/tile_fusion.h"
#include "op/tile_operator.h"

namespace mindspore {
namespace dpico {
STATUS TileMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                       const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }
  auto tile_prim = api::utils::cast<api::SharedPtr<ops::TileFusion>>(prim);
  MS_ASSERT(tile_prim != nullptr);

  auto tile_operator = std::make_unique<mapper::TileOperator>();
  if (tile_operator == nullptr) {
    MS_LOG(ERROR) << "tile_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, tile_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  tile_operator->SetOpType(mapper::OpType::TILE);
  if (tile_prim->GetAttr(ops::kDims) != nullptr) {
    auto dims = api::GetValue<std::vector<int64_t>>(tile_prim->GetAttr(ops::kDims));
    if (dims.size() == 1) {  // tf tile has multiple axis
      tile_operator->SetAxis(static_cast<int32_t>(dims.at(0)));
    }
  }

  DataInfo data_info;
  vector<int64_t> tiles;
  if (cnode->inputs().size() > kInputIndex2 &&
      FetchDataFromParameterNode(cnode, kInputIndex2, &data_info) == lite::RET_OK) {
    if (data_info.data_type_ != static_cast<int>(kNumberTypeInt32)) {
      MS_LOG(ERROR) << "data_type not correct";
      return RET_ERROR;
    }
    auto data = reinterpret_cast<int32_t *>(data_info.data_.data());
    if (data == nullptr) {
      MS_LOG(ERROR) << "data is nullptr. " << cnode->fullname_with_scope();
      return RET_ERROR;
    }
    int data_size;
    if (GetDataSizeFromTensor(&data_info, &data_size) != RET_OK) {
      MS_LOG(ERROR) << "get data size from tensor failed.";
      return RET_ERROR;
    }
    (void)std::transform(data, data + data_size, std::back_inserter(tiles),
                         [](const int64_t &value) { return static_cast<int64_t>(value); });
  } else if (tile_prim->GetAttr(dpico::kMultiples) != nullptr) {
    if (tile_prim->GetAttr(ops::kDims) != nullptr) {
      tiles = api::GetValue<std::vector<int64_t>>(tile_prim->GetAttr(dpico::kMultiples));
    }
  } else {
    MS_LOG(ERROR) << "null param";
    return RET_ERROR;
  }

  if (tiles.size() == 1) {
    tile_operator->SetTileTiles(static_cast<int32_t>(tiles.at(0)));
  } else {
    tile_operator->SetTileRepeats(tiles);
  }
  if (PushOfflineArgs(cnode, tile_operator.get(), 1) != RET_OK) {
    MS_LOG(ERROR) << "push offline args failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  base_operators->push_back(std::move(tile_operator));
  return RET_OK;
}
REG_MAPPER(TileFusion, TileMapper)
}  // namespace dpico
}  // namespace mindspore
