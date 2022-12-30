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

#include "tools/converter/adapter/acl/mapper/tile_fusion_mapper.h"
#include <memory>
#include <vector>
#include <algorithm>
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/converter/adapter/acl/common/utils.h"
#include "ops/op_utils.h"
#include "ops/tile.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
namespace {
constexpr auto kNameTileInputNum = 3;
}  // namespace

STATUS TileFusionMapper::Mapper(const CNodePtr &cnode) {
  MS_CHECK_TRUE_RET(cnode != nullptr, RET_ERROR);
  if (cnode->size() != kNameTileInputNum) {
    MS_LOG(ERROR) << "The input size of tile must be " << kNameTileInputNum
                  << ", while the real size is: " << cnode->size();
    return RET_ERROR;
  }
  auto repeats_input = cnode->input(kNameTileInputNum - 1);
  MS_CHECK_TRUE_RET(repeats_input != nullptr, RET_ERROR);
  if (utils::isa<ParameterPtr>(repeats_input)) {
    MS_LOG(WARNING) << "The repeats node is not parameter.";
    ParameterPtr repeats_param = repeats_input->cast<ParameterPtr>();
    MS_CHECK_TRUE_RET(repeats_param != nullptr, RET_ERROR);
    auto data = acl::GetIntParameterData(repeats_param);
    std::vector<int64_t> multiples;
    std::transform(data.begin(), data.end(), std::back_inserter(multiples),
                   [](int32_t x) -> int64_t { return static_cast<int64_t>(x); });
    ValueNodePtr value_node = NewValueNode<std::vector<int64_t>>(multiples);
    MS_CHECK_TRUE_RET(value_node != nullptr, RET_ERROR);
    std::vector<int64_t> shape_vec_shape = {static_cast<int64_t>(multiples.size())};
    auto abstract = std::make_shared<abstract::AbstractTensor>(kInt64, shape_vec_shape);
    value_node->set_abstract(abstract);
    cnode->set_input(kNameTileInputNum - 1, value_node);
  }

  ops::Tile tile;
  auto dst_prim = tile.GetPrim();
  if (MoveAttrMap(cnode, dst_prim) != RET_OK) {
    MS_LOG(ERROR) << "TileFusion mapper failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameTileFusion, TileFusionMapper)
}  // namespace lite
}  // namespace mindspore
