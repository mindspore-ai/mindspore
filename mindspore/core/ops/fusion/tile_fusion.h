/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_TILE_FUSION_H_
#define MINDSPORE_CORE_OPS_TILE_FUSION_H_
#include <vector>

#include "ops/tile.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameTileFusion = "TileFusion";
/// \brief TileFusion defined Tile operator prototype of lite.
class MIND_API TileFusion : public Tile {
 public:
  MIND_API_BASE_MEMBER(TileFusion);
  /// \brief Constructor.
  TileFusion() : Tile(kNameTileFusion) {}

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] dims Define this operation will be performed on which axes.
  void Init(const std::vector<int64_t> &dims);

  /// \brief Method to set dims attribute.
  ///
  /// \param[in] dims Define this operation will be performed on which axes.
  void set_dims(const std::vector<int64_t> &dims);

  /// \brief Method to get dims attribute.
  ///
  /// \return axes.
  std::vector<int64_t> get_dims() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_TILE_FUSION_H_
