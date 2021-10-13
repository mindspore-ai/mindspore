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
#ifndef MINDSPORE_CORE_OPS_RESIZE_NEAREST_NEIGHBOR_H_
#define MINDSPORE_CORE_OPS_RESIZE_NEAREST_NEIGHBOR_H_

#include <vector>
#include <memory>

#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameResizeNearestNeighbor = "ResizeNearestNeighbor";
/// \brief Resizes the input tensor by using the nearest neighbor algorithm.
/// Refer to Python API @ref mindspore.ops.ResizeNearestNeighbor for more details.
class MS_CORE_API ResizeNearestNeighbor : public PrimitiveC {
 public:
  /// \brief Constructor.
  ResizeNearestNeighbor() : PrimitiveC(kNameResizeNearestNeighbor) {}
  /// \brief Destructor.
  ~ResizeNearestNeighbor() = default;
  MS_DECLARE_PARENT(ResizeNearestNeighbor, PrimitiveC);
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.ResizeNearestNeighbor for the inputs.
  void Init(const std::vector<int64_t> &size, const bool align_corners = false);
  /// \brief Set size.
  void set_size(const std::vector<int64_t> &size);
  /// \brief Set align_corners.
  void set_align_corners(const bool align_corners);
  /// \brief Get size.
  ///
  /// \return size.
  std::vector<int64_t> get_size() const;
  /// \brief Get align_corners.
  ///
  /// \return align_corners.
  bool get_align_corners() const;
};

using PrimResizeNearestNeighborPtr = std::shared_ptr<ResizeNearestNeighbor>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_RESIZE_NEAREST_NEIGHBOR_H_
