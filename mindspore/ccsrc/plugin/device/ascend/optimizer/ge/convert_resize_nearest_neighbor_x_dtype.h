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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_GE_CONVERT_RESIZE_NEAREST_NEIGHBOR_X_DTYPE_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_GE_CONVERT_RESIZE_NEAREST_NEIGHBOR_X_DTYPE_H_

#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class ConvertResizeNearestNeighborXDtype : public PatternProcessPass {
 public:
  explicit ConvertResizeNearestNeighborXDtype(bool multigraph = true)
      : PatternProcessPass("convert_resize_nearest_neighbor_x_dtype", multigraph) {}
  ~ConvertResizeNearestNeighborXDtype() override = default;

  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &node, const EquivPtr &) const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_GE_CONVERT_RESIZE_NEAREST_NEIGHBOR_X_DTYPE_H_
