/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include <memory>
#include "plugin/device/ascend/kernel/internal/tile.h"
namespace mindspore {
namespace kernel {
internal::OpParamPtr InternalTile::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                 const std::vector<KernelTensor *> &outputs) {
  internal::OpParamPtr param_ptr = std::make_shared<internal::OpParam>();
  internal::ExpandParam op_param;

  auto dims = inputs.at(kIndex1);
  auto dims_list = dims->GetValue<std::vector<int64_t>>().value();
  auto shape_in = inputs[kIndex0]->GetShapeVector();
  ShapeVector shape_out;
  for (size_t i = 0; i < dims_list.size(); ++i) {
    shape_out.emplace_back(dims_list[i] * shape_in[i]);
  }
  for (auto shape : shape_out) {
    op_param.shape.emplace_back(shape);
  }

  param_ptr->specificParam = op_param;
  param_ptr->opId = internal::OpId::Tile;
  return param_ptr;
}

void InternalTile::SetInOutIdx() {
  inputsIdxMap_[kIndex0] = kIndex0;
  outputsIdxMap_[kIndex0] = kIndex0;
}

MS_INTERNAL_KERNEL_FACTORY_REG(Tile, InternalTile);
}  // namespace kernel
}  // namespace mindspore