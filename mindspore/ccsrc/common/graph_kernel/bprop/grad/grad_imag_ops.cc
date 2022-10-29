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
#include "common/graph_kernel/bprop/bprop_irbuilder.h"
#include "common/graph_kernel/bprop/expander/common_utils.h"
#include "include/common/utils/utils.h"

namespace mindspore::expander::bprop {
REG_BPROP_BUILDER("ResizeBicubic").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto images = ib->GetInput(kIndex0);
  auto size = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto images_type = ib->GetDtype(images);
  std::set<TypePtr> type_list = {kInt8, kInt16, kInt32, kInt64, kUInt8, kUInt16, kFloat16};
  if (type_list.count(images_type) == 0) {
    images = ib->Cast(images, kFloat64);
  }
  auto dx = ib->Emit(
    "ResizeBicubicGrad", {dout, images},
    {{"align_corners", ib->GetAttr("align_corners")}, {"half_pixel_centers", ib->GetAttr("half_pixel_centers")}});
  return {dx, ib->Emit("ZerosLike", {size})};
});
}  // namespace mindspore::expander::bprop
