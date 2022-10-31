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
#include "include/common/utils/utils.h"

namespace mindspore::expander::bprop {
REG_BPROP_BUILDER("BNTrainingReduce").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  return {ib->ZerosLike(x)};
});

REG_BPROP_BUILDER("MinMaxUpdatePerLayer").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto x_min = ib->GetInput(kIndex1);
  auto x_max = ib->GetInput(kIndex2);
  return {ib->ZerosLike(x), ib->ZerosLike(x_min), ib->ZerosLike(x_max)};
});

REG_BPROP_BUILDER("MinMaxUpdatePerChannel").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto x_min = ib->GetInput(kIndex1);
  auto x_max = ib->GetInput(kIndex2);
  return {ib->ZerosLike(x), ib->ZerosLike(x_min), ib->ZerosLike(x_max)};
});

REG_BPROP_BUILDER("WtsARQ").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto w_min = ib->GetInput(kIndex1);
  auto w_max = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  return {dout, ib->ZerosLike(w_min), ib->ZerosLike(w_max)};
});
}  // namespace mindspore::expander::bprop
