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
#include "ir/anf.h"

namespace mindspore::expander::bprop {
REG_BPROP_BUILDER(kAllReduceOpName).SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto op = GetValue<std::string>(ib->GetAttr("op"));
  auto dy = dout;
  if (op == "prod") {
    dy = ib->Mul(dout, out);
  }
  auto dx = ib->Emit(kAllReduceOpName, {dy},
                     {{"op", MakeValue("sum")},
                      {"group", ib->GetAttr("group")},
                      {"index", ib->GetAttr("index")},
                      {"fusion", ib->GetAttr("fusion")},
                      {"no_eliminate", ib->GetAttr("no_eliminate")}});
  auto primitive = GetCNodePrimitive(dx->get());
  MS_EXCEPTION_IF_NULL(primitive);
  auto ins_name = primitive->instance_name();
  primitive->set_instance_name("grad" + ins_name);
  if (op == "prod") {
    return {ib->RealDiv(dx, x)};
  } else if (op == "sum") {
    return {dx};
  } else {
    auto z = ib->Emit("Equal", {x, out});
    z = ib->Cast(z, ib->GetDtype(dx));
    return {ib->Mul(dx, z)};
  }
});
}  // namespace mindspore::expander::bprop
