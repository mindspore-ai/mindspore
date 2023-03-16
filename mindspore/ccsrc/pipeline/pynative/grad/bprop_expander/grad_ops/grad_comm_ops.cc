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
#include "pipeline/pynative/grad/bprop_expander/bprop_irbuilder.h"
#include "pipeline/pynative/grad/bprop_expander/grad_ops/common_utils.h"
#include "include/common/utils/utils.h"
#include "ir/anf.h"

namespace mindspore::expander::bprop {
REG_BPROP_BUILDERS_BEGIN(GradCommOps)
REG_BPROP_BUILDER("AllReduce").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
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
    auto z = ib->Equal(x, out);
    z = ib->Cast(z, ib->GetDtype(dx));
    return {ib->Mul(dx, z)};
  }
});

REG_BPROP_BUILDER("AllGather").SetUnusedInputs({i0, i1}).SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit(kReduceScatterOpName, {dout},
                     {{"op", MakeValue("sum")},
                      {"rank_size", ib->GetAttr("rank_size")},
                      {"group", ib->GetAttr("group")},
                      {"fusion", ib->GetAttr("fusion")},
                      {"no_eliminate", MakeValue(true)}});
  auto ins_name = ib->GetInstanceName();
  auto primitive = GetCNodePrimitive(dx->get());
  primitive->set_instance_name("grad" + ins_name);
  auto rank_size = GetValue<int64_t>(ib->GetAttr("rank_size"));
  if (rank_size == 0) {
    MS_LOG(EXCEPTION) << "The 'rank_size' can not be zero, but got" << rank_size;
  }
  auto mean_flag = GetValue<bool>(ib->GetAttr("mean_flag"));
  if (mean_flag) {
    auto scale = ib->Tensor(1.0 / rank_size, kFloat32);
    dx = ib->Mul(dx, scale);
  }
  return {dx};
});

REG_BPROP_BUILDER("_MirrorOperator").SetUnusedInputs({i0, i1}).SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto dout = ib->GetInput(kIndex2);
  auto dev_num = GetValue<int64_t>(ib->GetAttr("dev_num"));
  bool mean_flag = GetValue<bool>(ib->GetAttr("mean_flag"));
  if (dev_num == 1) {
    return {dout};
  }
  DAttr attrs{{"op", MakeValue("sum")},
              {"group", ib->GetAttr("group")},
              {"fusion", ib->GetAttr("fusion")},
              {"no_eliminate", MakeValue(true)}};
  if (ib->GetAttr("parameter") != nullptr) {
    attrs.insert({"parameter", ib->GetAttr("parameter")});
  }
  auto dx = ib->Emit(kAllReduceOpName, {dout}, attrs);
  auto ins_name = ib->GetInstanceName();
  auto primitive = GetCNodePrimitive(dx->get());
  primitive->set_instance_name("grad_mirror" + ins_name);
  if (mean_flag) {
    auto float_one = ib->Emit("scalar_cast", {ib->Value(1.0), ib->EmitValue(ib->GetDtype(dx))});
    auto num = ib->Emit("scalar_cast", {ib->Value(dev_num), ib->EmitValue(ib->GetDtype(dx))});
    dx = ib->Mul(dx, ib->Cast(float_one / num, ib->GetDtype(dx)));
  }
  return {dx};
});
REG_BPROP_BUILDERS_END
}  // namespace mindspore::expander::bprop
