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
#include "frontend/expander/bprop/bprop_irbuilder.h"
#include "ops/other_op_name.h"
#include "frontend/expander/bprop/grad_ops/common_utils.h"
#include "include/common/utils/utils.h"
#include "ir/anf.h"

namespace mindspore::expander::bprop {
REG_BPROP_BUILDERS_BEGIN(GradCommOps)
REG_BPROP_BUILDER("AllReduce").SetBody(BODYFUNC(ib) {
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
  dx->set_debug_info("grad" + dx->debug_info());
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

REG_BPROP_BUILDER("NeighborExchange").SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("NeighborExchange", {dout},
                     {{"send_rank_ids", ib->GetAttr("recv_rank_ids")},
                      {"recv_rank_ids", ib->GetAttr("send_rank_ids")},
                      {"recv_shapes", ib->GetAttr("send_shapes")},
                      {"send_shapes", ib->GetAttr("recv_shapes")},
                      {"recv_type", ib->GetAttr("recv_type")},
                      {"group", ib->GetAttr("group")}});
  auto ins_name = ib->GetInstanceName();
  dx->set_debug_info("grad" + ins_name);
  return {dx};
});

REG_BPROP_BUILDER("AllGather").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit(kReduceScatterOpName, {dout},
                     {{"op", MakeValue("sum")},
                      {"rank_size", ib->GetAttr("rank_size")},
                      {"group", ib->GetAttr("group")},
                      {"fusion", ib->GetAttr("fusion")},
                      {"no_eliminate", MakeValue(true)}});
  auto ins_name = ib->GetInstanceName();
  dx->set_debug_info("grad" + ins_name);
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

REG_BPROP_BUILDER("_MirrorOperator").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
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
    (void)attrs.emplace_back("parameter", ib->GetAttr("parameter"));
  }
  auto dx = ib->Emit(kAllReduceOpName, {dout}, attrs);
  auto ins_name = ib->GetInstanceName();
  dx->set_debug_info("grad_mirror" + ins_name);
  if (mean_flag) {
    dx = ib->Mul(dx, ib->Tensor(1.0 / dev_num, ib->GetDtype(dx)));
  }
  return {dx};
});
REG_BPROP_BUILDERS_END
}  // namespace mindspore::expander::bprop
