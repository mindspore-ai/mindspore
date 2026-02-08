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
#include "primitive/other_op_name.h"
#include "frontend/expander/grad/grad_utils.h"
#include "ir/anf.h"

namespace mindspore::expander::bprop {
std::string GetStringFromNode(const NodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  ValuePtr value_ptr = node->BuildValue();
  MS_EXCEPTION_IF_NULL(value_ptr);
  if (!value_ptr->isa<StringImm>()) {
    MS_LOG(EXCEPTION) << "The value of node is not a string, node:" << node->ToString();
  }

  return GetValue<std::string>(value_ptr);
}

REG_BPROP_BUILDERS_BEGIN(GradCommOps)
REG_BPROP_BUILDER("AllReduce").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
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
  const auto &dout = ib->GetInput(i2);
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
  const auto &dout = ib->GetInput(i2);
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
    auto scale = ib->ValueByType(1.0 / rank_size, kFloat32);
    dx = ib->Muls(dx, scale);
  }
  return {dx};
});

REG_BPROP_BUILDER("_MirrorOperator").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  const auto &dout = ib->GetInput(i2);
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
    dx = ib->Muls(dx, ib->ValueByType(1.0 / dev_num, ib->GetDtype(dx)));
  }
  return {dx};
});

REG_BPROP_BUILDER(kInnerCommAllReduceOpName).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &op = ib->GetInput(i1);
  const auto &op_type = GetStringFromNode(op);
  const auto &group = ib->GetInput(i2);
  const auto &out = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i4);

  auto dy = dout;
  if (op_type == "prod") {
    dy = ib->Mul(dout, out);
  }

  auto dx = ib->InnerCommAllReduce(dy, op, group);
  dx->set_debug_info("grad" + dx->debug_info());
  if (op_type == "prod") {
    dx = ib->RealDiv(dx, x);
  } else if (op_type != "sum") {
    auto z = ib->Equal(x, out);
    z = ib->Cast(z, ib->GetDtype(dx));
    dx = ib->Mul(dx, z);
  }

  return {dx, ib->OutZeros(op), ib->OutZeros(group)};
});

REG_BPROP_BUILDER(kInnerCommAllGatherOpName).SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  const auto &rank_size = ib->GetInput(i1);
  const auto &group = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);

  auto dx = ib->InnerCommReduceScatter(dout, rank_size, ib->Value("sum"), group);
  auto ins_name = ib->GetInstanceName();
  dx->set_debug_info("grad" + ins_name);

  return {dx, ib->OutZeros(rank_size), ib->OutZeros(group)};
});

REG_BPROP_BUILDER(kInnerCommReduceScatterOpName).SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  const auto &rank_size = ib->GetInput(i1);
  const auto &op = ib->GetInput(i2);
  auto op_type = GetStringFromNode(op);
  const auto &group = ib->GetInput(i3);

  if (op_type != "sum") {
    MS_LOG(EXCEPTION) << "The reducescatter bprop only support ReduceOp.SUM until now.";
  }
  const auto &dout = ib->GetInput(i5);
  auto dx = ib->InnerCommAllGather(dout, rank_size, group);
  auto ins_name = ib->GetInstanceName();
  dx->set_debug_info("grad" + ins_name);
  return {dx, ib->OutZeros(rank_size), ib->OutZeros(op), ib->OutZeros(group)};
});

REG_BPROP_BUILDER(kInnerCommIRecvOpName).FreeUselessValues_IO({i2, i4}, {}).SetBody(BODYFUNC(ib) {
  const auto &tag = ib->GetInput(i0);
  const auto &rank = ib->GetInput(i1);
  const auto &shape = ib->GetInput(i2);
  const auto &group = ib->GetInput(i3);
  const auto &dtype_node = ib->GetInput(i4);
  const auto &dout = ib->GetInput(i6);

  auto out_tensor = ib->Tensor(0.0, kFloat16);

  auto send_out = ib->InnerCommIsend(dout, rank, group, tag);
  auto dx = ib->Depend(ib->Cast(out_tensor, dtype_node), send_out);

  return {
    dx, ib->OutZeros(tag), ib->OutZeros(rank), ib->OutZeros(shape), ib->OutZeros(group), ib->OutZeros(dtype_node)};
});

REG_BPROP_BUILDER(kInnerCommISendOpName).FreeUselessValues_IO({i0}, {}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &rank = ib->GetInput(i1);
  const auto &group = ib->GetInput(i2);
  const auto &tag = ib->GetInput(i3);

  auto shape = ib->GetShape(input);
  auto dtype = ib->GetDtype(input);
  auto dx = ib->InnerCommIrecv(tag, rank, ib->Value(shape), group, ib->Value(dtype));

  return {dx, ib->OutZeros(rank), ib->OutZeros(group), ib->OutZeros(tag)};
});

REG_BPROP_BUILDER(kInnerCommAllToAllVOpName).FreeUselessValues_IO({i0}, {}).SetBody(BODYFUNC(ib) {
  auto group = ib->GetInput(i1);
  auto send_numel_list = ib->GetInput(i2);
  auto recv_numel_list = ib->GetInput(i3);
  auto rank_size = ib->GetInput(i4);
  auto split_sizes_empty = ib->GetInput(i5);
  auto dout = ib->GetInput(i7);
  auto dx = ib->InnerCommAllToAllV(dout, group, recv_numel_list, send_numel_list, rank_size, split_sizes_empty);
  return {dx,
          ib->OutZeros(group),
          ib->OutZeros(send_numel_list),
          ib->OutZeros(recv_numel_list),
          ib->OutZeros(rank_size),
          ib->OutZeros(split_sizes_empty)};
});
REG_BPROP_BUILDERS_END
}  // namespace mindspore::expander::bprop
