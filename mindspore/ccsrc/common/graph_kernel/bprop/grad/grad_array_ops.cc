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
REG_BPROP_BUILDER("Flatten").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Reshape(dout, ib->GetShape(x));
  return {dx};
});

REG_BPROP_BUILDER(kReshapeOpName).SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto shp = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto shapex = ib->GetShape(x);
  return {ib->Reshape(dout, shapex), ib->ZerosLike(shp)};
});

REG_BPROP_BUILDER("NonZero").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  return {ib->ZerosLike(x)};
});

REG_BPROP_BUILDER("BatchMatMul").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto ta = GetValue<bool>(ib->GetAttr("transpose_a"));
  auto tb = GetValue<bool>(ib->GetAttr("transpose_b"));
  auto x = ib->GetInput(kIndex0);
  auto w = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);

  NodePtr dx;
  if (ta) {
    dx =
      ib->Emit("BatchMatMul", {w, dout}, {{"transpose_a", MakeValue(ta && tb)}, {"transpose_b", MakeValue(ta || !tb)}});
  } else {
    dx =
      ib->Emit("BatchMatMul", {dout, w}, {{"transpose_a", MakeValue(ta && tb)}, {"transpose_b", MakeValue(ta || !tb)}});
  }

  NodePtr dw;
  if (tb) {
    dw = ib->Emit("BatchMatMul", {dout, x},
                  {{"transpose_a", MakeValue((!ta) || tb)}, {"transpose_b", MakeValue(ta && tb)}});
  } else {
    dw = ib->Emit("BatchMatMul", {x, dout},
                  {{"transpose_a", MakeValue((!ta) || tb)}, {"transpose_b", MakeValue(ta && tb)}});
  }

  return {BinopGradCommonWithShift(ib, x, w, dx, dw, 2)};
});

REG_BPROP_BUILDER("Argmax").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  return {ib->ZerosLike(x)};
});

REG_BPROP_BUILDER("Argmin").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  return {ib->ZerosLike(x)};
});

REG_BPROP_BUILDER("Diag").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto dout = ib->GetInput(kIndex2);
  return {ib->Emit("DiagPart", {dout})};
});

REG_BPROP_BUILDER("DiagPart").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto dout = ib->GetInput(kIndex2);
  return {ib->Emit("Diag", {dout})};
});

REG_BPROP_BUILDER("SpaceToBatch").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto dout = ib->GetInput(kIndex2);
  auto dx =
    ib->Emit("BatchToSpace", {dout}, {{"block_size", ib->GetAttr("block_size")}, {"crops", ib->GetAttr("paddings")}});
  return {dx};
});

REG_BPROP_BUILDER("BatchToSpace").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto dout = ib->GetInput(kIndex2);
  auto dx =
    ib->Emit("SpaceToBatch", {dout}, {{"block_size", ib->GetAttr("block_size")}, {"paddings", ib->GetAttr("crops")}});
  return {dx};
});

REG_BPROP_BUILDER("ReverseSequence").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto seq_lengths = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Emit("ReverseSequence", {dout, seq_lengths},
                     {{"batch_dim", ib->GetAttr("batch_dim")}, {"seq_dim", ib->GetAttr("seq_dim")}});
  return {dx, ib->ZerosLike(seq_lengths)};
});

REG_BPROP_BUILDER("TensorScatterAdd").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto indices = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex4);
  auto update_grad = ib->Emit("GatherNd", {dout, indices});
  return {dout, ib->ZerosLike(indices), update_grad};
});

REG_BPROP_BUILDER(kConcatOpName).SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto axis = ib->GetAttr<int64_t>(kAttrAxis);
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto input_shapes = ib->GetShapes(x);
  if (input_shapes.empty()) {
    MS_EXCEPTION(ValueError) << "For 'ConcatOffset', 'x' can not be empty";
  }
  // axis
  auto rank = input_shapes[0].size();
  auto rank_i = SizeToLong(rank);
  if (rank == 0 || axis < -rank_i || axis >= rank_i) {
    MS_EXCEPTION(ValueError) << "For 'ConcatOffset', input shapes rank can not be 0 and 'axis' must be in range [-"
                             << rank_i << ", " << rank_i << "), but got " << axis;
  }
  if (axis < 0) {
    axis += rank_i;
  }
  auto axis_s = LongToSize(axis);
  // is_uniform
  bool is_uniform = true;
  for (size_t i = 0; i < input_shapes.size(); ++i) {
    if (input_shapes[i].size() != rank) {
      MS_EXCEPTION(ValueError) << "For 'ConcatOffset', input shapes [" << i
                               << "] and input shapes [0] must have same rank, but got: " << input_shapes[i].size()
                               << " vs " << rank;
    }
    if (input_shapes[i][axis_s] != input_shapes[0][axis_s]) {
      is_uniform = false;
    }
  }
  // use Split if is_uniform is true
  if (is_uniform) {
    auto input_nums = SizeToLong(input_shapes.size());
    auto dx = ib->Emit(kSplitOpName, {dout}, {{kAttrAxis, MakeValue(axis)}, {kAttrOutputNum, MakeValue(input_nums)}});
    return {dx};
  }
  // else use Slice
  NodePtrList res;
  int64_t sum_axis = 0;
  for (size_t i = 0; i < input_shapes.size(); ++i) {
    std::vector<int64_t> offset(rank, 0);
    offset[axis_s] = sum_axis;
    sum_axis += input_shapes[i][axis_s];
    auto slice_out = ib->Emit(kSliceOpName, {dout, ib->Value(offset), ib->Value(input_shapes[i])});
    res.push_back(slice_out);
  }
  return {ib->MakeTuple(res)};
});
}  // namespace mindspore::expander::bprop
