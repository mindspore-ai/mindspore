/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "frontend/expander/bprop/grad_ops/common_utils.h"
#include "include/common/utils/utils.h"

namespace mindspore::expander::bprop {
NodePtrList SequenceToTensorGrad(BpropIRBuilder *ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex3);
  dout = ib->Cast(dout, ib->GetDtype(x));
  auto dx = ib->TensorToSequence(dout, x->abstract());
  return {dx, ib->OutZeros(ib->GetInput(kIndex1))};
}

NodePtrList TensorToSequenceGrad(BpropIRBuilder *ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->SequenceToTensor(dout, ib->GetDtype(x));
  return {dx};
}

NodePtrList SequenceSetItemGrad(BpropIRBuilder *ib) {
  auto idx = ib->GetInput(kIndex1);
  auto value = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto dx = ib->SequenceSetItem(dout, idx, ib->ZerosLike(value));
  auto dvalue = ib->TupleGetItem(dout, idx);
  return {dx, ib->OutZeros(idx), dvalue};
}

NodePtrList SequenceMaxMinGrad(BpropIRBuilder *ib) {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto index = ib->Emit("SequenceIndex", {x, out});
  auto dx = ib->SequenceSetItem(ib->ZerosLike(x), index, dout);
  return {dx};
}
REG_BPROP_BUILDERS_BEGIN(GradSequenceOps)
REG_BPROP_BUILDER("make_range").SetBody(ReturnZeros);
REG_BPROP_BUILDER("SequenceCount").SetBody(ReturnZeros);
REG_BPROP_BUILDER("sequence_len").SetBody(ReturnZeros);

REG_BPROP_BUILDER("SequenceAdd").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto out_offset = ib->Emit("SequenceAddOffset", {x, y});
  auto dx = ib->SequenceSlice(dout, ib->TupleGetItem(out_offset, 0), ib->Len(x), ib->Value<int64_t>(1));
  auto dy = ib->SequenceSlice(dout, ib->TupleGetItem(out_offset, 1), ib->ScalarAdd(ib->Len(x), ib->Len(y)),
                              ib->Value<int64_t>(1));
  return {dx, dy};
});

REG_BPROP_BUILDER("SequenceUnstack").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("SequenceStack", {dout}, {{"axis", ib->GetAttr("axis")}});
  return {dx};
});

REG_BPROP_BUILDER("SequenceSlice").SetUnusedInputs({i4}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto start = ib->GetInput(kIndex1);
  auto stop = ib->GetInput(kIndex2);
  auto step = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex5);
  auto dx = ib->Emit("SequenceSliceGrad", {dout, x, start, stop, step});
  return {dx, ib->OutZeros(start), ib->OutZeros(stop), ib->OutZeros(step)};
});

REG_BPROP_BUILDER("SequenceIndex").SetBody(ReturnZeros);
REG_BPROP_BUILDER("InSequence").SetBody(ReturnZeros);
REG_BPROP_BUILDER("tuple_equal").SetBody(ReturnZeros);
REG_BPROP_BUILDER("list_equal").SetBody(ReturnZeros);
REG_BPROP_BUILDER("shape_mul").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("ShapeMulGrad", {x, dout});
  return {dx};
});

REG_BPROP_BUILDER("tuple_setitem").SetUnusedInputs({i0, i3}).SetBody(SequenceSetItemGrad);
REG_BPROP_BUILDER("list_setitem").SetUnusedInputs({i0, i3}).SetBody(SequenceSetItemGrad);
REG_BPROP_BUILDER("ListInplaceReverse").SetBody(ReturnZeros);
REG_BPROP_BUILDER("ListInplaceExtend").SetBody(ReturnZeros);
REG_BPROP_BUILDER("ListInplaceInsert").SetBody(ReturnZeros);
REG_BPROP_BUILDER("ListInplacePop").SetBody(ReturnZeros);

REG_BPROP_BUILDER("ListAppend").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  auto value = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Emit("ListAppendAndInsertGrad", {dout, ib->Value<int64_t>(-1)});
  return {dx, ib->OutZeros(value)};
});

REG_BPROP_BUILDER("ListInsert").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  auto idx = ib->GetInput(kIndex1);
  auto value = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto dx = ib->Emit("ListAppendAndInsertGrad", {dout, idx});
  return {dx, ib->OutZeros(idx), ib->OutZeros(value)};
});

REG_BPROP_BUILDER("TupleToTensor").SetUnusedInputs({i0, i1, i2}).SetBody(SequenceToTensorGrad);
REG_BPROP_BUILDER("ListToTensor").SetUnusedInputs({i0, i1, i2}).SetBody(SequenceToTensorGrad);
REG_BPROP_BUILDER("TensorToTuple").SetUnusedInputs({i0, i1, i2}).SetBody(TensorToSequenceGrad);
REG_BPROP_BUILDER("ListToTensor").SetUnusedInputs({i0, i1, i2}).SetBody(TensorToSequenceGrad);

REG_BPROP_BUILDER("ScalarToTensor").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex3);
  dout = ib->Cast(dout, ib->GetDtype(x));
  auto dx = ib->Emit("TensorToScalar", {dout});
  return {dx, ib->OutZeros(ib->GetInput(kIndex1))};
});

REG_BPROP_BUILDER("TensorToScalar").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Emit("ScalarToTensor", {dout, ib->Value(ib->GetDtype(x))});
  return {dx};
});

REG_BPROP_BUILDER("SequenceMul").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->SequenceSlice(dout, ib->Value<int64_t>(0LL), ib->Len(x), ib->Value<int64_t>(1));
  return {dx, ib->OutZeros(y)};
});

REG_BPROP_BUILDER("SequenceMax").SetBody(SequenceMaxMinGrad);
REG_BPROP_BUILDER("SequenceMin").SetBody(SequenceMaxMinGrad);
REG_BPROP_BUILDER("tuple_le").SetBody(ReturnZeros);
REG_BPROP_BUILDER("tuple_lt").SetBody(ReturnZeros);
REG_BPROP_BUILDER("list_le").SetBody(ReturnZeros);
REG_BPROP_BUILDER("list_lt").SetBody(ReturnZeros);
REG_BPROP_BUILDER("tuple_greater_than").SetBody(ReturnZeros);
REG_BPROP_BUILDER("list_greater_than").SetBody(ReturnZeros);
REG_BPROP_BUILDER("tuple_greater_equal").SetBody(ReturnZeros);
REG_BPROP_BUILDER("list_greater_equal").SetBody(ReturnZeros);
REG_BPROP_BUILDERS_END
}  // namespace mindspore::expander::bprop
