/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "transform/graph_ir/op_declare/array_ops_declare.h"
#include <vector>
#include <string>

namespace mindspore::transform {
// const
INPUT_MAP(Const) = EMPTY_INPUT_MAP;
ATTR_MAP(Const) = {{"value", ATTR_DESC(value, AnyTraits<AnyValue>())}};
OUTPUT_MAP(Const) = {{0, OUTPUT_DESC(y)}};

// Constant
INPUT_MAP(Constant) = EMPTY_INPUT_MAP;
ATTR_MAP(Constant) = {{"value", ATTR_DESC(value, AnyTraits<AnyValue>())}};
OUTPUT_MAP(Constant) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Constant, kNameConst, ADPT_DESC(Constant, Const))

// ScalarSummary
INPUT_MAP(Summary) = {{2, INPUT_DESC(x)}};
ATTR_MAP(Summary) = EMPTY_ATTR_MAP;
#ifndef ENABLE_SECURITY
REG_ADPT_DESC(ScalarSummary, prim::kPrimScalarSummary->name(), ADPT_DESC(Summary))
REG_ADPT_DESC(ImageSummary, prim::kPrimImageSummary->name(), ADPT_DESC(Summary))
REG_ADPT_DESC(TensorSummary, prim::kPrimTensorSummary->name(), ADPT_DESC(Summary))
REG_ADPT_DESC(HistogramSummary, prim::kPrimHistogramSummary->name(), ADPT_DESC(Summary))
#endif
REG_ADPT_DESC(Debug, prim::kPrimDebug->name(), ADPT_DESC(Summary))

// Data
INPUT_MAP(Data) = EMPTY_INPUT_MAP;
ATTR_MAP(Data) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Data) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Data, kNameParam, ADPT_DESC(Data))

// Shape
INPUT_MAP(Shape) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Shape) = {{"dtype", ATTR_DESC(dtype, AnyTraits<int64_t>())}};
OUTPUT_MAP(Shape) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Shape, kNameShape, ADPT_DESC(Shape))

// TensorShape
REG_ADPT_DESC(TensorShape, kNameTensorShape, ADPT_DESC(Shape))

// DynamicShape
REG_ADPT_DESC(DynamicShape, kNameDynamicShape, ADPT_DESC(Shape))

// GetShape
INPUT_MAP(GetShape) = EMPTY_INPUT_MAP;
DYN_INPUT_MAP(GetShape) = {{1, DYN_INPUT_DESC(x)}};
ATTR_MAP(GetShape) = EMPTY_ATTR_MAP;
OUTPUT_MAP(GetShape) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(GetShape, kNameGetShape, ADPT_DESC(GetShape));

// Reshape
INPUT_MAP(Reshape) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(shape)}};
ATTR_MAP(Reshape) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Reshape) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Reshape, kNameReshape, ADPT_DESC(Reshape))
REG_ADPT_DESC(FlattenGrad, kNameFlattenGrad, ADPT_DESC(Reshape))

// TransShape
INPUT_MAP(TransShape) = {{1, INPUT_DESC(x)}};
INPUT_ATTR_MAP(TransShape) = {{2, ATTR_DESC(outShape, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())}};
ATTR_MAP(TransShape) = EMPTY_ATTR_MAP;
OUTPUT_MAP(TransShape) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(TransShape, kNameTransShape, ADPT_DESC(TransShape))

// MirrorPad
INPUT_MAP(MirrorPad) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(paddings)}};
ATTR_MAP(MirrorPad) = {{"mode", ATTR_DESC(mode, AnyTraits<std::string>())}};
OUTPUT_MAP(MirrorPad) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MirrorPad, kNameMirrorPad, ADPT_DESC(MirrorPad))

// MirrorPadGrad
INPUT_MAP(MirrorPadGrad) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(paddings)}};
ATTR_MAP(MirrorPadGrad) = {{"mode", ATTR_DESC(mode, AnyTraits<std::string>())}};
OUTPUT_MAP(MirrorPadGrad) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MirrorPadGrad, kNameMirrorPadGrad, ADPT_DESC(MirrorPadGrad))

// Expand
INPUT_MAP(Expand) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(shape)}};
ATTR_MAP(Expand) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Expand) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Expand, "Expand", ADPT_DESC(Expand))

// ExpandDims
INPUT_MAP(ExpandDims) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(axis)}};
ATTR_MAP(ExpandDims) = EMPTY_ATTR_MAP;
OUTPUT_MAP(ExpandDims) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ExpandDims, kNameExpandDims, ADPT_DESC(ExpandDims))

// Squeeze
INPUT_MAP(Squeeze) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Squeeze) = {{"axis", ATTR_DESC(axis, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())}};
OUTPUT_MAP(Squeeze) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Squeeze, prim::kPrimSqueeze->name(), ADPT_DESC(Squeeze))

INPUT_MAP(SqueezeV3) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(axes)}};
ATTR_MAP(SqueezeV3) = EMPTY_ATTR_MAP;
OUTPUT_MAP(SqueezeV3) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(SqueezeV3, prim::kPrimSqueezeV3->name(), ADPT_DESC(SqueezeV3))

// ReverseSequence
INPUT_MAP(ReverseSequence) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(seq_lengths)}};
ATTR_MAP(ReverseSequence) = {{"seq_dim", ATTR_DESC(seq_dim, AnyTraits<int64_t>())},
                             {"batch_dim", ATTR_DESC(batch_dim, AnyTraits<int64_t>())}};
OUTPUT_MAP(ReverseSequence) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ReverseSequence, kNameReverseSequence, ADPT_DESC(ReverseSequence))

// EditDistance
INPUT_MAP(EditDistance) = {{1, INPUT_DESC(hypothesis_indices)}, {2, INPUT_DESC(hypothesis_values)},
                           {3, INPUT_DESC(hypothesis_shape)},   {4, INPUT_DESC(truth_indices)},
                           {5, INPUT_DESC(truth_values)},       {6, INPUT_DESC(truth_shape)}};
ATTR_MAP(EditDistance) = {{"normalize", ATTR_DESC(normalize, AnyTraits<bool>())}};
OUTPUT_MAP(EditDistance) = {{0, OUTPUT_DESC(output)}};
REG_ADPT_DESC(EditDistance, kNameEditDistance, ADPT_DESC(EditDistance))

// NonZeroWithValue
INPUT_MAP(NonZeroWithValue) = {{1, INPUT_DESC(x)}};
ATTR_MAP(NonZeroWithValue) = {{"transpose", ATTR_DESC(transpose, AnyTraits<bool>())}};
OUTPUT_MAP(NonZeroWithValue) = {{0, OUTPUT_DESC(value)}, {1, OUTPUT_DESC(index)}, {2, OUTPUT_DESC(count)}};
REG_ADPT_DESC(NonZeroWithValue, kNameNonZeroWithValue, ADPT_DESC(NonZeroWithValue))

// NonZeroWithValueShape
INPUT_MAP(NonZeroWithValueShape) = {{1, INPUT_DESC(value)}, {2, INPUT_DESC(index)}, {3, INPUT_DESC(count)}};
ATTR_MAP(NonZeroWithValueShape) = EMPTY_ATTR_MAP;
OUTPUT_MAP(NonZeroWithValueShape) = {{0, OUTPUT_DESC(out_value)}, {1, OUTPUT_DESC(out_index)}};
REG_ADPT_DESC(NonZeroWithValueShape, kNameNonZeroWithValueShape, ADPT_DESC(NonZeroWithValueShape))

// Unsqueeze
INPUT_MAP(Unsqueeze) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Unsqueeze) = {{"axis", ATTR_DESC(axes, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())}};
OUTPUT_MAP(Unsqueeze) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Unsqueeze, kNameUnsqueeze, ADPT_DESC(Unsqueeze))

// Identity
INPUT_MAP(Identity) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Identity) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Identity) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(IdentityLoad, kNameLoad, ADPT_DESC(Identity))
REG_ADPT_DESC(IdentityTupleGetItem, kNameTupleGetItem, ADPT_DESC(Identity))
REG_ADPT_DESC(IdentityIdentity, kNameIdentity, ADPT_DESC(Identity))

// IdentityN
INPUT_MAP(IdentityN) = EMPTY_INPUT_MAP;
DYN_INPUT_MAP(IdentityN) = {{1, DYN_INPUT_DESC(x)}};
ATTR_MAP(IdentityN) = EMPTY_ATTR_MAP;
DYN_OUTPUT_MAP(IdentityN) = {{0, DYN_OUTPUT_DESC(y)}};
REG_ADPT_DESC(IdentityNMakeTuple, kNameMakeTuple, ADPT_DESC(IdentityN))
REG_ADPT_DESC(IdentityNDepend, kNameDepend, ADPT_DESC(IdentityN))
REG_ADPT_DESC(IdentityNReturn, kNameReturn, ADPT_DESC(IdentityN))

// Where
INPUT_MAP(SelectV2) = {{1, INPUT_DESC(condition)}, {2, INPUT_DESC(then)}, {3, INPUT_DESC(else)}};
ATTR_MAP(SelectV2) = EMPTY_ATTR_MAP;
OUTPUT_MAP(SelectV2) = {{0, OUTPUT_DESC(result)}};
REG_ADPT_DESC(SelectV2, kNameWhere, ADPT_DESC(SelectV2))

// Unique
INPUT_MAP(Unique) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Unique) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Unique) = {{0, OUTPUT_DESC(y)}, {1, OUTPUT_DESC(idx)}};
REG_ADPT_DESC(Unique, kNameUnique, ADPT_DESC(Unique))

// BroadcastGradientArgs
INPUT_MAP(BroadcastGradientArgs) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(BroadcastGradientArgs) = EMPTY_ATTR_MAP;
OUTPUT_MAP(BroadcastGradientArgs) = {{0, OUTPUT_DESC(y1)}, {1, OUTPUT_DESC(y2)}};
REG_ADPT_DESC(BroadcastGradientArgs, kNameDynamicBroadcastGradientArgs, ADPT_DESC(BroadcastGradientArgs))
}  // namespace mindspore::transform
