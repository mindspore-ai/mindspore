/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
REG_ADPT_DESC(ScalarSummary, prim::kPrimScalarSummary->name(), ADPT_DESC(Summary))
REG_ADPT_DESC(ImageSummary, prim::kPrimImageSummary->name(), ADPT_DESC(Summary))
REG_ADPT_DESC(TensorSummary, prim::kPrimTensorSummary->name(), ADPT_DESC(Summary))
REG_ADPT_DESC(HistogramSummary, prim::kPrimHistogramSummary->name(), ADPT_DESC(Summary))
REG_ADPT_DESC(Debug, prim::kPrimDebug->name(), ADPT_DESC(Summary))

// Data
INPUT_MAP(Data) = EMPTY_INPUT_MAP;
ATTR_MAP(Data) = EMPTY_ATTR_MAP;
REG_ADPT_DESC(Data, kNameParam, ADPT_DESC(Data))

// Reshape
INPUT_MAP(Reshape) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(shape)}};
ATTR_MAP(Reshape) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Reshape) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Reshape, kNameReshape, ADPT_DESC(Reshape))
REG_ADPT_DESC(FlattenGrad, kNameFlattenGrad, ADPT_DESC(Reshape))

// TransShape
INPUT_MAP(TransShape) = {{1, INPUT_DESC(x)}};
INPUT_ATTR_MAP(TransShape) = {{2, ATTR_DESC(outShape, AnyTraits<int>(), AnyTraits<std::vector<int64_t>>())}};
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

// ExpandDims
INPUT_MAP(ExpandDims) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(axis)}};
ATTR_MAP(ExpandDims) = EMPTY_ATTR_MAP;
OUTPUT_MAP(ExpandDims) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ExpandDims, kNameExpandDims, ADPT_DESC(ExpandDims))

// Squeeze
INPUT_MAP(Squeeze) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Squeeze) = {{"axis", ATTR_DESC(axis, AnyTraits<int>(), AnyTraits<std::vector<int64_t>>())}};
OUTPUT_MAP(Squeeze) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Squeeze, prim::kPrimSqueeze->name(), ADPT_DESC(Squeeze))

// ReverseSequence
INPUT_MAP(ReverseSequence) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(seq_lengths)}};
ATTR_MAP(ReverseSequence) = {{"seq_dim", ATTR_DESC(seq_dim, AnyTraits<int>())},
                             {"batch_dim", ATTR_DESC(batch_dim, AnyTraits<int>())}};
OUTPUT_MAP(ReverseSequence) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ReverseSequence, kNameReverseSequence, ADPT_DESC(ReverseSequence))
}  // namespace mindspore::transform
