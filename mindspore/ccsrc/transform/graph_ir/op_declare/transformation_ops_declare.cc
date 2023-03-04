/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "transform/graph_ir/op_declare/transformation_ops_declare.h"
#include <vector>
#include <string>

namespace mindspore::transform {
// Flatten
INPUT_MAP(Flatten) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Flatten) = {{"axis", ATTR_DESC(axis, AnyTraits<int64_t>())}};
OUTPUT_MAP(Flatten) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Flatten, prim::kPrimFlatten->name(), ADPT_DESC(Flatten))

// Unpack
INPUT_MAP(Unpack) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Unpack) = {{"axis", ATTR_DESC(axis, AnyTraits<int64_t>())}, {"num", ATTR_DESC(num, AnyTraits<int64_t>())}};
DYN_OUTPUT_MAP(Unpack) = {{0, DYN_OUTPUT_DESC(y)}};
REG_ADPT_DESC(Unpack, prim::kUnstack, ADPT_DESC(Unpack))

// ExtractImagePatches
INPUT_MAP(ExtractImagePatches) = {{1, INPUT_DESC(x)}};
ATTR_MAP(ExtractImagePatches) = {
  {"ksizes", ATTR_DESC(ksizes, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
  {"strides", ATTR_DESC(strides, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
  {"rates", ATTR_DESC(rates, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
  {"padding", ATTR_DESC(padding, AnyTraits<std::string>())}};
OUTPUT_MAP(ExtractImagePatches) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ExtractImagePatches, kNameExtractImagePatches, ADPT_DESC(ExtractImagePatches))

// Transpose
INPUT_MAP(TransposeD) = {{1, INPUT_DESC(x)}};
INPUT_ATTR_MAP(TransposeD) = {{2, ATTR_DESC(perm, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())}};
ATTR_MAP(TransposeD) = EMPTY_ATTR_MAP;
// Do not set Transpose operator output descriptor
REG_ADPT_DESC(TransposeD, prim::kPrimTranspose->name(), ADPT_DESC(TransposeD))

// Transpose
INPUT_MAP(Transpose) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(perm)}};
ATTR_MAP(Transpose) = EMPTY_ATTR_MAP;
// Do not set Transpose operator output descriptor
REG_ADPT_DESC(Transpose, "TransposeV1", ADPT_DESC(Transpose))

// SpaceToDepth
INPUT_MAP(SpaceToDepth) = {{1, INPUT_DESC(x)}};
ATTR_MAP(SpaceToDepth) = {{"block_size", ATTR_DESC(block_size, AnyTraits<int64_t>())}};
OUTPUT_MAP(SpaceToDepth) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(SpaceToDepth, kNameSpaceToDepth, ADPT_DESC(SpaceToDepth))

// DepthToSpace
INPUT_MAP(DepthToSpace) = {{1, INPUT_DESC(x)}};
ATTR_MAP(DepthToSpace) = {{"block_size", ATTR_DESC(block_size, AnyTraits<int64_t>())}};
OUTPUT_MAP(DepthToSpace) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(DepthToSpace, kNameDepthToSpace, ADPT_DESC(DepthToSpace))

// SpaceToBatchD
INPUT_MAP(SpaceToBatchD) = {{1, INPUT_DESC(x)}};
ATTR_MAP(SpaceToBatchD) = {
  {"block_size", ATTR_DESC(block_size, AnyTraits<int64_t>())},
  {"paddings", ATTR_DESC(paddings, AnyTraits<std::vector<std::vector<int64_t>>>(), AnyTraits<std::vector<int64_t>>())}};
OUTPUT_MAP(SpaceToBatchD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(SpaceToBatchD, kNameSpaceToBatch, ADPT_DESC(SpaceToBatchD))

// SpaceToBatchND
INPUT_MAP(SpaceToBatchND) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(block_shape)}, {3, INPUT_DESC(paddings)}};
ATTR_MAP(SpaceToBatchND) = EMPTY_ATTR_MAP;
OUTPUT_MAP(SpaceToBatchND) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(SpaceToBatchND, kNameSpaceToBatchNDD, ADPT_DESC(SpaceToBatchND))
REG_ADPT_DESC(SpaceToBatchTF, kNameSpaceToBatchTF, ADPT_DESC(SpaceToBatchND))

// BatchToSpaceD
INPUT_MAP(BatchToSpaceD) = {{1, INPUT_DESC(x)}};
ATTR_MAP(BatchToSpaceD) = {
  {"block_size", ATTR_DESC(block_size, AnyTraits<int64_t>())},
  {"crops", ATTR_DESC(crops, AnyTraits<std::vector<std::vector<int64_t>>>(), AnyTraits<std::vector<int64_t>>())}};
OUTPUT_MAP(BatchToSpaceD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(BatchToSpaceD, kNameBatchToSpace, ADPT_DESC(BatchToSpaceD))

// BatchToSpaceND
INPUT_MAP(BatchToSpaceND) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(block_shape)}, {3, INPUT_DESC(crops)}};
ATTR_MAP(BatchToSpaceND) = EMPTY_ATTR_MAP;
OUTPUT_MAP(BatchToSpaceND) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(BatchToSpaceND, kNameBatchToSpaceNd, ADPT_DESC(BatchToSpaceND))
REG_ADPT_DESC(BatchToSpaceTF, kNameBatchToSpaceTF, ADPT_DESC(BatchToSpaceND))
}  // namespace mindspore::transform
