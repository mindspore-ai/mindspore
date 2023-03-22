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
REG_ADPT_DESC(Unstack, prim::kUnstack, ADPT_DESC(Unpack))
REG_ADPT_DESC(Unpack, prim::kUnpack, ADPT_DESC(Unpack))

// ExtractImagePatches
INPUT_MAP(ExtractImagePatches) = {{1, INPUT_DESC(x)}};
ATTR_MAP(ExtractImagePatches) = {
  {"ksizes", ATTR_DESC(ksizes, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
  {"strides", ATTR_DESC(strides, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
  {"rates", ATTR_DESC(rates, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
  {"padding", ATTR_DESC(padding, AnyTraits<std::string>())}};
OUTPUT_MAP(ExtractImagePatches) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ExtractImagePatches, kNameExtractImagePatches, ADPT_DESC(ExtractImagePatches))

// TransData
INPUT_MAP(TransData) = {{1, INPUT_DESC(src)}};
ATTR_MAP(TransData) = {{"src_format", ATTR_DESC(src_format, AnyTraits<std::string>())},
                       {"dst_format", ATTR_DESC(dst_format, AnyTraits<std::string>())},
                       {"groups", ATTR_DESC(groups, AnyTraits<int64_t>())}};
OUTPUT_MAP(TransData) = {{0, OUTPUT_DESC(dst)}};
REG_ADPT_DESC(TransData, kNameTransData, ADPT_DESC(TransData))

// TransDataRNN
INPUT_MAP(TransDataRNN) = {{1, INPUT_DESC(src)}};
ATTR_MAP(TransDataRNN) = {{"src_format", ATTR_DESC(src_format, AnyTraits<std::string>())},
                          {"dst_format", ATTR_DESC(dst_format, AnyTraits<std::string>())},
                          {"input_size", ATTR_DESC(input_size, AnyTraits<int64_t>())},
                          {"hidden_size", ATTR_DESC(hidden_size, AnyTraits<int64_t>())}};
OUTPUT_MAP(TransDataRNN) = {{0, OUTPUT_DESC(dst)}};
REG_ADPT_DESC(TransDataRNN, prim::kPrimTransDataRNN->name(), ADPT_DESC(TransDataRNN))

// Transpose
INPUT_MAP(Transpose) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(perm)}};
ATTR_INPUT_MAP(Transpose) = {{"perm", "perm"}};
ATTR_MAP(Transpose) = EMPTY_ATTR_MAP;
// Do not set Transpose operator output descriptor
OUTPUT_MAP(Transpose) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Transpose, prim::kPrimTranspose->name(), ADPT_DESC(Transpose))
REG_ADPT_DESC(TransposeD, prim::kPrimTransposeD->name(), ADPT_DESC(Transpose))

// SpaceToDepth
INPUT_MAP(SpaceToDepth) = {{1, INPUT_DESC(x)}};
ATTR_MAP(SpaceToDepth) = {{"block_size", ATTR_DESC(block_size, AnyTraits<int64_t>())},
                          {"data_format", ATTR_DESC(data_format, AnyTraits<std::string>())}};
OUTPUT_MAP(SpaceToDepth) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(SpaceToDepth, kNameSpaceToDepth, ADPT_DESC(SpaceToDepth))

// DepthToSpace
INPUT_MAP(DepthToSpace) = {{1, INPUT_DESC(x)}};
ATTR_MAP(DepthToSpace) = {{"block_size", ATTR_DESC(block_size, AnyTraits<int64_t>())},
                          {"mode", ATTR_DESC(mode, AnyTraits<std::string>())},
                          {"data_format", ATTR_DESC(data_format, AnyTraits<std::string>())}};
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
ATTR_INPUT_MAP(SpaceToBatchND) = {{"block_shape", "block_shape"}, {"paddings", "paddings"}};
ATTR_MAP(SpaceToBatchND) = EMPTY_ATTR_MAP;
OUTPUT_MAP(SpaceToBatchND) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(SpaceToBatchND, kNameSpaceToBatchND, ADPT_DESC(SpaceToBatchND))
REG_ADPT_DESC(SpaceToBatchTF, kNameSpaceToBatchTF, ADPT_DESC(SpaceToBatchND))
REG_ADPT_DESC(SpaceToBatchNDD, kNameSpaceToBatchNDD, ADPT_DESC(SpaceToBatchND))

// BatchToSpaceD
INPUT_MAP(BatchToSpaceD) = {{1, INPUT_DESC(x)}};
ATTR_MAP(BatchToSpaceD) = {
  {"block_size", ATTR_DESC(block_size, AnyTraits<int64_t>())},
  {"crops", ATTR_DESC(crops, AnyTraits<std::vector<std::vector<int64_t>>>(), AnyTraits<std::vector<int64_t>>())}};
OUTPUT_MAP(BatchToSpaceD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(BatchToSpaceD, kNameBatchToSpace, ADPT_DESC(BatchToSpaceD))

// BatchToSpace
INPUT_MAP(BatchToSpace) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(crops)}};
ATTR_INPUT_MAP(BatchToSpace) = {{"crops", "crops"}};
ATTR_MAP(BatchToSpace) = {{"block_size", ATTR_DESC(block_size, AnyTraits<int64_t>())}};
OUTPUT_MAP(BatchToSpace) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(BatchToSpace, kBatchToSpaceDOpName, ADPT_DESC(BatchToSpace))

// SpaceToBatch
INPUT_MAP(SpaceToBatch) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(paddings)}};
ATTR_INPUT_MAP(SpaceToBatch) = {{"paddings", "paddings"}};
ATTR_MAP(SpaceToBatch) = {{"block_size", ATTR_DESC(block_size, AnyTraits<int64_t>())}};
OUTPUT_MAP(SpaceToBatch) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(SpaceToBatch, kSpaceToBatchDOpName, ADPT_DESC(SpaceToBatch))

// ExtractVolumePatches
INPUT_MAP(ExtractVolumePatches) = {{1, INPUT_DESC(x)}};
ATTR_MAP(ExtractVolumePatches) = {
  {"kernel_size", ATTR_DESC(ksizes, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
  {"strides", ATTR_DESC(strides, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
  {"padding", ATTR_DESC(padding, AnyTraits<std::string>())}};
OUTPUT_MAP(ExtractVolumePatches) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ExtractVolumePatches, prim::kPrimExtractVolumePatches->name(), ADPT_DESC(ExtractVolumePatches))

// BatchToSpaceND
INPUT_MAP(BatchToSpaceND) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(block_shape)}, {3, INPUT_DESC(crops)}};
ATTR_INPUT_MAP(BatchToSpaceND) = {{"block_shape", "block_shape"}, {"crops", "crops"}};
ATTR_MAP(BatchToSpaceND) = EMPTY_ATTR_MAP;
OUTPUT_MAP(BatchToSpaceND) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(BatchToSpaceND, kNameBatchToSpaceNd, ADPT_DESC(BatchToSpaceND))
REG_ADPT_DESC(BatchToSpaceNDD, kBatchToSpaceNDDOpName, ADPT_DESC(BatchToSpaceND))
REG_ADPT_DESC(BatchToSpaceTF, kNameBatchToSpaceTF, ADPT_DESC(BatchToSpaceND))
REG_ADPT_DESC(kNameBatchToSpaceNdV2, kNameBatchToSpaceNdV2, ADPT_DESC(BatchToSpaceND))
REG_ADPT_DESC(kNameBatchToSpaceNDD, kBatchToSpaceNDDOpName, ADPT_DESC(BatchToSpaceND))
}  // namespace mindspore::transform
