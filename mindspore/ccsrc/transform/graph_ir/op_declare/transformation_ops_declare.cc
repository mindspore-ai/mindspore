/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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
#include <string>
#include <vector>
#include "ops/array_ops.h"
#include "ops/nn_ops.h"
#include "ops/image_ops.h"

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
REG_ADPT_DESC(Unstack, mindspore::kUnstackOpName, ADPT_DESC(Unpack))
REG_ADPT_DESC(Unpack, mindspore::kUnpackOpName, ADPT_DESC(Unpack))

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

// TfIdfVectorizer
INPUT_MAP(TfIdfVectorizer) = {{1, INPUT_DESC(input)}};
ATTR_MAP(TfIdfVectorizer) = {
  {"max_gram_length", ATTR_DESC(max_gram_length, AnyTraits<int64_t>())},
  {"max_skip_count", ATTR_DESC(max_skip_count, AnyTraits<int64_t>())},
  {"min_gram_length", ATTR_DESC(min_gram_length, AnyTraits<int64_t>())},
  {"mode", ATTR_DESC(mode, AnyTraits<std::string>())},
  {"ngram_counts", ATTR_DESC(ngram_counts, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"ngram_indexes", ATTR_DESC(ngram_indexes, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"pool_int64s", ATTR_DESC(pool_int64s, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"pool_strings", ATTR_DESC(pool_strings, AnyTraits<std::vector<std::string>>())},
  {"weights", ATTR_DESC(weights, AnyTraits<std::vector<float>>())}};
OUTPUT_MAP(TfIdfVectorizer) = {{0, OUTPUT_DESC(output)}};
REG_ADPT_DESC(TfIdfVectorizer, kNameTfIdfVectorizer, ADPT_DESC(TfIdfVectorizer))

// AffineGrid
INPUT_MAP(AffineGrid) = {{1, INPUT_DESC(theta)}, {2, INPUT_DESC(output_size)}};
ATTR_MAP(AffineGrid) = {{"align_corners", ATTR_DESC(align_corners, AnyTraits<bool>())}};
OUTPUT_MAP(AffineGrid) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(AffineGrid, prim::kPrimAffineGrid->name(), ADPT_DESC(AffineGrid));

// AffineGridGrad
CUST_INPUT_MAP(AffineGridGrad) = {{1, INPUT_DESC(y_grad)}, {2, INPUT_DESC(x_size)}};
CUST_ATTR_MAP(AffineGridGrad) = {{"align_corners", ATTR_DESC(align_corners, AnyTraits<bool>())}};
CUST_OUTPUT_MAP(AffineGridGrad) = {{0, OUTPUT_DESC(x_grad)}};
REG_ADPT_DESC(AffineGridGrad, prim::kPrimAffineGridGrad->name(), CUST_ADPT_DESC(AffineGridGrad));

// Im2col
INPUT_MAP(Im2col) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Im2col) = {{"ksizes", ATTR_DESC(ksizes, AnyTraits<std::vector<int64_t>>())},
                    {"strides", ATTR_DESC(strides, AnyTraits<std::vector<int64_t>>())},
                    {"dilations", ATTR_DESC(dilations, AnyTraits<std::vector<int64_t>>())},
                    {"padding_mode", ATTR_DESC(padding_mode, AnyTraits<std::string>())},
                    {"pads", ATTR_DESC(pads, AnyTraits<std::vector<int64_t>>())}};
OUTPUT_MAP(Im2col) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Im2col, kNameIm2col, ADPT_DESC(Im2col))
}  // namespace mindspore::transform
