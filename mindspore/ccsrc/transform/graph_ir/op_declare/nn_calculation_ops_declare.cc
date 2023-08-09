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

#include "transform/graph_ir/op_declare/nn_calculation_ops_declare.h"
#include <string>
#include <vector>
#include "ops/conv_pool_ops.h"
#include "ops/nn_ops.h"

namespace mindspore::transform {
// BiasAddGrad
INPUT_MAP(BiasAddGrad) = {{1, INPUT_DESC(x)}};
ATTR_MAP(BiasAddGrad) = {{"format", ATTR_DESC(data_format, AnyTraits<std::string>())}};
OUTPUT_MAP(BiasAddGrad) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(BiasAddGrad, prim::kPrimBiasAddGrad->name(), ADPT_DESC(BiasAddGrad))

// Conv2D
INPUT_MAP(Conv2D) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(filter)}, {3, INPUT_DESC(bias)}};
ATTR_MAP(Conv2D) = {
  {"stride", ATTR_DESC(strides, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"pad_list", ATTR_DESC(pads, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"dilation", ATTR_DESC(dilations, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"format", ATTR_DESC(data_format, AnyTraits<std::string>())},
  {"group", ATTR_DESC(groups, AnyTraits<int64_t>())},
};
OUTPUT_MAP(Conv2D) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Conv2D, prim::kPrimConv2D->name(), ADPT_DESC(Conv2D))

// Conv2DBackpropInput
INPUT_MAP(Conv2DBackpropInput) = {{1, INPUT_DESC(out_backprop)}, {2, INPUT_DESC(filter)}, {3, INPUT_DESC(input_size)}};
ATTR_INPUT_MAP(Conv2DBackpropInput) = {{"input_sizes", "input_size"}};
ATTR_MAP(Conv2DBackpropInput) = {
  {"pad_list", ATTR_DESC(pads, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"stride", ATTR_DESC(strides, AnyTraits<std::vector<int64_t>>())},
  {"dilation", ATTR_DESC(dilations, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"format", ATTR_DESC(data_format, AnyTraits<std::string>())},
  {"group", ATTR_DESC(groups, AnyTraits<int64_t>())},
};
OUTPUT_MAP(Conv2DBackpropInput) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Conv2DBackpropInput, prim::kPrimConv2DBackpropInput->name(), ADPT_DESC(Conv2DBackpropInput))
REG_ADPT_DESC(Conv2DBackpropInputD, kNameConv2DBackpropInputD, ADPT_DESC(Conv2DBackpropInput))

// Conv2DBackpropInputD
INPUT_MAP(Conv2DBackpropInputD) = {{1, INPUT_DESC(out_backprop)}, {2, INPUT_DESC(filter)}};
INPUT_ATTR_MAP(Conv2DBackpropInputD) = {
  {3, ATTR_DESC(input_size, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};
ATTR_MAP(Conv2DBackpropInputD) = {
  {"pad_list", ATTR_DESC(pads, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"stride", ATTR_DESC(strides, AnyTraits<std::vector<int64_t>>())},
  {"dilation", ATTR_DESC(dilations, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"format", ATTR_DESC(data_format, AnyTraits<std::string>())},
  {"group", ATTR_DESC(groups, AnyTraits<int64_t>())},
};
OUTPUT_MAP(Conv2DBackpropInputD) = {{0, OUTPUT_DESC(y)}};

// Conv2DBackpropInput for tf inference
REG_ADPT_DESC(Conv2DBackpropInputV2, kNameConv2DBackpropInputV2, ADPT_DESC(Conv2DBackpropInput))

// Deconvolution for caffe inference
INPUT_MAP(Deconvolution) = {
  {1, INPUT_DESC(x)}, {2, INPUT_DESC(filter)}, {3, INPUT_DESC(bias)}, {4, INPUT_DESC(offset_w)}};
ATTR_MAP(Deconvolution) = {
  {"stride", ATTR_DESC(strides, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"pad_list", ATTR_DESC(pads, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"dilation", ATTR_DESC(dilations, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"group", ATTR_DESC(groups, AnyTraits<int64_t>())},
  {"format", ATTR_DESC(data_format, AnyTraits<string>())},
  {"offset", ATTR_DESC(offset_x, AnyTraits<int64_t>())}};
OUTPUT_MAP(Deconvolution) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Deconvolution, kNameDeconvolution, ADPT_DESC(Deconvolution))
REG_ADPT_DESC(Conv2DTranspose, kConv2DTransposeOpName, ADPT_DESC(Conv2DBackpropInput))

// Conv2DTransposeD for tf onnx inference
INPUT_MAP(Conv2DTransposeD) = {
  {1, INPUT_DESC(x)}, {2, INPUT_DESC(filter)}, {3, INPUT_DESC(bias)}, {4, INPUT_DESC(offset_w)}};
ATTR_MAP(Conv2DTransposeD) = {
  {"input_size", ATTR_DESC(input_size, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"stride", ATTR_DESC(strides, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"pad_list", ATTR_DESC(pads, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"dilation", ATTR_DESC(dilations, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"group", ATTR_DESC(groups, AnyTraits<int64_t>())},
  {"data_format", ATTR_DESC(data_format, AnyTraits<string>())},
  {"output_paddings", ATTR_DESC(output_padding, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"offset", ATTR_DESC(offset_x, AnyTraits<int64_t>())}};
OUTPUT_MAP(Conv2DTransposeD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Conv2DTransposeD, kNameConv2DTransposeD, ADPT_DESC(Conv2DTransposeD))

// Conv2DBackpropFilter
INPUT_MAP(Conv2DBackpropFilter) = {{1, INPUT_DESC(out_backprop)}, {2, INPUT_DESC(x)}, {3, INPUT_DESC(filter_size)}};
ATTR_INPUT_MAP(Conv2DBackpropFilter) = {{"filter_sizes", "filter_size"}};
ATTR_MAP(Conv2DBackpropFilter) = {
  {"pad_list", ATTR_DESC(pads, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"stride", ATTR_DESC(strides, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"dilation", ATTR_DESC(dilations, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"format", ATTR_DESC(data_format, AnyTraits<std::string>())},
  {"group", ATTR_DESC(groups, AnyTraits<int64_t>())},
};
OUTPUT_MAP(Conv2DBackpropFilter) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Conv2DBackpropFilter, prim::kPrimConv2DBackpropFilter->name(), ADPT_DESC(Conv2DBackpropFilter))
REG_ADPT_DESC(Conv2DBackpropFilterD, kNameConv2DBackpropFilterD, ADPT_DESC(Conv2DBackpropFilter))

// Conv2DBackpropFilterD
INPUT_MAP(Conv2DBackpropFilterD) = {{1, INPUT_DESC(out_backprop)}, {2, INPUT_DESC(x)}};
INPUT_ATTR_MAP(Conv2DBackpropFilterD) = {
  {3, ATTR_DESC(filter_size, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};
ATTR_MAP(Conv2DBackpropFilterD) = {
  {"pad_list", ATTR_DESC(pads, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"stride", ATTR_DESC(strides, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"dilation", ATTR_DESC(dilations, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"format", ATTR_DESC(data_format, AnyTraits<std::string>())},
  {"group", ATTR_DESC(groups, AnyTraits<int64_t>())},
};
OUTPUT_MAP(Conv2DBackpropFilterD) = {{0, OUTPUT_DESC(y)}};

// Conv3DTransposeD
INPUT_MAP(Conv3DTransposeD) = {
  {1, INPUT_DESC(x)}, {2, INPUT_DESC(filter)}, {3, INPUT_DESC(bias)}, {4, INPUT_DESC(offset_w)}};
ATTR_MAP(Conv3DTransposeD) = {
  {"input_size", ATTR_DESC(input_size, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"strides", ATTR_DESC(strides, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"pad_list", ATTR_DESC(pads, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"dilations", ATTR_DESC(dilations, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"groups", ATTR_DESC(groups, AnyTraits<int64_t>())},
  {"format", ATTR_DESC(data_format, AnyTraits<std::string>())},
  {"output_padding", ATTR_DESC(output_padding, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
};
OUTPUT_MAP(Conv3DTransposeD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Conv3DTransposeD, kNameConv3DTranspose, ADPT_DESC(Conv3DTransposeD))

// Conv3D
INPUT_MAP(Conv3D) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(filter)}, {3, INPUT_DESC(bias)}, {4, INPUT_DESC(offset_w)}};
ATTR_MAP(Conv3D) = {
  {"strides", ATTR_DESC(strides, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"pad_list", ATTR_DESC(pads, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"dilations", ATTR_DESC(dilations, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"groups", ATTR_DESC(groups, AnyTraits<int64_t>())},
  {"format", ATTR_DESC(data_format, AnyTraits<std::string>())},
  {"offset_x", ATTR_DESC(offset_x, AnyTraits<int64_t>())},
};
OUTPUT_MAP(Conv3D) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Conv3D, kNameConv3D, ADPT_DESC(Conv3D))

// Conv3DBackpropInput
INPUT_MAP(Conv3DBackpropInput) = {{1, INPUT_DESC(filter)}, {2, INPUT_DESC(out_backprop)}, {3, INPUT_DESC(input_size)}};
ATTR_MAP(Conv3DBackpropInput) = {
  {"pad_list", ATTR_DESC(pads, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"strides", ATTR_DESC(strides, AnyTraits<std::vector<int64_t>>())},
  {"dilations", ATTR_DESC(dilations, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"format", ATTR_DESC(data_format, AnyTraits<std::string>())},
  {"groups", ATTR_DESC(groups, AnyTraits<int64_t>())},
};
OUTPUT_MAP(Conv3DBackpropInput) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Conv3DBackpropInput, kNameConv3DBackpropInput, ADPT_DESC(Conv3DBackpropInput))

// DepthwiseConv2D
INPUT_MAP(DepthwiseConv2D) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(filter)}, {3, INPUT_DESC(bias)}};
ATTR_MAP(DepthwiseConv2D) = {
  {"stride", ATTR_DESC(strides, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"pad_list", ATTR_DESC(pads, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"dilation", ATTR_DESC(dilations, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"format", ATTR_DESC(data_format, AnyTraits<std::string>())},
};
OUTPUT_MAP(DepthwiseConv2D) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(DepthwiseConv2DNative, prim::kPrimDepthwiseConv2dNative->name(), ADPT_DESC(DepthwiseConv2D))
REG_ADPT_DESC(DepthwiseConv2D, kDepthwiseConv2DOpName, ADPT_DESC(DepthwiseConv2D))

// DepthwiseConv2DBackpropInputD
INPUT_MAP(DepthwiseConv2DBackpropInputD) = {{2, INPUT_DESC(filter)}, {3, INPUT_DESC(out_backprop)}};
INPUT_ATTR_MAP(DepthwiseConv2DBackpropInputD) = {
  {1, ATTR_DESC(input_size, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};
ATTR_MAP(DepthwiseConv2DBackpropInputD) = {
  {"stride", ATTR_DESC(strides, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"pad_list", ATTR_DESC(pads, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"dilation", ATTR_DESC(dilations, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"format", ATTR_DESC(data_format, AnyTraits<std::string>())},
};
OUTPUT_MAP(DepthwiseConv2DBackpropInputD) = {{0, OUTPUT_DESC(input_grad)}};
REG_ADPT_DESC(DepthwiseConv2DBackpropInputD, prim::kPrimDepthwiseConv2dNativeBackpropInput->name(),
              ADPT_DESC(DepthwiseConv2DBackpropInputD))

// DepthwiseConv2DBackpropFilterD
INPUT_MAP(DepthwiseConv2DBackpropFilterD) = {{1, INPUT_DESC(input)}, {3, INPUT_DESC(out_backprop)}};
INPUT_ATTR_MAP(DepthwiseConv2DBackpropFilterD) = {
  {2, ATTR_DESC(filter_size, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};
ATTR_MAP(DepthwiseConv2DBackpropFilterD) = {
  {"stride", ATTR_DESC(strides, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"pad_list", ATTR_DESC(pads, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"dilation", ATTR_DESC(dilations, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"format", ATTR_DESC(data_format, AnyTraits<std::string>())},
};
OUTPUT_MAP(DepthwiseConv2DBackpropFilterD) = {{0, OUTPUT_DESC(filter_grad)}};
REG_ADPT_DESC(DepthwiseConv2DBackpropFilterD, prim::kPrimDepthwiseConv2dNativeBackpropFilter->name(),
              ADPT_DESC(DepthwiseConv2DBackpropFilterD))

// DeformableOffsets
INPUT_MAP(DeformableOffsets) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(offsets)}};
ATTR_MAP(DeformableOffsets) = {
  {"strides", ATTR_DESC(strides, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"pads", ATTR_DESC(pads, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"ksize", ATTR_DESC(ksize, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"format", ATTR_DESC(data_format, AnyTraits<std::string>())},
  {"dilations", ATTR_DESC(dilations, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"deformable_groups", ATTR_DESC(deformable_groups, AnyTraits<int64_t>())},
  {"modulated", ATTR_DESC(modulated, AnyTraits<bool>())}};
OUTPUT_MAP(DeformableOffsets) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(DeformableOffsets, kDeformableOffsetsOpName, ADPT_DESC(DeformableOffsets))

// DeformableOffsetsGrad
INPUT_MAP(DeformableOffsetsGrad) = {{1, INPUT_DESC(grad)}, {2, INPUT_DESC(x)}, {3, INPUT_DESC(offsets)}};
ATTR_MAP(DeformableOffsetsGrad) = {
  {"strides", ATTR_DESC(strides, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"pads", ATTR_DESC(pads, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"ksize", ATTR_DESC(ksize, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"data_format", ATTR_DESC(data_format, AnyTraits<std::string>())},
  {"dilations", ATTR_DESC(dilations, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"deformable_groups", ATTR_DESC(deformable_groups, AnyTraits<int64_t>())},
  {"modulated", ATTR_DESC(modulated, AnyTraits<bool>())}};
OUTPUT_MAP(DeformableOffsetsGrad) = {{0, OUTPUT_DESC(grad_x)}, {1, OUTPUT_DESC(grad_offsets)}};
REG_ADPT_DESC(DeformableOffsetsGrad, prim::kPrimDeformableOffsetsGrad->name(), ADPT_DESC(DeformableOffsetsGrad))

// Conv3DTranspose
INPUT_MAP(Conv3DTranspose) = {{1, INPUT_DESC(input_size)},
                              {2, INPUT_DESC(x)},
                              {3, INPUT_DESC(filter)},
                              {4, INPUT_DESC(bias)},
                              {5, INPUT_DESC(offset_w)}};
ATTR_MAP(Conv3DTranspose) = {
  {"strides", ATTR_DESC(strides, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"pad_list", ATTR_DESC(pads, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"dilations", ATTR_DESC(dilations, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"groups", ATTR_DESC(groups, AnyTraits<int64_t>())},
  {"format", ATTR_DESC(data_format, AnyTraits<std::string>())},
  {"output_padding", ATTR_DESC(output_padding, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"offset_x", ATTR_DESC(offset_x, AnyTraits<int64_t>())}};
ATTR_INPUT_MAP(Conv3DTranspose) = {{"input_size", "input_size"}};
OUTPUT_MAP(Conv3DTranspose) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Conv3DTranspose, kConv3DTransposeDOpName, ADPT_DESC(Conv3DTranspose))

// Conv3DBackpropFilter
INPUT_MAP(Conv3DBackpropFilter) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(out_backprop)}, {3, INPUT_DESC(filter_size)}};
ATTR_INPUT_MAP(Conv3DBackpropFilter) = {{"filter_size", "filter_size"}};
ATTR_MAP(Conv3DBackpropFilter) = {
  {"strides", ATTR_DESC(strides, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"pad_list", ATTR_DESC(pads, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"dilations", ATTR_DESC(dilations, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"groups", ATTR_DESC(groups, AnyTraits<int64_t>())},
  {"format", ATTR_DESC(data_format, AnyTraits<std::string>())},
};
OUTPUT_MAP(Conv3DBackpropFilter) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Conv3DBackpropFilter, kConv3DBackpropFilterOpName, ADPT_DESC(Conv3DBackpropFilter))

// DeformableConv2d
INPUT_MAP(DeformableConv2D) = {
  {1, INPUT_DESC(x)}, {2, INPUT_DESC(filter)}, {3, INPUT_DESC(offsets)}, {4, INPUT_DESC(bias)}};
ATTR_MAP(DeformableConv2D) = {
  {"strides", ATTR_DESC(strides, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"pads", ATTR_DESC(pads, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"dilations", ATTR_DESC(dilations, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"format", ATTR_DESC(data_format, AnyTraits<std::string>())},
  {"deformable_groups", ATTR_DESC(deformable_groups, AnyTraits<int64_t>())},
  {"modulated", ATTR_DESC(modulated, AnyTraits<bool>())},
};
OUTPUT_MAP(DeformableConv2D) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(DeformableConv2D, kDeformableConv2dOpName, ADPT_DESC(DeformableConv2D))

// WKV
INPUT_MAP(Wkv) = {{1, INPUT_DESC(w)}, {2, INPUT_DESC(u)}, {3, INPUT_DESC(k)}, {4, INPUT_DESC(v)},
                  {5, INPUT_DESC(m)}, {6, INPUT_DESC(p)}, {7, INPUT_DESC(q)}};
ATTR_MAP(Wkv) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Wkv) = {{0, OUTPUT_DESC(o)}, {1, OUTPUT_DESC(mo)}, {2, OUTPUT_DESC(po)}, {3, OUTPUT_DESC(qo)}};
REG_ADPT_DESC(Wkv, kNameWKV, ADPT_DESC(Wkv))

// WKVGrad
INPUT_MAP(WkvGrad) = {
  {1, INPUT_DESC(w)}, {2, INPUT_DESC(u)}, {3, INPUT_DESC(k)}, {4, INPUT_DESC(v)}, {5, INPUT_DESC(gy)}};
ATTR_MAP(WkvGrad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(WkvGrad) = {{0, OUTPUT_DESC(gw)}, {1, OUTPUT_DESC(gu)}, {2, OUTPUT_DESC(gk)}, {3, OUTPUT_DESC(gv)}};
REG_ADPT_DESC(WkvGrad, kNameWKVGrad, ADPT_DESC(WkvGrad))

// Conv2DTranspose for tf onnx inference
INPUT_MAP(Conv2DTranspose) = {{1, INPUT_DESC(input_size)},
                              {2, INPUT_DESC(x)},
                              {3, INPUT_DESC(filter)},
                              {4, INPUT_DESC(bias)},
                              {5, INPUT_DESC(offset_w)}};
ATTR_MAP(Conv2DTranspose) = {
  {"stride", ATTR_DESC(strides, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"dilation", ATTR_DESC(dilations, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"data_format", ATTR_DESC(data_format, AnyTraits<string>())},
  {"group", ATTR_DESC(groups, AnyTraits<int64_t>())},
  {"pad_list", ATTR_DESC(pads, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"output_paddings", ATTR_DESC(output_padding, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"offset", ATTR_DESC(offset_x, AnyTraits<int64_t>())}};
OUTPUT_MAP(Conv2DTranspose) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Conv2DTransposeV2, kNameConv2DTransposeV2, ADPT_DESC(Conv2DTranspose))
}  // namespace mindspore::transform
