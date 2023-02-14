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
#include <vector>
#include <string>

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
REG_ADPT_DESC(Conv2DBackpropInputD, prim::kPrimConv2DBackpropInput->name(), ADPT_DESC(Conv2DBackpropInputD))

// Conv2DBackpropInput for tf inference
INPUT_MAP(Conv2DBackpropInput) = {{1, INPUT_DESC(input_size)}, {2, INPUT_DESC(filter)}, {3, INPUT_DESC(out_backprop)}};
ATTR_MAP(Conv2DBackpropInput) = {
  {"stride", ATTR_DESC(strides, AnyTraits<std::vector<int64_t>>())},
  {"dilation", ATTR_DESC(dilations, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"pad_list", ATTR_DESC(pads, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"data_format", ATTR_DESC(data_format, AnyTraits<std::string>())},
};
OUTPUT_MAP(Conv2DBackpropInput) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Conv2DBackpropInput, kNameConv2DBackpropInputV2, ADPT_DESC(Conv2DBackpropInput))

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
REG_ADPT_DESC(Conv2DTranspose, kConv2DTransposeOpName, ADPT_DESC(Conv2DBackpropInputD))

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
REG_ADPT_DESC(Conv2DBackpropFilterD, prim::kPrimConv2DBackpropFilter->name(), ADPT_DESC(Conv2DBackpropFilterD))

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
REG_ADPT_DESC(Conv3DTransposeD, kNameConv3DTransposeD, ADPT_DESC(Conv3DTransposeD))

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

// Conv3DBackpropInputD
INPUT_MAP(Conv3DBackpropInputD) = {{1, INPUT_DESC(out_backprop)}, {2, INPUT_DESC(filter)}};
INPUT_ATTR_MAP(Conv3DBackpropInputD) = {
  {3, ATTR_DESC(input_size, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};
ATTR_MAP(Conv3DBackpropInputD) = {
  {"pad_list", ATTR_DESC(pads, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"strides", ATTR_DESC(strides, AnyTraits<std::vector<int64_t>>())},
  {"dilations", ATTR_DESC(dilations, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"format", ATTR_DESC(data_format, AnyTraits<std::string>())},
  {"groups", ATTR_DESC(groups, AnyTraits<int64_t>())},
};
OUTPUT_MAP(Conv3DBackpropInputD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Conv3DBackpropInputD, kNameConv3DBackpropInputD, ADPT_DESC(Conv3DBackpropInputD))

// Conv3DBackpropFilterD
INPUT_MAP(Conv3DBackpropFilterD) = {{1, INPUT_DESC(out_backprop)}, {2, INPUT_DESC(x)}};
INPUT_ATTR_MAP(Conv3DBackpropFilterD) = {
  {3, ATTR_DESC(filter_size, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};
ATTR_MAP(Conv3DBackpropFilterD) = {
  {"strides", ATTR_DESC(strides, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"pad_list", ATTR_DESC(pads, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"dilations", ATTR_DESC(dilations, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"groups", ATTR_DESC(groups, AnyTraits<int64_t>())},
  {"format", ATTR_DESC(data_format, AnyTraits<std::string>())},
};
OUTPUT_MAP(Conv3DBackpropFilterD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Conv3DBackpropFilterD, kNameConv3DBackpropFilterD, ADPT_DESC(Conv3DBackpropFilterD))

// DepthwiseConv2D
INPUT_MAP(DepthwiseConv2D) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(filter)}, {3, INPUT_DESC(bias)}};
ATTR_MAP(DepthwiseConv2D) = {
  {"stride", ATTR_DESC(strides, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"pad_list", ATTR_DESC(pads, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"dilation", ATTR_DESC(dilations, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"format", ATTR_DESC(data_format, AnyTraits<std::string>())},
};
OUTPUT_MAP(DepthwiseConv2D) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(DepthwiseConv2D, prim::kPrimDepthwiseConv2dNative->name(), ADPT_DESC(DepthwiseConv2D))

// DepthwiseConv2DBackpropInputD
INPUT_MAP(DepthwiseConv2DBackpropInputD) = {{2, INPUT_DESC(filter)}, {3, INPUT_DESC(out_backprop)}};
INPUT_ATTR_MAP(DepthwiseConv2DBackpropInputD) = {
  {1, ATTR_DESC(input_size, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};
ATTR_MAP(DepthwiseConv2DBackpropInputD) = {
  {"stride", ATTR_DESC(strides, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"pad_list", ATTR_DESC(pads, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"dilation", ATTR_DESC(dilations, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
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
}  // namespace mindspore::transform
