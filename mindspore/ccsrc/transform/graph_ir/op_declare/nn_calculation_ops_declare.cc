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
  {"stride", ATTR_DESC(strides, "pad", AnyTraits<std::vector<int64_t>>())},
  {"dilation", ATTR_DESC(dilations, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"format", ATTR_DESC(data_format, AnyTraits<std::string>())},
  {"group", ATTR_DESC(groups, AnyTraits<int64_t>())},
};
OUTPUT_MAP(Conv2DBackpropInputD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Conv2DBackpropInputD, prim::kPrimConv2DBackpropInput->name(), ADPT_DESC(Conv2DBackpropInputD))

// Conv2DBackpropFilterD
INPUT_MAP(Conv2DBackpropFilterD) = {{1, INPUT_DESC(out_backprop)}, {2, INPUT_DESC(x)}};
INPUT_ATTR_MAP(Conv2DBackpropFilterD) = {
  {3, ATTR_DESC(filter_size, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};
ATTR_MAP(Conv2DBackpropFilterD) = {
  {"pad_list", ATTR_DESC(pads, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"stride", ATTR_DESC(strides, "pad", AnyTraits<std::vector<int64_t>>())},
  {"dilation", ATTR_DESC(dilations, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"format", ATTR_DESC(data_format, AnyTraits<std::string>())},
  {"group", ATTR_DESC(groups, AnyTraits<int64_t>())},
};
OUTPUT_MAP(Conv2DBackpropFilterD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Conv2DBackpropFilterD, prim::kPrimConv2DBackpropFilter->name(), ADPT_DESC(Conv2DBackpropFilterD))

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
}  // namespace mindspore::transform
