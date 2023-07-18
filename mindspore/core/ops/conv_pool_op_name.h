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

#ifndef MINDSPORE_CORE_BASE_CONV_POOL_OP_NAME_H_
#define MINDSPORE_CORE_BASE_CONV_POOL_OP_NAME_H_

namespace mindspore {
constexpr auto kFractionalMaxPoolWithFixedKsizeOpName = "FractionalMaxPoolWithFixedKsize";
constexpr auto kFractionalMaxPoolGradWithFixedKsizeOpName = "FractionalMaxPoolGradWithFixedKsize";
constexpr auto kAdaptiveMaxPool3DOpName = "AdaptiveMaxPool3D";
constexpr auto kFractionalMaxPool3DWithFixedKsizeOpName = "FractionalMaxPool3DWithFixedKsize";
constexpr auto kFractionalMaxPool3DGradWithFixedKsizeOpName = "FractionalMaxPool3DGradWithFixedKsize";
constexpr auto kFractionalMaxPoolOpName = "FractionalMaxPool";
constexpr auto kFractionalMaxPoolGradOpName = "FractionalMaxPoolGrad";
constexpr auto kFractionalAvgPoolOpName = "FractionalAvgPool";
constexpr auto kFractionalAvgPoolGradOpName = "FractionalAvgPoolGrad";
constexpr auto kMaxUnpool2DOpName = "MaxUnpool2D";
constexpr auto kMaxUnpool2DGradOpName = "MaxUnpool2DGrad";
constexpr auto kMaxUnpool3DOpName = "MaxUnpool3D";
constexpr auto kMaxUnpool3DGradOpName = "MaxUnpool3DGrad";
constexpr auto kAdaptiveMaxPool2DOpName = "AdaptiveMaxPool2D";
constexpr auto kMaxPool3DWithArgmaxOpName = "MaxPool3DWithArgmax";
constexpr auto kMaxPool3DGradWithArgmaxOpName = "MaxPool3DGradWithArgmax";
constexpr auto kAdaptiveMaxPool3DGradOpName = "AdaptiveMaxPool3DGrad";
constexpr auto kConv2DTransposeOpName = "Conv2DTranspose";
constexpr auto kAdaptiveMaxPool2dOpName = "AdaptiveMaxPool2d";
constexpr auto kAdaptiveAvgPool2DOpName = "AdaptiveAvgPool2D";
constexpr auto kAdaptiveAvgPool2DGradOpName = "AdaptiveAvgPool2DGrad";
constexpr auto kAdaptiveAvgPool3DOpName = "AdaptiveAvgPool3D";
constexpr auto kAdaptiveAvgPool3DGradOpName = "AdaptiveAvgPool3DGrad";
constexpr auto kAdaptiveMaxPool2DGradOpName = "AdaptiveMaxPool2DGrad";
constexpr auto kAvgPool3DGradOpName = "AvgPool3DGrad";
constexpr auto kAvgPool3DGradDOpName = "AvgPool3DGradD";
constexpr auto kAvgPool3DOpName = "AvgPool3D";
constexpr auto kAvgPoolGradGeOpName = "AvgPoolGradGe";
constexpr auto kAvgPool3DDOpName = "AvgPool3DD";
constexpr auto kAvgPoolGradOpName = "AvgPoolGrad";
constexpr auto kAvgPoolGradV1OpName = "AvgPoolGradV1";
constexpr auto kAvgPoolGradDOpName = "AvgPoolGradD";
constexpr auto kAvgPoolGradVmOpName = "AvgPoolGradVm";
constexpr auto kAvgPoolOpName = "AvgPool";
constexpr auto kAvgPoolV1OpName = "AvgPoolV1";
constexpr auto kConv2DBackpropFilterOpName = "Conv2DBackpropFilter";
constexpr auto kConv2DBackpropFilterDOpName = "Conv2DBackpropFilterD";
constexpr auto kConv2DBackpropInputOpName = "Conv2DBackpropInput";
constexpr auto kConv2DBackpropInputDOpName = "Conv2DBackpropInputD";
constexpr auto kConv2DOpName = "Conv2D";
constexpr auto kConv3DBackpropFilterOpName = "Conv3DBackpropFilter";
constexpr auto kConv3DBackpropFilterDOpName = "Conv3DBackpropFilterD";
constexpr auto kConv3DBackpropInputOpName = "Conv3DBackpropInput";
constexpr auto kConv3DBackpropInputDOpName = "Conv3DBackpropInputD";
constexpr auto kConv3DOpName = "Conv3D";
constexpr auto kConv3DTransposeDOpName = "Conv3DTransposeD";
constexpr auto kConv3DTransposeOpName = "Conv3DTranspose";
constexpr auto kDepthwiseConv2dNativeBackpropFilterOpName = "DepthwiseConv2dNativeBackpropFilter";
constexpr auto kDepthwiseConv2dNativeBackpropFilterDOpName = "DepthwiseConv2dNativeBackpropFilterD";
constexpr auto kDepthwiseConv2dNativeBackpropInputOpName = "DepthwiseConv2dNativeBackpropInput";
constexpr auto kDepthwiseConv2dNativeBackpropInputDOpName = "DepthwiseConv2dNativeBackpropInputD";
constexpr auto kDepthwiseConv2dNativeOpName = "DepthwiseConv2dNative";
constexpr auto kMaxPool3DGradGradOpName = "MaxPool3DGradGrad";
constexpr auto kMaxPool3DGradGradDOpName = "MaxPool3DGradGradD";
constexpr auto kMaxPool3DGradOpName = "MaxPool3DGrad";
constexpr auto kMaxPool3DOpName = "MaxPool3D";
constexpr auto kMaxPoolGradOpName = "MaxPoolGrad";
constexpr auto kMaxPoolGradV1OpName = "MaxPoolGradV1";
constexpr auto kMaxPoolGradWithArgmaxOpName = "MaxPoolGradWithArgmax";
constexpr auto kMaxPoolOpName = "MaxPool";
constexpr auto kMaxPoolV1OpName = "MaxPoolV1";
constexpr auto kMaxPoolWithArgmaxOpName = "MaxPoolWithArgmax";
constexpr auto kPoolingOpName = "Pooling";
constexpr auto kPSROIPoolingOpName = "PSROIPooling";
constexpr auto kPSROIPoolingV2OpName = "PSROIPoolingV2";
constexpr auto kPSROIPoolingGradOpName = "PSROIPoolingGrad";
constexpr auto kPSROIPoolingGradV2DOpName = "PSROIPoolingGradV2D";
}  // namespace mindspore

#endif  // MINDSPORE_CORE_BASE_CONV_POOL_OP_NAME_H_
