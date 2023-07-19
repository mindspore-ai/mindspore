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

#ifndef MINDSPORE_CORE_BASE_CONV_POOL_OPS_H_
#define MINDSPORE_CORE_BASE_CONV_POOL_OPS_H_

#include <memory>
#include "ops/conv_pool_op_name.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "utils/hash_map.h"

namespace mindspore {
namespace prim {
// Conv
GVAR_DEF(PrimitivePtr, kPrimConv2DBackpropInput, std::make_shared<Primitive>("Conv2DBackpropInput"));
GVAR_DEF(PrimitivePtr, kPrimConv2DBackpropInputD, std::make_shared<Primitive>("Conv2DBackpropInputD"));
GVAR_DEF(PrimitivePtr, kPrimConv2DBackpropFilter, std::make_shared<Primitive>("Conv2DBackpropFilter"));
GVAR_DEF(PrimitivePtr, kPrimConv2DBackpropFilterD, std::make_shared<Primitive>("Conv2DBackpropFilterD"));
GVAR_DEF(PrimitivePtr, kPrimConv3DBackpropInput, std::make_shared<Primitive>("Conv3DBackpropInput"));
GVAR_DEF(PrimitivePtr, kPrimConv3DBackpropFilter, std::make_shared<Primitive>("Conv3DBackpropFilter"));
GVAR_DEF(PrimitivePtr, kPrimConv2D, std::make_shared<Primitive>("Conv2D"));
GVAR_DEF(PrimitivePtr, kPrimConv3D, std::make_shared<Primitive>("Conv3D"));
GVAR_DEF(PrimitivePtr, kPrimConv2DTranspose, std::make_shared<Primitive>(kConv2DTransposeOpName));
GVAR_DEF(PrimitivePtr, kPrimConv3DTranspose, std::make_shared<Primitive>("Conv3DTranspose"));
GVAR_DEF(PrimitivePtr, kPrimDepthwiseConv2dNative, std::make_shared<Primitive>("DepthwiseConv2dNative"));
GVAR_DEF(PrimitivePtr, kPrimDepthwiseConv2dNativeD, std::make_shared<Primitive>("DepthwiseConv2dNativeD"));
GVAR_DEF(PrimitivePtr, kPrimDepthwiseConv2dNativeBackpropFilter,
         std::make_shared<Primitive>("DepthwiseConv2dNativeBackpropFilter"));
GVAR_DEF(PrimitivePtr, kPrimDepthwiseConv2dNativeBackpropInput,
         std::make_shared<Primitive>("DepthwiseConv2dNativeBackpropInput"));

// Pool
GVAR_DEF(PrimitivePtr, kPrimFractionalMaxPool, std::make_shared<Primitive>("FractionalMaxPool"));
GVAR_DEF(PrimitivePtr, kPrimFractionalMaxPoolGrad, std::make_shared<Primitive>("FractionalMaxPoolGrad"));
GVAR_DEF(PrimitivePtr, kPrimFractionalMaxPool3DWithFixedKsize,
         std::make_shared<Primitive>("FractionalMaxPool3DWithFixedKsize"));
GVAR_DEF(PrimitivePtr, kPrimFractionalMaxPool3DGradWithFixedKsize,
         std::make_shared<Primitive>("FractionalMaxPool3DGradWithFixedKsize"));
GVAR_DEF(PrimitivePtr, kPrimFractionalAvgPool, std::make_shared<Primitive>("FractionalAvgPool"));
GVAR_DEF(PrimitivePtr, kPrimFractionalAvgPoolGrad, std::make_shared<Primitive>("FractionalAvgPoolGrad"));
GVAR_DEF(PrimitivePtr, kPrimAdaptiveMaxPool2DGrad, std::make_shared<Primitive>("AdaptiveMaxPool2DGrad"));
GVAR_DEF(PrimitivePtr, kPrimAdaptiveMaxPool3DGrad, std::make_shared<Primitive>("AdaptiveMaxPool3DGrad"));
GVAR_DEF(PrimitivePtr, kPrimFractionalMaxPoolWithFixedKsize,
         std::make_shared<Primitive>(kFractionalMaxPoolWithFixedKsizeOpName));
GVAR_DEF(PrimitivePtr, kPrimFractionalMaxPoolGradWithFixedKsize,
         std::make_shared<Primitive>(kFractionalMaxPoolGradWithFixedKsizeOpName));
GVAR_DEF(PrimitivePtr, kPrimAdaptiveAvgPool3D, std::make_shared<Primitive>("AdaptiveAvgPool3D"));
GVAR_DEF(PrimitivePtr, kPrimAdaptiveAvgPool3DGrad, std::make_shared<Primitive>("AdaptiveAvgPool3DGrad"));
GVAR_DEF(PrimitivePtr, kPrimAdaptiveAvgPool2D, std::make_shared<Primitive>("AdaptiveAvgPool2D"));
GVAR_DEF(PrimitivePtr, kPrimAdaptiveAvgPool2DGrad, std::make_shared<Primitive>("AdaptiveAvgPool2DGrad"));
GVAR_DEF(PrimitivePtr, kPrimPooling, std::make_shared<Primitive>("Pooling"));
GVAR_DEF(PrimitivePtr, kPrimPoolingGrad, std::make_shared<Primitive>("PoolingGrad"));
GVAR_DEF(PrimitivePtr, kPrimPSROIPooling, std::make_shared<Primitive>("PSROIPooling"));
GVAR_DEF(PrimitivePtr, kPrimPSROIPoolingGrad, std::make_shared<Primitive>("PSROIPoolingGrad"));
GVAR_DEF(PrimitivePtr, kPrimPSROIPoolingV2, std::make_shared<Primitive>("PSROIPoolingV2"));
GVAR_DEF(PrimitivePtr, kPrimPSROIPoolingGradV2D, std::make_shared<Primitive>("PSROIPoolingGradV2D"));
GVAR_DEF(PrimitivePtr, kPrimROIPooling, std::make_shared<Primitive>("ROIPooling"));
GVAR_DEF(PrimitivePtr, kPrimMaxPool, std::make_shared<Primitive>("MaxPool"));
GVAR_DEF(PrimitivePtr, kPrimMaxPool3D, std::make_shared<Primitive>("MaxPool3D"));
GVAR_DEF(PrimitivePtr, kPrimMaxPoolGrad, std::make_shared<Primitive>("MaxPoolGrad"));
GVAR_DEF(PrimitivePtr, kPrimMaxPool3DGrad, std::make_shared<Primitive>("MaxPool3DGrad"));
GVAR_DEF(PrimitivePtr, kPrimMaxPoolV1, std::make_shared<Primitive>("MaxPoolV1"));
GVAR_DEF(PrimitivePtr, kPrimMaxPoolGradV1, std::make_shared<Primitive>("MaxPoolGradV1"));
GVAR_DEF(PrimitivePtr, kPrimMaxPoolGradGrad, std::make_shared<Primitive>("MaxPoolGradGrad"));
GVAR_DEF(PrimitivePtr, kPrimMaxPool3DGradGrad, std::make_shared<Primitive>("MaxPool3DGradGrad"));
GVAR_DEF(PrimitivePtr, kPrimMaxPoolWithArgmax, std::make_shared<Primitive>("MaxPoolWithArgmax"));
GVAR_DEF(PrimitivePtr, kPrimMaxPoolGradWithArgmax, std::make_shared<Primitive>("MaxPoolGradWithArgmax"));
GVAR_DEF(PrimitivePtr, kPrimMaxPoolGradWithArgmaxV2, std::make_shared<Primitive>("MaxPoolGradWithArgmaxV2"));
GVAR_DEF(PrimitivePtr, kPrimMaxPoolGradGradWithArgmax, std::make_shared<Primitive>("MaxPoolGradGradWithArgmax"));
GVAR_DEF(PrimitivePtr, kPrimMaxPoolWithArgmaxV2, std::make_shared<Primitive>("MaxPoolWithArgmaxV2"));
GVAR_DEF(PrimitivePtr, kPrimMaxPool3DWithArgmax, std::make_shared<Primitive>("MaxPool3DWithArgmax"));
GVAR_DEF(PrimitivePtr, kPrimMaxPool3DGradWithArgmax, std::make_shared<Primitive>("MaxPool3DGradWithArgmax"));
GVAR_DEF(PrimitivePtr, kPrimMaxUnpool2D, std::make_shared<Primitive>(kMaxUnpool2DOpName));
GVAR_DEF(PrimitivePtr, kPrimMaxUnpool2DGrad, std::make_shared<Primitive>(kMaxUnpool2DGradOpName));
GVAR_DEF(PrimitivePtr, kPrimMaxUnpool3D, std::make_shared<Primitive>(kMaxUnpool3DOpName));
GVAR_DEF(PrimitivePtr, kPrimMaxUnpool3DGrad, std::make_shared<Primitive>(kMaxUnpool3DGradOpName));
GVAR_DEF(PrimitivePtr, kPrimAvgPool, std::make_shared<Primitive>("AvgPool"));
GVAR_DEF(PrimitivePtr, kPrimAvgPool3D, std::make_shared<Primitive>("AvgPool3D"));
GVAR_DEF(PrimitivePtr, kPrimAvgPool3DD, std::make_shared<Primitive>("AvgPool3DD"));
GVAR_DEF(PrimitivePtr, kPrimAvgPoolGrad, std::make_shared<Primitive>("AvgPoolGrad"));
GVAR_DEF(PrimitivePtr, kPrimAvgPool3DGrad, std::make_shared<Primitive>("AvgPool3DGrad"));
GVAR_DEF(PrimitivePtr, kPrimAvgPool3DGradD, std::make_shared<Primitive>("AvgPool3DGradD"));
GVAR_DEF(PrimitivePtr, kPrimAvgPoolGradVm, std::make_shared<Primitive>("AvgPoolGradVm"));
GVAR_DEF(PrimitivePtr, kPrimAvgPoolGradGe, std::make_shared<Primitive>("AvgPoolGradGe"));
GVAR_DEF(PrimitivePtr, kPrimAvgPoolV1, std::make_shared<Primitive>("AvgPoolV1"));
GVAR_DEF(PrimitivePtr, kPrimAvgPoolGradV1, std::make_shared<Primitive>("AvgPoolGradV1"));
GVAR_DEF(PrimitivePtr, kPrimAdaptiveMaxPool2D, std::make_shared<Primitive>(kAdaptiveMaxPool2DOpName));
GVAR_DEF(PrimitivePtr, kPrimAdaptiveMaxPool2d, std::make_shared<Primitive>("AdaptiveMaxPool2d"));
GVAR_DEF(PrimitivePtr, kPrimAdaptiveMaxPool3D, std::make_shared<Primitive>(kAdaptiveMaxPool3DOpName));
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CORE_BASE_CONV_POOL_OPS_H_
