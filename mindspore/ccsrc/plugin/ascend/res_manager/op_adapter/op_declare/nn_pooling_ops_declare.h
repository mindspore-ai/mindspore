/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NN_POOLING_OPS_DECLARE_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NN_POOLING_OPS_DECLARE_H_

#include "include/utils/utils.h"
#include "plugin/ascend/res_manager/op_adapter/custom_op_proto/cust_nn_ops.h"
#include "plugin/ascend/res_manager/op_adapter/op_declare/op_declare_macro.h"

DECLARE_OP_ADAPTER(MaxPoolWithArgmax)
DECLARE_OP_USE_OUTPUT(MaxPoolWithArgmax)

DECLARE_OP_ADAPTER(MaxPoolWithArgmaxV2)
DECLARE_OP_USE_OUTPUT(MaxPoolWithArgmaxV2)

DECLARE_OP_ADAPTER(MaxPoolGradWithArgmax)
DECLARE_OP_USE_OUTPUT(MaxPoolGradWithArgmax)

DECLARE_OP_ADAPTER(MaxPoolGradWithArgmaxV2)
DECLARE_OP_USE_OUTPUT(MaxPoolGradWithArgmaxV2)

DECLARE_OP_ADAPTER(MaxPoolGradGradWithArgmax)
DECLARE_OP_USE_OUTPUT(MaxPoolGradGradWithArgmax)

DECLARE_OP_ADAPTER(MaxPool)
DECLARE_OP_USE_OUTPUT(MaxPool)

DECLARE_OP_ADAPTER(MaxPoolGrad)
DECLARE_OP_USE_OUTPUT(MaxPoolGrad)

DECLARE_OP_ADAPTER(MaxPoolGradGrad)
DECLARE_OP_USE_OUTPUT(MaxPoolGradGrad)

DECLARE_OP_ADAPTER(MaxPool3D)
DECLARE_OP_USE_OUTPUT(MaxPool3D)

DECLARE_OP_ADAPTER(MaxPool3DWithArgmax)
DECLARE_OP_USE_OUTPUT(MaxPool3DWithArgmax)

DECLARE_OP_ADAPTER(MaxPool3DGrad)
DECLARE_OP_USE_OUTPUT(MaxPool3DGrad)

DECLARE_OP_ADAPTER(MaxPool3DGradGrad)
DECLARE_OP_USE_OUTPUT(MaxPool3DGradGrad)

DECLARE_OP_ADAPTER(AvgPool)
DECLARE_OP_USE_OUTPUT(AvgPool)

DECLARE_OP_ADAPTER(AdaptiveMaxPool2d)
DECLARE_OP_USE_OUTPUT(AdaptiveMaxPool2d)

DECLARE_OP_ADAPTER(AvgPool3D)
DECLARE_OP_USE_OUTPUT(AvgPool3D)

DECLARE_OP_ADAPTER(AvgPool3DD)
DECLARE_OP_USE_OUTPUT(AvgPool3DD)

DECLARE_OP_ADAPTER(AvgPoolGrad)
DECLARE_OP_USE_OUTPUT(AvgPoolGrad)

DECLARE_OP_ADAPTER(Pooling)
DECLARE_OP_USE_OUTPUT(Pooling)

DECLARE_OP_ADAPTER(MaxPoolV3)
DECLARE_OP_USE_OUTPUT(MaxPoolV3)

DECLARE_OP_ADAPTER(AvgPoolV2)
DECLARE_OP_USE_OUTPUT(AvgPoolV2)

DECLARE_OP_ADAPTER(GlobalAveragePool)
DECLARE_OP_USE_OUTPUT(GlobalAveragePool)

DECLARE_OP_ADAPTER(Upsample)
DECLARE_OP_USE_OUTPUT(Upsample)

DECLARE_OP_ADAPTER(AvgPool3DGrad)
DECLARE_OP_USE_OUTPUT(AvgPool3DGrad)

DECLARE_OP_ADAPTER(Dilation2DBackpropFilter)
DECLARE_OP_USE_OUTPUT(Dilation2DBackpropFilter)

DECLARE_OP_ADAPTER(Dilation2DBackpropInput)
DECLARE_OP_USE_OUTPUT(Dilation2DBackpropInput)

DECLARE_CUST_OP_ADAPTER(AdaptiveAvgPool3dGrad)
DECLARE_CUST_OP_USE_OUTPUT(AdaptiveAvgPool3dGrad)

DECLARE_CUST_OP_ADAPTER(AdaptiveMaxPool2dGrad)
DECLARE_CUST_OP_USE_OUTPUT(AdaptiveMaxPool2dGrad)

// hotfix only for lite
DECLARE_OP_ADAPTER(AdaptiveAvgPool2d)
DECLARE_OP_USE_OUTPUT(AdaptiveAvgPool2d)

DECLARE_CUST_OP_ADAPTER(AdaptiveAvgPool2D)
DECLARE_CUST_OP_USE_OUTPUT(AdaptiveAvgPool2D)

DECLARE_CUST_OP_ADAPTER(AdaptiveAvgPool3d)
DECLARE_CUST_OP_USE_OUTPUT(AdaptiveAvgPool3d)

DECLARE_CUST_OP_ADAPTER(AdaptiveAvgPool2DGrad)
DECLARE_CUST_OP_USE_OUTPUT(AdaptiveAvgPool2DGrad)

DECLARE_OP_ADAPTER(DataFormatVecPermute)
DECLARE_OP_USE_OUTPUT(DataFormatVecPermute)

DECLARE_CUST_OP_ADAPTER(MaxPool3DGradWithArgmax)
DECLARE_CUST_OP_USE_OUTPUT(MaxPool3DGradWithArgmax)

DECLARE_OP_ADAPTER(FractionalAvgPoolGrad)
DECLARE_OP_USE_OUTPUT(FractionalAvgPoolGrad)

DECLARE_OP_ADAPTER(FractionalMaxPool)
DECLARE_OP_USE_OUTPUT(FractionalMaxPool)

DECLARE_OP_ADAPTER(FractionalMaxPoolGrad)
DECLARE_OP_USE_OUTPUT(FractionalMaxPoolGrad)

DECLARE_OP_ADAPTER(FractionalAvgPool)
DECLARE_OP_USE_OUTPUT(FractionalAvgPool)

DECLARE_OP_ADAPTER(NthElement)
DECLARE_OP_USE_OUTPUT(NthElement)

DECLARE_OP_ADAPTER(AdaptiveAvgPool)
DECLARE_OP_USE_OUTPUT(AdaptiveAvgPool)

DECLARE_CUST_OP_ADAPTER(MaxUnpool2D)
DECLARE_CUST_OP_USE_OUTPUT(MaxUnpool2D)

DECLARE_CUST_OP_ADAPTER(MaxUnpool3D)
DECLARE_CUST_OP_USE_OUTPUT(MaxUnpool3D)

DECLARE_CUST_OP_ADAPTER(MaxUnpool3DGrad)
DECLARE_CUST_OP_USE_OUTPUT(MaxUnpool3DGrad)

DECLARE_CUST_OP_ADAPTER(MaxUnpool2DGrad)
DECLARE_CUST_OP_USE_OUTPUT(MaxUnpool2DGrad)

DECLARE_CUST_OP_ADAPTER(FractionalMaxPool3DWithFixedKsize)
DECLARE_CUST_OP_USE_OUTPUT(FractionalMaxPool3DWithFixedKsize)

DECLARE_CUST_OP_ADAPTER(FractionalMaxPool3DGradWithFixedKsize)
DECLARE_CUST_OP_USE_OUTPUT(FractionalMaxPool3DGradWithFixedKsize)

DECLARE_CUST_OP_ADAPTER(MaxPool3DWithArgmax)
DECLARE_CUST_OP_USE_OUTPUT(MaxPool3DWithArgmax)
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NN_POOLING_OPS_DECLARE_H_
