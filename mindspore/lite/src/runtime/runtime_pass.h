/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_RUNTIME_PASS_H_
#define MINDSPORE_LITE_SRC_RUNTIME_RUNTIME_PASS_H_

#ifndef RUNTIME_PASS_CLIP
#include <vector>
#include "src/lite_kernel.h"
#include "src/sub_graph_kernel.h"
#include "schema/ops_generated.h"
#include "schema/model_generated.h"

namespace mindspore::lite {

void RuntimePass(std::vector<kernel::LiteKernel *> *subgraphs, std::vector<Tensor *> *tensors);

/* Nc4hw4 PASS
 * before  : --(nhwc)-- CONV --(nhwc)-- TRANSPOSE --(nchw)-- IN --(nchw)-- TRANSPOSE --(nhwc)--
 * after   : --(nhwc)-- CONV --(nc4hw4)-- IN --(nhwc)--
 * */
static const schema::PrimitiveType Nc4hw4FormatTransposeOp = schema::PrimitiveType_Transpose;
static const std::vector<schema::PrimitiveType> Nc4hw4FormatOutOpList = {schema::PrimitiveType_Conv2DFusion};
static const std::vector<schema::PrimitiveType> Nc4hw4FormatInOpList = {schema::PrimitiveType_InstanceNorm};

/*
 * ConvNormC4 PASS
 * before  : --(nhwc)-- CONV --(nhwc)-- INSTANCENORM --(nhwc)--
 * after   : --(nhwc)-- CONV --(nc4hw4)-- INSTANCENORM --(nhwc)--
 *
 * before  : --(nhwc)-- CONV --(nhwc)-- ACT --(nhwc)-- INSTANCENORM --(nhwc)--
 * after   : --(nhwc)-- CONV --(nc4hw4)-- ACT --(nc4hw4)-- INSTANCENORM --(nhwc)--
 * */
static const schema::PrimitiveType ConvNormC4OpConv2DFusion = schema::PrimitiveType_Conv2DFusion;
static const schema::PrimitiveType ConvNormC4OpActivation = schema::PrimitiveType_Activation;
static const schema::PrimitiveType ConvNormC4OpInstanceNorm = schema::PrimitiveType_InstanceNorm;

}  // namespace mindspore::lite
#endif
#endif  // MINDSPORE_LITE_SRC_RUNTIME_RUNTIME_PASS_H_
