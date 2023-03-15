/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_OPTIMIZER_REG_CPU_CONST_INPUT_TO_ATTR_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_OPTIMIZER_REG_CPU_CONST_INPUT_TO_ATTR_H_

#include "include/backend/optimizer/op_adaptation_info_factory.h"

// Do not add operators here, the input to attribute function has been abandoned
namespace mindspore::opt {
#define RER_CPU_STATIC_CONST_TO_ATTR(op_name, ...) RER_CONST_TO_ATTR_LIST(op_name, kCPUDevice, false, __VA_ARGS__)
#define RER_CPU_DYNAMIC_CONST_TO_ATTR(op_name, ...) RER_CONST_TO_ATTR_LIST(op_name, kCPUDevice, true, __VA_ARGS__)

RER_CPU_DYNAMIC_CONST_TO_ATTR(kCastOpName, 1);
RER_CPU_DYNAMIC_CONST_TO_ATTR(kFillOpName, 0);
RER_CPU_DYNAMIC_CONST_TO_ATTR(kScalarToTensorOpName, 1);
RER_CPU_DYNAMIC_CONST_TO_ATTR(kTupleToTensorOpName, 1);
RER_CPU_DYNAMIC_CONST_TO_ATTR(kListToTensorOpName, 1);
RER_CPU_DYNAMIC_CONST_TO_ATTR(kScalarCastOpName, 1);

RER_CPU_STATIC_CONST_TO_ATTR(kCastOpName, 1);
RER_CPU_STATIC_CONST_TO_ATTR(kCOO2CSROpName, 1);
RER_CPU_STATIC_CONST_TO_ATTR(kCSR2COOOpName, 1);
RER_CPU_STATIC_CONST_TO_ATTR(kCSRDivOpName, 3);
RER_CPU_STATIC_CONST_TO_ATTR(kCSRGatherOpName, 3);
RER_CPU_STATIC_CONST_TO_ATTR(kCSRMMOpName, 3);
RER_CPU_STATIC_CONST_TO_ATTR(kCSRMulOpName, 3);
RER_CPU_STATIC_CONST_TO_ATTR(kCSRMVOpName, 3);
RER_CPU_STATIC_CONST_TO_ATTR(kCSRReduceSumOpName, 3, 4);
RER_CPU_STATIC_CONST_TO_ATTR(kFillOpName, 0);
RER_CPU_STATIC_CONST_TO_ATTR(kScalarToTensorOpName, 1);
RER_CPU_STATIC_CONST_TO_ATTR(kTupleToTensorOpName, 1);
RER_CPU_STATIC_CONST_TO_ATTR(kListToTensorOpName, 1);
RER_CPU_STATIC_CONST_TO_ATTR(kScalarCastOpName, 1);
}  // namespace mindspore::opt

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_OPTIMIZER_REG_CPU_CONST_INPUT_TO_ATTR_H_
