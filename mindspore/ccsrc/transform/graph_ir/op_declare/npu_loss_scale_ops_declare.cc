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

#include "transform/graph_ir/op_declare/npu_loss_scale_ops_declare.h"

namespace mindspore::transform {
// NPUGetFloatStatus
INPUT_MAP(NPUGetFloatStatus) = {{1, INPUT_DESC(addr)}};
OUTPUT_MAP(NPUGetFloatStatus) = {{0, OUTPUT_DESC(data)}};
ATTR_MAP(NPUGetFloatStatus) = EMPTY_ATTR_MAP;
REG_ADPT_DESC(NPUGetFloatStatus, kNameNPUGetFloatStatus, ADPT_DESC(NPUGetFloatStatus))

// NPUAllocFloatStatus
INPUT_MAP(NPUAllocFloatStatus) = EMPTY_INPUT_MAP;
ATTR_MAP(NPUAllocFloatStatus) = EMPTY_ATTR_MAP;
OUTPUT_MAP(NPUAllocFloatStatus) = {{0, OUTPUT_DESC(data)}};
REG_ADPT_DESC(NPUAllocFloatStatus, kNameNPUAllocFloatStatus, ADPT_DESC(NPUAllocFloatStatus))

// NPUClearFloatStatus
INPUT_MAP(NPUClearFloatStatus) = {{1, INPUT_DESC(addr)}};
OUTPUT_MAP(NPUClearFloatStatus) = {{0, OUTPUT_DESC(data)}};
ATTR_MAP(NPUClearFloatStatus) = EMPTY_ATTR_MAP;
REG_ADPT_DESC(NPUClearFloatStatus, kNameNPUClearFloatStatus, ADPT_DESC(NPUClearFloatStatus))

// NPUGetFloatStatusV2
INPUT_MAP(NPUGetFloatStatusV2) = EMPTY_INPUT_MAP;
OUTPUT_MAP(NPUGetFloatStatusV2) = {{0, OUTPUT_DESC(data)}};
ATTR_MAP(NPUGetFloatStatusV2) = EMPTY_ATTR_MAP;
REG_ADPT_DESC(NPUGetFloatStatusV2, kNameNPUGetFloatStatusV2, ADPT_DESC(NPUGetFloatStatusV2))

// NPUClearFloatStatusV2
INPUT_MAP(NPUClearFloatStatusV2) = EMPTY_INPUT_MAP;
OUTPUT_MAP(NPUClearFloatStatusV2) = EMPTY_OUTPUT_MAP;
ATTR_MAP(NPUClearFloatStatusV2) = EMPTY_ATTR_MAP;
REG_ADPT_DESC(NPUClearFloatStatusV2, kNameNPUClearFloatStatusV2, ADPT_DESC(NPUClearFloatStatusV2))
}  // namespace mindspore::transform
