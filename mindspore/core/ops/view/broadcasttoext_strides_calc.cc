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

#include "ops/view/broadcast_to_strides_calc.h"
#include "ops/view/broadcasttoext_strides_calc.h"
#include <memory>
namespace mindspore::ops {
TensorStorageInfoPtrList BroadCastToExtCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
 auto input_tensor = inputs[0]->cast<tensor::TensorPtr>();
 MS_EXCEPTION_IF_NULL(input_tensor);
 auto input_x = GetValue<std::vector<int64_t>>(inputs[1]);
 return BroadCastToProcess(input_tensor, input_x);
}

REG_VIEW_STRIDES_CALC_FUN(BroadcastTo, BroadCastToCalc);
}  // namespace mindspore::ops
