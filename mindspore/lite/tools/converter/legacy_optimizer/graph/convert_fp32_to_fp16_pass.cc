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

#include "tools/converter/legacy_optimizer/graph/convert_fp32_to_fp16_pass.h"
#include <queue>
#include <vector>
#include "tools/converter/converter_context.h"
#include "src/common/log_adapter.h"
#include "tools/common/graph_util.h"
#include "tools/common/tensor_util.h"
#include "include/errorcode.h"
#include "schema/inner/model_generated.h"
#include "base/float16.h"
#include "src/common/log_util.h"

namespace mindspore {
namespace lite {
namespace {
constexpr int kFp16ToFp32Multiply = 2;
}

STATUS ConvertFP32ToFP16Pass::Run(schema::MetaGraphT *graph) {
  if (!need_convert_) {
    return RET_NO_CHANGE;
  }
  CHECK_NULL_RETURN(graph);
  bool if_changed = false;
  for (auto &tensor : graph->allTensors) {
    if (tensor->dataType != kNumberTypeFloat32 || tensor->data.empty()) {
      continue;
    }
    auto ele_num = lite::GetShapeSize(tensor->dims);
    auto origin_data = tensor->data;
    if (origin_data.size() != ele_num * sizeof(float) || origin_data.size() % kFp16ToFp32Multiply != 0) {
      MS_LOG(ERROR) << "Tensor data length error.";
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
      return RET_ERROR;
    }
    std::vector<uint8_t> new_data(origin_data.size() / kFp16ToFp32Multiply);
    auto fp32_data = reinterpret_cast<float *>(origin_data.data());
    auto fp16_data = reinterpret_cast<float16 *>(new_data.data());
    CHECK_NULL_RETURN(fp32_data);
    CHECK_NULL_RETURN(fp16_data);
    for (size_t i = 0; i < ele_num; i++) {
      fp16_data[i] = float16(fp32_data[i]);
    }
    tensor->data.swap(new_data);
    tensor->dataType = kNumberTypeFloat16;
    new_data.clear();
    if_changed = true;
  }
  return if_changed ? RET_OK : RET_NO_CHANGE;
}
}  // namespace lite
}  // namespace mindspore
