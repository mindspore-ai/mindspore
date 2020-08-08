/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "src/ops/ops.h"
#include "include/errorcode.h"
#include "utils/log_adapter.h"
#include "src/ir/tensor.h"

namespace mindspore::lite {
namespace {
constexpr int kSliceInputNum = 1;
constexpr int kSliceOutputNum = 1;
}  // namespace

int Slice::InferShape(std::vector<tensor::Tensor *> inputs, std::vector<tensor::Tensor *> outputs) {
  MS_ASSERT(this->primitive != nullptr);
  if (inputs.size() != kSliceInputNum || outputs.size() != kSliceOutputNum) {
    MS_LOG(ERROR) << "input size:" << inputs.size() << ",output size:" << outputs.size();
    return RET_PARAM_INVALID;
  }
  auto input = inputs.at(0);
  auto input_shape = input->shape();
  auto slice_prim = this->primitive->value_as_Slice();
  std::vector<int32_t> slice_begin(slice_prim->begin()->begin(), slice_prim->begin()->end());
  std::vector<int32_t> slice_size(slice_prim->size()->begin(), slice_prim->size()->end());
  std::vector<int32_t> output_shape(input_shape.size());
  for (int i = 0; i < input_shape.size(); ++i) {
    if (slice_size[i] < 0 && slice_size[i] != -1) {
      MS_LOG(ERROR) << "Invalid size input!size[" << i << "]=" << slice_size[i];
      return RET_PARAM_INVALID;
    }
    if (slice_begin[i] < 0) {
      MS_LOG(ERROR) << "Invalid begin input " << slice_begin[i] << " which should be >= 0";
      return RET_PARAM_INVALID;
    }
    if (input_shape[i] <= slice_begin[i]) {
      MS_LOG(ERROR) << "Invalid begin input!begin[" << i << "]=" << slice_begin[i]
                    << " which should be <= " << input_shape[i];
      return RET_PARAM_INVALID;
    }
    if (slice_size[i] > (input_shape[i] - slice_begin[i])) {
      MS_LOG(ERROR) << "Invalid size input " << slice_size[i]
                    << " which should be <= " << input_shape[i] - slice_begin[i];
      return RET_PARAM_INVALID;
    }

    output_shape[i] = slice_size[i] < 0 ? input_shape[i] - slice_begin[i] : slice_size[i];
  }

  outputs[0]->set_shape(output_shape);
  outputs[0]->set_data_type(input->data_type());
  outputs[0]->SetFormat(input->GetFormat());

  return RET_OK;
}
}  // namespace mindspore::lite
