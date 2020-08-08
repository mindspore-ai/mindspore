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
#include "src/common/common.h"

namespace mindspore::lite {
int Nhwc2Nchw::InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) {
  MS_ASSERT(this->primitive != nullptr);
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  std::vector<int> nhwc_shape = input->shape();
  if (nhwc_shape.size() != 4) {
    output->set_shape(nhwc_shape);
  } else {
    std::vector<int> nchw_shape{nhwc_shape};
    nchw_shape[NCHW_N] = nhwc_shape[NHWC_N];
    nchw_shape[NCHW_C] = nhwc_shape[NHWC_C];
    nchw_shape[NCHW_H] = nhwc_shape[NHWC_H];
    nchw_shape[NCHW_W] = nhwc_shape[NHWC_W];
    output->set_shape(nchw_shape);
  }
  output->SetFormat(schema::Format_NCHW);
  output->set_data_type(input->data_type());
  return RET_OK;
}
}  // namespace mindspore::lite
