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
int GatherNd::InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) {
  MS_ASSERT(this->primitive != nullptr);
  if (inputs_.size() != kDoubleNum) {
    MS_LOG(ERROR) << "GatherNd should have two inputs";
    return RET_INPUT_TENSOR_ERROR;
  }
  if (outputs_.size() != kSingleNum) {
    MS_LOG(ERROR) << "GatherNd should have one outputs";
    return RET_INPUT_TENSOR_ERROR;
  }

  auto input = inputs_.at(0);
  MS_ASSERT(input != nullptr);
  auto indices = inputs_.at(1);
  MS_ASSERT(indices != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);

  auto in_shape = input->shape();
  int in_rank = in_shape.size();
  auto indices_shape = indices->shape();
  int indices_rank = indices_shape.size();

  if (indices_shape[indices_rank - 1] > in_rank) {
    MS_LOG(ERROR) << "Input of indices data is error!";
    return RET_ERROR;
  }

  std::vector<int> out_shape;
  int i = 0;
  for (i = 0; i < indices_rank - 1; ++i) {
    out_shape.emplace_back(indices_shape[i]);
  }
  for (i = indices_shape[indices_rank - 1]; i < in_rank; ++i) {
    out_shape.emplace_back(in_shape[i]);
  }

  output->set_shape(out_shape);
  output->set_data_type(input->data_type());
  output->SetFormat(input->GetFormat());

  return RET_OK;
}
}  // namespace mindspore::lite
