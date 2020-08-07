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
#include "src/ir/tensor.h"
#include "utils/log_adapter.h"

namespace mindspore::lite {
int EmbeddingLookup::InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) {
  MS_ASSERT(this->primitive != nullptr);
  if (inputs_.size() < kDoubleNum) {
    MS_LOG(ERROR) << "Embedding Lookup should have at least two inputs";
    return RET_INPUT_TENSOR_ERROR;
  }

  if (outputs_.size() != kSingleNum) {
    MS_LOG(ERROR) << "Embedding Lookup should have one outputs";
    return RET_INPUT_TENSOR_ERROR;
  }

  auto params_ = inputs_.front();
  MS_ASSERT(params_ != nullptr);
  auto ids = inputs_.back();
  MS_ASSERT(ids != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);

  auto embedding_shape = params_->shape();
  embedding_shape.erase(embedding_shape.begin());

  std::vector<int> output_shape(ids->shape());
  for (size_t i = 0; i < embedding_shape.size(); ++i) {
    output_shape.push_back(embedding_shape.at(i));
  }

  for (int i = 1; i < inputs_.size() - 1; ++i) {
    auto embedding_shape_t = inputs_.at(i)->shape();
    embedding_shape_t.erase(embedding_shape_t.begin());
    if (embedding_shape_t != embedding_shape) {
      MS_LOG(ERROR) << "The embedded layers should have the same shape";
      return RET_INPUT_TENSOR_ERROR;
    }
  }

  output->set_shape(output_shape);
  output->set_data_type(params_->data_type());
  return RET_OK;
}
}  // namespace mindspore::lite
