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

#include "src/ops/gather.h"
#include "include/errorcode.h"
#include "utils/log_adapter.h"
#include "src/ir/tensor.h"

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int Gather::GetAxis() const { return this->primitive_->value.AsGather()->axis; }
int Gather::GetBatchDims() const { return this->primitive_->value.AsGather()->batchDims; }

void Gather::SetAxis(int axis) { this->primitive_->value.AsGather()->axis = axis; }
void Gather::SetBatchDims(int batch_dims) { this->primitive_->value.AsGather()->batchDims = batch_dims; }

#else

int Gather::GetAxis() const { return this->primitive_->value_as_Gather()->axis(); }
int Gather::GetBatchDims() const { return this->primitive_->value_as_Gather()->batchDims(); }

#endif

int Gather::InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  if (inputs_.size() != kDoubleNum) {
    MS_LOG(ERROR) << "Gather should have two inputs";
    return RET_INPUT_TENSOR_ERROR;
  }
  if (outputs_.size() != kSingleNum) {
    MS_LOG(ERROR) << "Gather should have one outputs";
    return RET_INPUT_TENSOR_ERROR;
  }
  auto input = inputs_.at(0);
  MS_ASSERT(input != nullptr);
  auto indices = inputs_.at(1);
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(input != nullptr);
  output->set_data_type(input->data_type());
  output->SetFormat(input->GetFormat());
  if (!GetInferFlag()) {
    return RET_OK;
  }

  MS_ASSERT(gather_prim != nullptr);
  int axis = GetAxis();
  int batch_dims = GetBatchDims();
  if (axis < 0) {
    axis += input->shape().size();
  }
  auto indices_shape = indices->shape();
  int indices_rank = indices_shape.size();
  if (batch_dims != 0) {
    MS_LOG(ERROR) << "batchDims  " << batch_dims << " != 0, which is not support";
    return RET_ERROR;
  }
  auto in_shape = input->shape();
  int in_rank = in_shape.size();
  if (in_rank < axis + 1) {
    MS_LOG(ERROR) << "input[0]'s rank is less than axis + 1";
    return RET_ERROR;
  }
  std::vector<int> out_shape{in_shape};
  out_shape.erase(out_shape.begin() + axis);
  for (int i = 0; i < indices_rank; i++) {
    out_shape.insert(out_shape.begin() + axis, indices_shape[i]);
  }
  output->set_shape(out_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
