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
constexpr int kShapeInputNum = 1;
constexpr int kShapeOutputNum = 1;

}  // namespace
int Shape::InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) {
  if (inputs_.size() != kShapeInputNum) {
    MS_LOG(ERROR) << "inputs to Shape operator should be 1, but " << inputs_.size() << " is given.";
    return RET_ERROR;
  }
  if (outputs_.size() != kShapeOutputNum) {
    MS_LOG(ERROR) << "outputs to Shape operator should be 1, but " << outputs_.size() << " is given.";
    return RET_ERROR;
  }

  auto in_tensor = inputs_.front();
  auto out_tensor = outputs_.front();
  std::vector<int> out_shape;
  out_shape.push_back(static_cast<int>(in_tensor->shape().size()));

  auto ret_shape = out_tensor->set_shape(out_shape);
  if (ret_shape != 1 || size_t(out_tensor->shape()[0]) != in_tensor->shape().size()) {
    MS_LOG(ERROR) << "Set shape fails.";
    return RET_ERROR;
  }
  auto ret_dtype = out_tensor->set_data_type(in_tensor->data_type());
  if (ret_dtype != in_tensor->data_type()) {
    MS_LOG(ERROR) << "Set datatype fails.";
    return RET_ERROR;
  }

  // todo
  // auto ret_data = out_tensor->MallocData();
  // if (ret_data != 0) {
  //   MS_LOG(ERROR) << "Allocate memory fails.";
  //   return RET_ERROR;
  // }

  return RET_OK;
}
}  // namespace mindspore::lite

