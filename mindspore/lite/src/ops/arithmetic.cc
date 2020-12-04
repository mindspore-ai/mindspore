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

#include "src/ops/arithmetic.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "src/tensor.h"

namespace mindspore {
namespace lite {

int Arithmetic::InferShape(std::vector<lite::Tensor *> inputs_, std::vector<lite::Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  if (inputs_.size() != kDoubleNum) {
    MS_LOG(ERROR) << "The number of input must be " << kDoubleNum;
    return RET_INPUT_TENSOR_ERROR;
  }
  if (outputs_.size() != kSingleNum) {
    MS_LOG(ERROR) << "The number of output must be " << kSingleNum;
    return RET_INPUT_TENSOR_ERROR;
  }
  auto input0 = inputs_[0];
  MS_ASSERT(input0 != nullptr);
  auto input1 = inputs_[1];
  MS_ASSERT(input1 != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);

  auto input_shape0 = input0->shape();
  auto input_shape1 = input1->shape();
  auto format = input0->format();
  output->set_format(format);
  output->set_data_type(input0->data_type());
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  if (input_shape0.size() > 10 || input_shape1.size() > 10) {
    int wrong_dim = input_shape0.size() > input_shape1.size() ? input_shape0.size() : input_shape1.size();
    MS_LOG(ERROR) << "Not support input dim: " << wrong_dim << ", The input dim must be less than 10";
    return RET_ERROR;
  }
  in_shape0_.resize(10);
  in_shape1_.resize(10);
  out_shape_.resize(10);

  ndim_ = input_shape0.size();
  if (input_shape0.size() < input_shape1.size()) {
    ndim_ = input_shape1.size();
    auto fill_dim_num = input_shape1.size() - input_shape0.size();
    int j = 0;
    for (size_t i = 0; i < input_shape1.size(); i++) {
      if (i < fill_dim_num) {
        in_shape0_[i] = 1;
      } else {
        in_shape0_[i] = input_shape0[j++];
      }
      in_shape1_[i] = input_shape1[i];
    }
    format = input0->format();
  } else if (input_shape0.size() > input_shape1.size()) {
    ndim_ = input_shape0.size();
    auto fill_dim_num = input_shape0.size() - input_shape1.size();
    int j = 0;
    for (size_t i = 0; i < input_shape0.size(); i++) {
      if (i < fill_dim_num) {
        in_shape1_[i] = 1;
      } else {
        in_shape1_[i] = input_shape1[j++];
      }
      in_shape0_[i] = input_shape0[i];
    }
  } else {
    for (size_t i = 0; i < input_shape0.size(); i++) {
      in_shape1_[i] = input_shape1[i];
      in_shape0_[i] = input_shape0[i];
    }
  }

  std::vector<int> output_shape;
  for (int i = 0; i < ndim_; i++) {
    if (in_shape0_[i] != in_shape1_[i]) {
      if (in_shape0_[i] == 1) {
        out_shape_[i] = in_shape1_[i];
      } else if (in_shape1_[i] == 1) {
        out_shape_[i] = in_shape0_[i];
      } else {
        MS_LOG(ERROR) << "shapes of input tensors can not be broadCasted";
        return -1;
      }
      broadcasting_ = true;
    } else {
      out_shape_[i] = in_shape0_[i];
    }
    output_shape.push_back(out_shape_[i]);
  }

  output->set_shape(output_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
