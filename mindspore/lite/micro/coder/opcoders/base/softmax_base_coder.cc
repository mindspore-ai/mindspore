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
#include "coder/opcoders/base/softmax_base_coder.h"
#include <vector>
#include <type_traits>

namespace mindspore::lite::micro {

int SoftmaxBaseCoder::Init() {
  this->softmax_param_ = reinterpret_cast<SoftmaxParameter *>(parameter_);
  std::vector<int> in_shape = input_tensor_->shape();
  size_t in_dims = in_shape.size();
  MS_CHECK_TRUE(in_dims < std::extent<decltype(softmax_param_->input_shape_)>::value,
                "in_dims should be less than input_shape_ size");
  int ele_size = 1;
  softmax_param_->n_dim_ = in_dims;
  for (int i = 0; i < static_cast<int>(in_dims); i++) {
    softmax_param_->input_shape_[i] = in_shape.at(i);
    ele_size *= in_shape.at(i);
  }
  softmax_param_->element_size_ = ele_size;
  return RET_OK;
}

int SoftmaxBaseCoder::ReSize() {
  std::vector<int> in_shape = input_tensor_->shape();
  size_t in_dims = in_shape.size();
  MS_CHECK_TRUE(in_dims < std::extent<decltype(softmax_param_->input_shape_)>::value,
                "in_dims should be less than input_shape_ size");
  int ele_size = 1;
  softmax_param_->n_dim_ = in_dims;
  if (softmax_param_->axis_ == -1) {
    softmax_param_->axis_ += in_dims;
  }
  for (size_t i = 0; i < in_dims; i++) {
    softmax_param_->input_shape_[i] = in_shape.at(i);
    ele_size *= in_shape.at(i);
  }
  softmax_param_->element_size_ = ele_size;
  return RET_OK;
}

}  // namespace mindspore::lite::micro
