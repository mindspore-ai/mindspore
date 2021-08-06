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

#include "minddata/dataset/audio/kernels/audio_utils.h"

namespace mindspore {
namespace dataset {

template <typename T>
Status AmplitudeToDB(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, T multiplier, T amin,
                     T db_multiplier, T top_db) {
  TensorShape input_shape = input->shape();
  TensorShape to_shape = input_shape.Rank() == 2
                           ? TensorShape({1, 1, input_shape[-2], input_shape[-1]})
                           : TensorShape({input->Size() / (input_shape[-3] * input_shape[-2] * input_shape[-1]),
                                          input_shape[-3], input_shape[-2], input_shape[-1]});
  RETURN_IF_NOT_OK(input->Reshape(to_shape));

  std::vector<T> max_val;
  int step = to_shape[-3] * input_shape[-2] * input_shape[-1];
  int cnt = 0;
  T temp_max = std::numeric_limits<T>::lowest();
  for (auto itr = input->begin<T>(); itr != input->end<T>(); itr++) {
    // do clamp
    *itr = *itr < amin ? log10(amin) * multiplier : log10(*itr) * multiplier;
    *itr -= multiplier * db_multiplier;
    // calculate max by axis
    cnt++;
    if ((*itr) > temp_max) temp_max = *itr;
    if (cnt % step == 0) {
      max_val.push_back(temp_max);
      temp_max = std::numeric_limits<T>::lowest();
    }
  }

  if (!std::isnan(top_db)) {
    int ind = 0;
    for (auto itr = input->begin<T>(); itr != input->end<T>(); itr++, ind++) {
      float lower_bound = max_val[ind / step] - top_db;
      *itr = std::max((*itr), static_cast<T>(lower_bound));
    }
  }
  RETURN_IF_NOT_OK(input->Reshape(input_shape));
  *output = input;
  return Status::OK();
}
template Status AmplitudeToDB<float>(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output,
                                     float multiplier, float amin, float db_multiplier, float top_db);
template Status AmplitudeToDB<double>(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output,
                                      double multiplier, double amin, double db_multiplier, double top_db);
}  // namespace dataset
}  // namespace mindspore
