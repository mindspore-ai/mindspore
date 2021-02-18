/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "src/train/train_utils.h"
#include <vector>
#include "include/errorcode.h"
#include "include/ms_tensor.h"
#include "src/common/utils.h"

namespace mindspore {
namespace lite {

float CalculateSparseClassification(tensor::MSTensor *input, tensor::MSTensor *output) {
  if ((input->shape().size() != 1) || (input->data_type() != kNumberTypeInt32) || (output->shape().size() != 2)) {
    MS_LOG(WARNING) << "SparceClassification got a " << input->shape() << "-D input tensor, " << output->shape()
                    << "-D output tensor";
    return 0.0;
  }

  int batch_size = input->shape().at(0);
  int num_of_classes = output->shape().at(1);
  auto labels = reinterpret_cast<int *>(input->MutableData());
  auto predictions = reinterpret_cast<float *>(output->MutableData());
  float accuracy = 0.0;
  for (int b = 0; b < batch_size; b++) {
    int max_idx = 0;
    float max_score = predictions[num_of_classes * b];
    for (int c = 1; c < num_of_classes; c++) {
      if (predictions[num_of_classes * b + c] > max_score) {
        max_score = predictions[num_of_classes * b + c];
        max_idx = c;
      }
    }
    if (labels[b] == max_idx) accuracy += 1.0;
  }
  return accuracy / (static_cast<float>(batch_size));
}

float CalculateOneHotClassification(tensor::MSTensor *input, tensor::MSTensor *output) {
  if ((input->shape().size() != 2) || (output->shape().size() != 2)) {
    MS_LOG(WARNING) << "OneHotClassification got a " << input->shape() << "-D input tensor, " << output->shape()
                    << "-D output tensor";
    return 0.0;
  }

  int batch_size = input->shape().at(0);
  int num_of_classes = input->shape().at(1);
  auto labels = reinterpret_cast<float *>(input->MutableData());
  auto predictions = reinterpret_cast<float *>(output->MutableData());
  float accuracy = 0.0;
  for (int b = 0; b < batch_size; b++) {
    int label = 0;
    int max_idx = 0;
    float max_label_score = labels[num_of_classes * b];
    float max_score = predictions[num_of_classes * b];
    for (int c = 1; c < num_of_classes; c++) {
      if (predictions[num_of_classes * b + c] > max_score) {
        max_score = predictions[num_of_classes * b + c];
        max_idx = c;
      }
      if (labels[num_of_classes * b + c] > max_label_score) {
        max_label_score = labels[num_of_classes * b + c];
        label = c;
      }
    }
    if (label == max_idx) accuracy += 1.0;
  }
  return accuracy / (static_cast<float>(batch_size));
}

}  // namespace lite
}  // namespace mindspore
