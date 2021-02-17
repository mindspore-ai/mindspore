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

#include "include/train/accuracy_metrics.h"
#include "include/errorcode.h"
#include "src/common/utils.h"
#include "src/tensor.h"
#include "src/train/train_utils.h"

namespace mindspore {
namespace lite {

AccuracyMetrics::AccuracyMetrics(int accuracy_metrics, const std::vector<int> &input_indexes,
                                 const std::vector<int> &output_indexes)
    : Metrics() {
  if (input_indexes.size() == output_indexes.size()) {
    input_indexes_ = input_indexes;
    output_indexes_ = output_indexes;
  } else {
    MS_LOG(WARNING) << "input to output mapping vectors sizes do not match";
  }
  if (accuracy_metrics != METRICS_CLASSIFICATION) {
    MS_LOG(WARNING) << "Only classification metrics is supported";
  } else {
    accuracy_metrics_ = accuracy_metrics;
  }
}

void AccuracyMetrics::Update(std::vector<tensor::MSTensor *> inputs, std::vector<tensor::MSTensor *> outputs) {
  for (unsigned int i = 0; i < input_indexes_.size(); i++) {
    if ((inputs.size() <= static_cast<unsigned int>(input_indexes_[i])) ||
        (outputs.size() <= static_cast<unsigned int>(output_indexes_[i]))) {
      MS_LOG(WARNING) << "indices " << input_indexes_[i] << "/" << output_indexes_[i]
                      << " is outside of input/output range";
      return;
    }
    float accuracy = 0.0;
    if (inputs.at(input_indexes_[i])->data_type() == kNumberTypeInt32) {
      accuracy = CalculateSparseClassification(inputs.at(input_indexes_[i]), outputs.at(output_indexes_[i]));
    } else {
      accuracy = CalculateOneHotClassification(inputs.at(input_indexes_[i]), outputs.at(output_indexes_[i]));
    }
    total_accuracy_ += accuracy;
    total_steps_ += 1.0;
  }
}

float AccuracyMetrics::Eval() {
  if (total_steps_ == 0.0) {
    MS_LOG(WARNING) << "Accuary can not be calculated, because the number of samples is 0.";
    return 0.0;
  }

  return (total_accuracy_ / total_steps_);
}

}  // namespace lite
}  // namespace mindspore
