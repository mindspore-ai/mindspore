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
#include <cstddef>
#include <string>
#include <vector>
#include <memory>
#include "include/train/accuracy_metrics.h"
#include "include/api/metrics/accuracy.h"
#include "src/cxx_api/metrics/metrics_impl.h"
#include "src/common/log_adapter.h"

namespace mindspore {
AccuracyMetrics::AccuracyMetrics(int accuracy_metrics, const std::vector<int> &input_indexes,
                                 const std::vector<int> &output_indexes) {
  metrics_impl_ = new (std::nothrow)
    MetricsImpl(new (std::nothrow) lite::AccuracyMetrics(accuracy_metrics, input_indexes, output_indexes));
  if (metrics_impl_ == nullptr) {
    MS_LOG(ERROR) << "Metrics implement new failed";
  }
}

AccuracyMetrics::~AccuracyMetrics() {
  if (metrics_impl_ == nullptr) {
    MS_LOG(ERROR) << "Metrics implement is null.";
    return;
  }
  auto internal_metrics = metrics_impl_->GetInternalMetrics();
  if (internal_metrics != nullptr) {
    delete internal_metrics;
  }
  delete metrics_impl_;
  metrics_impl_ = nullptr;
}

void AccuracyMetrics::Clear() {
  if (metrics_impl_ == nullptr) {
    MS_LOG(ERROR) << "Metrics implement is null.";
    return;
  }
  auto internal_metrics = metrics_impl_->GetInternalMetrics();
  (reinterpret_cast<lite::AccuracyMetrics *>(internal_metrics))->Clear();
}

float AccuracyMetrics::Eval() {
  if (metrics_impl_ == nullptr) {
    MS_LOG(ERROR) << "Metrics implement is null.";
    return 0.0f;
  }
  auto internal_metrics = metrics_impl_->GetInternalMetrics();
  return (reinterpret_cast<lite::AccuracyMetrics *>(internal_metrics))->Eval();
}
}  // namespace mindspore
