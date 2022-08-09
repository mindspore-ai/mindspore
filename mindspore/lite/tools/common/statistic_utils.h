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

#ifndef MINDSPORE_LITE_TOOLS_COMMON_STATISTIC_UTILS_H_
#define MINDSPORE_LITE_TOOLS_COMMON_STATISTIC_UTILS_H_

#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cfloat>
#include <utility>
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "nnacl/op_base.h"
#include "mindapi/base/type_id.h"

namespace mindspore::lite {
std::pair<float, float> GetFloatMinMaxValue(const float *data, int size);

template <typename T>
std::pair<T, T> GetMinMaxValue(const T *data, size_t data_size) {
  MS_ASSERT(data != nullptr);
  MS_ASSERT(data_size > 1);
  T min = data[0];
  T max = data[0];
  for (size_t i = 1; i < data_size; i++) {
    min = std::min(min, data[i]);
    max = std::max(max, data[i]);
  }
  return {min, max};
}

template <typename T>
T GetMinValue(const std::vector<T> &data_vector) {
  MS_ASSERT(!data_vector.empty());
  return *min_element(data_vector.begin(), data_vector.end());
}

template <typename T>
T GetMaxValue(const std::vector<T> &data_vector) {
  MS_ASSERT(!data_vector.empty());
  return *max_element(data_vector.begin(), data_vector.end());
}

template <typename T>
float Quantile(const std::vector<T> &data_vector, float q) {
  MS_ASSERT(q >= 0.0f && q <= 1.0f);
  std::vector<T> bak_data(data_vector);
  std::sort(bak_data.begin(), bak_data.end());
  const int n = bak_data.size();
  float id = (n - 1) * q;
  int lo = std::floor(id);
  int hi = std::ceil(id);
  float qs = bak_data.at(lo);
  float h = (id - lo);
  return (1.0 - h) * qs + h * bak_data.at(hi);
}

template <typename T>
float GetMeanValue(const std::vector<T> &data_vector) {
  MS_ASSERT(!data_vector.empty());
  float sum = std::accumulate(std::begin(data_vector), std::end(data_vector), 0.0);
  float mean = sum / data_vector.size();
  return mean;
}

template <typename T>
std::pair<float, float> GetMeanVar(const std::vector<T> &data_vector) {
  MS_ASSERT(!data_vector.empty());
  float mean = GetMeanValue(data_vector);
  float accumulate = 0.0;
  std::for_each(std::begin(data_vector), std::end(data_vector),
                [&](const float data) { accumulate += (data - mean) * (data - mean); });
  float var = sqrt(accumulate / data_vector.size());
  return {mean, var};
}

template <typename T>
float GetVarValue(const std::vector<T> &data_vector) {
  MS_ASSERT(!data_vector.empty());
  float mean = GetMeanValue(data_vector);
  float accumulate = 0.0;
  std::for_each(std::begin(data_vector), std::end(data_vector),
                [&](const float data) { accumulate += (data - mean) * (data - mean); });
  float var = sqrt(accumulate / data_vector.size());
  return var;
}

template <typename T>
float GetSparsity(const std::vector<T> &data_vector) {
  MS_ASSERT(!data_vector.empty());
  auto zero_nums = std::count(data_vector.begin(), data_vector.end(), 0);
  return 1.0 * zero_nums / data_vector.size();
}

template <typename T>
float GetClipRate(const T *origin, const T *compared, size_t size) {
  MS_ASSERT(origin != nullptr);
  MS_ASSERT(compared != nullptr);
  MS_ASSERT(size > 0);
  auto min = *std::min_element(compared, compared + size);
  auto max = *std::max_element(compared, compared + size);
  size_t total = 0;
  for (size_t i = 0; i < size; ++i) {
    if (origin[i] > max || origin[i] < min) {
      total++;
    }
  }
  return 1.0f * total / size;
}

inline float GetClipRate(const void *vector_a, const void *vector_b, size_t size, mindspore::TypeId type_id) {
  MS_ASSERT(vector_a != nullptr);
  MS_ASSERT(vector_b != nullptr);
  if (type_id == mindspore::kNumberTypeFloat32) {
    return mindspore::lite::GetClipRate<float>(static_cast<const float *>(vector_a),
                                               static_cast<const float *>(vector_b), size);
  } else if (type_id == mindspore::kNumberTypeInt32) {
    return mindspore::lite::GetClipRate(static_cast<const int *>(vector_a), static_cast<const int *>(vector_b), size);
  } else {
    MS_LOG(ERROR) << "Unsupported data type:" << type_id;
    return 0;
  }
}

template <typename T>
float GetCosSimilarity(const T *vector_a, const T *vector_b, size_t size) {
  MS_ASSERT(vector_a != nullptr);
  MS_ASSERT(vector_b != nullptr);
  double dot_sum = 0;
  double sum_a = 0;
  double sum_b = 0;
  for (size_t i = 0; i < size; i++) {
    if (std::is_same<T, float>::value && ((std::isnan(vector_a[i]) || std::isinf(vector_a[i])) ||
                                          (std::isnan(vector_b[i]) || std::isinf(vector_b[i])))) {
      MS_LOG(ERROR) << "tensor has nan or inf data, compare fail";
      return 0;
    }
    dot_sum += static_cast<double>(vector_a[i]) * static_cast<double>(vector_b[i]);
    sum_a += static_cast<double>(vector_a[i]) * static_cast<double>(vector_a[i]);
    sum_b += static_cast<double>(vector_b[i]) * static_cast<double>(vector_b[i]);
  }
  if (sum_a < DBL_EPSILON && sum_b < DBL_EPSILON) {
    return 1;
  } else if (sum_a * sum_b < DBL_EPSILON) {
    return 0;
  }
  return dot_sum / (std::sqrt(sum_a) * std::sqrt(sum_b));
}

inline float GetCosSimilarity(const void *vector_a, const void *vector_b, size_t size, mindspore::TypeId type_id) {
  MS_ASSERT(vector_a != nullptr);
  MS_ASSERT(vector_b != nullptr);
  if (type_id == mindspore::kNumberTypeFloat32) {
    return mindspore::lite::GetCosSimilarity<float>(static_cast<const float *>(vector_a),
                                                    static_cast<const float *>(vector_b), size);
  } else if (type_id == mindspore::kNumberTypeInt32) {
    return mindspore::lite::GetCosSimilarity(static_cast<const int *>(vector_a), static_cast<const int *>(vector_b),
                                             size);
  } else {
    MS_LOG(ERROR) << "Unsupported data type:" << type_id;
    return 0;
  }
}

template <typename T>
float KLDivergence(std::vector<T> p, std::vector<T> q) {
  auto sum = 0.0f;
  std::for_each(p.begin(), p.end(), [&sum](T item) { sum += item; });
  MS_ASSERT(sum > DBL_EPSILON);
  std::for_each(p.begin(), p.end(), [sum](T &item) { item /= sum; });
  sum = 0.0f;
  std::for_each(q.begin(), q.end(), [&sum](T item) { sum += item; });
  MS_ASSERT(sum > DBL_EPSILON);
  std::for_each(q.begin(), q.end(), [sum](T &item) { item /= sum; });

  float result = 0.0f;
  const size_t size = p.size();
  for (size_t i = 0; i < size; ++i) {
    if (p[i] != 0) {
      if (q[i] == 0) {
        result += 1.0f;
      } else {
        result += (p[i] * std::log((p[i]) / (q[i])));
      }
    }
  }
  return result;
}
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_TOOLS_COMMON_STATISTIC_UTILS_H_
