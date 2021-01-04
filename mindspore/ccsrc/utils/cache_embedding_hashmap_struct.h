/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#ifndef MINDSPORE_CCSRC_UTIL_CACHE_EMBBEDDING_HASHMAP_STRUCT_H_
#define MINDSPORE_CCSRC_UTIL_CACHE_EMBBEDDING_HASHMAP_STRUCT_H_

#include <math.h>

namespace mindspore {
const int64_t kNullTag = 0;
const int64_t kInitStep = -5;
const int64_t kEmptyRate = 4;
const double kGoldenRatio = 0.6180339;
template <typename T>
struct HashmapEntry {
  T key_;
  T value_;
  T step_;
  T tag_;

  bool IsEmpty() { return tag_ == kNullTag; }

  bool IsUsing(const T train_step) { return step_ >= (train_step - 1); }

  bool IsKey(const T emb_idx) { return key_ == emb_idx; }

  void SetEmpty() { tag_ = kNullTag; }
};

template <typename T>
T HashFunc(const T key, const size_t m) {
  return (T)(((kGoldenRatio * key) - floor(kGoldenRatio * key)) * m);
}
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_UTIL_CACHE_EMBBEDDING_HASHMAP_STRUCT_H_
