/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_PACK_WEIGHT_H_
#define MINDSPORE_LITE_SRC_PACK_WEIGHT_H_
#include <map>
#include <string>
#include <algorithm>
#include <utility>
#include <vector>
#include <set>
#include <mutex>
#include <unordered_map>
#include <memory>
#include "src/tensor.h"
#include "src/lite_session.h"
namespace mindspore::lite {
struct ModelConstWeight {
  // origin tensor data <-> packed tensor data
  std::map<const void *, void *> origin_and_packed_pair;
  std::shared_ptr<Allocator> allocator = nullptr;
  int numa_id = -1;
};

class PackWeight {
 public:
  PackWeight() = default;
  ~PackWeight();
  STATUS InitWeightManagerByBuf(const char *model_buf, size_t model_size, int numa_id = -1);
  char *GetNumaModelBuf(int numa_id);
  STATUS StoreOriginTensorData(const char *model_buf, const void *origin_tensor_data);
  void *GetPackedTensor(const void *tensor_data, const size_t size, bool *is_packed);

 private:
  void FreePackedWeight(ModelConstWeight *weight);

  std::mutex mtx_weight_;
  std::unordered_map<const char *, ModelConstWeight *> buf_model_weight_;
  std::unordered_map<int, char *> numa_model_buf_;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_PACK_WEIGHT_H_
