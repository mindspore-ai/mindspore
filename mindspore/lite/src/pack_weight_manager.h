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

#ifndef MINDSPORE_LITE_SRC_PACK_WEIGHT_MANAGER_H_
#define MINDSPORE_LITE_SRC_PACK_WEIGHT_MANAGER_H_
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
#include "include/model.h"
#ifdef SHARING_MODEL_WEIGHT
#include "src/pack_weight.h"
#endif
namespace mindspore::lite {
enum PackStatus : int8_t { NOTPACK = 1, PACKED = 2, MALLOC = 3 };

class PackWeightManager {
 public:
  static PackWeightManager *GetInstance();
  ~PackWeightManager() = default;
  STATUS InitByBuf(const char *model_buf, size_t model_size, int numa_id = -1);
  char *GetNumaModelBuf(int numa_id);
  STATUS StoreOriginTensorData(Model *model);
  void *GetPackedTensor(const void *tensor_data, const size_t size, bool *is_packed);
  void Free(void *tensor_data);

 private:
  PackWeightManager() = default;
#ifdef SHARING_MODEL_WEIGHT
  std::shared_ptr<PackWeight> pack_weight_ = nullptr;
#endif
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_PACK_WEIGHT_MANAGER_H_
