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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_PACK_WEIGHT_H_
#define MINDSPORE_LITE_SRC_RUNTIME_PACK_WEIGHT_H_
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
#include "src/litert/lite_session.h"
namespace mindspore::lite {
struct ModelConstWeight {
  // origin tensor data <-> packed tensor data
  std::map<const void *, void *> origin_and_packed_pair;
  std::shared_ptr<Allocator> allocator = nullptr;
  int numa_id = -1;
  std::unordered_map<int, void *> tensors_data;
  std::set<void *> fp16_fp32_data;
  bool copy_buf;
};

class PackWeight {
 public:
  PackWeight() = default;
  ~PackWeight();
  STATUS InitPackWeight(const void *model_buf, size_t model_size, std::string id, int numa_id,
                        bool need_copy_buf = true);
  char *GetSharedModelBuf(std::string id, int numa_id);
  STATUS StoreOriginTensorData(const void *model_buf, const void *origin_tensor_data);
  void *GetPackData(const void *tensor_data, const size_t size, bool *is_packed);
  STATUS ReplaceOriginTensorData(const void *model_buf, std::vector<Tensor *> *tensors, int tensor_index);
  void *ReplaceFp16Data(void *origin_fp16_data, size_t size);
  void FreePackWeight(std::string id, bool free_all = false);

 private:
  void FreePackedWeight(ModelConstWeight *weight);
  void FreeTensorData(ModelConstWeight *weight);
  void FreeFp16ToFp32Data(ModelConstWeight *weight);

  std::mutex mtx_weight_;
  std::unordered_map<void *, void *> fp16_fp32_data_pair_;
  // runner_id/model_id : { numa_id : ModelConstWeight }
  std::unordered_map<std::string, std::unordered_map<int, ModelConstWeight *>> model_weights_;
  // runner_id/model_id : { numa_id : shared model buf address }
  std::unordered_map<std::string, std::unordered_map<int, void *>> shared_bufs_;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_PACK_WEIGHT_H_
