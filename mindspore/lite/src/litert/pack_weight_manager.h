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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_PACK_WEIGHT_MANAGER_H_
#define MINDSPORE_LITE_SRC_RUNTIME_PACK_WEIGHT_MANAGER_H_
#include <memory>
#include <vector>
#include <unordered_map>
#include <map>
#include <string>
#include "include/model.h"
#include "include/errorcode.h"
#include "src/tensor.h"
#ifdef SHARING_MODEL_WEIGHT
#include "src/litert/pack_weight.h"
#endif
namespace mindspore::lite {
class PackWeightManager {
 public:
  static PackWeightManager *GetInstance();
  ~PackWeightManager() = default;
  STATUS InitPackWeightManager(const char *model_buf, size_t model_size, std::string *model_id,
                               const std::map<std::string, std::map<std::string, std::string>> *config_info);
  char *GetSharedModelBuf(const char *model_buf, std::string model_id,
                          const std::map<std::string, std::map<std::string, std::string>> *config_info,
                          bool *is_shared);
  STATUS StoreOriginTensorData(Model *model, std::vector<Tensor *> *all_tensors);
  void *GetPackData(const void *tensor_data, const size_t size, bool *is_packed);
  void Free(void *tensor_data);
  bool IsCopyTensor(int op_type);
  void *ReplaceFp16Data(void *origin_fp16_data, size_t size, bool *replace);
  void FreePackWeight(std::string id);
  std::string GenRunnerID();
  std::string GenModelID();

 private:
  void *MallocData(size_t size);
  void FreeData(void *tensor_data);
  PackWeightManager() = default;
  bool is_parallel_ = false;
#ifdef SHARING_MODEL_WEIGHT
  std::shared_ptr<PackWeight> pack_weight_ = nullptr;
#endif
  std::mutex manager_mutex_;
  std::vector<std::string> runner_ids_;
  std::vector<std::string> model_ids_;
  size_t runner_id_ = 1;
  size_t model_id_ = 1;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_PACK_WEIGHT_MANAGER_H_
