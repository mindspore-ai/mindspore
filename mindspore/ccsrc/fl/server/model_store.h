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

#ifndef MINDSPORE_CCSRC_FL_SERVER_MODEL_STORE_H_
#define MINDSPORE_CCSRC_FL_SERVER_MODEL_STORE_H_

#include <map>
#include <memory>
#include <string>
#include "fl/server/common.h"
#include "fl/server/memory_register.h"
#include "fl/server/executor.h"

namespace mindspore {
namespace fl {
namespace server {
// The initial iteration number is 0 in server.
constexpr size_t kInitIterationNum = 0;

// The initial iteration number after ModelStore is reset.
constexpr size_t kResetInitIterNum = 1;

// Server framework use ModelStore to store and query models.
// ModelStore stores multiple models because worker could get models of the previous iterations.
class ModelStore {
 public:
  static ModelStore &GetInstance() {
    static ModelStore instance;
    return instance;
  }

  // Initialize ModelStore with max count of models need to be stored.
  void Initialize(uint32_t max_count = 3);

  // Store the model of the given iteration. The model is acquired from Executor. If the current model count is already
  // max_model_count_, the earliest model will be replaced.
  void StoreModelByIterNum(size_t iteration, const std::map<std::string, AddressPtr> &model);

  // Get model of the given iteration.
  std::map<std::string, AddressPtr> GetModelByIterNum(size_t iteration);

  // Reset the stored models. Called when federated learning job finishes.
  void Reset();

  // Returns all models stored in ModelStore.
  const std::map<size_t, std::shared_ptr<MemoryRegister>> &iteration_to_model();

  // Returns the model size, which could be calculated at the initializing phase.
  size_t model_size() const;

 private:
  ModelStore() : max_model_count_(0), model_size_(0), iteration_to_model_({}) {}
  ~ModelStore() = default;
  ModelStore(const ModelStore &) = delete;
  ModelStore &operator=(const ModelStore &) = delete;

  // To store multiple models, new memory must assigned. The max memory size assigned for models is max_model_count_ *
  // model_size_.
  std::shared_ptr<MemoryRegister> AssignNewModelMemory();

  // Calculate the model size. This method should be called after iteration_to_model_ is initialized.
  size_t ComputeModelSize();

  size_t max_model_count_;
  size_t model_size_;

  // Initial model which is the model of iteration 0.
  std::shared_ptr<MemoryRegister> initial_model_;

  // The number of all models stored is max_model_count_.
  std::mutex model_mtx_;
  std::map<size_t, std::shared_ptr<MemoryRegister>> iteration_to_model_;
};
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_MODEL_STORE_H_
