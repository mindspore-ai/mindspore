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
#include "src/tensor.h"
#include "src/lite_session.h"
namespace mindspore::lite {
// tensor index <-> tensor data
using OriginWeight = std::map<size_t, const void *>;
using PackedWeight = std::map<size_t, void *>;
struct ModelConstWeight {
  PackedWeight packed_weight;
  OriginWeight origin_weight;
  std::vector<const Model *> lite_models;
  std::map<const void *, size_t> origin_data_index;
  std::set<void *> packed_data;
};

enum PackStatus : int8_t { NOTPACK = 1, PACKED = 2, MALLOC = 3 };

class PackWeightManager {
 public:
  static PackWeightManager *GetInstance();
  virtual ~PackWeightManager();

  void InitWeightManagerByPath(const std::string &model_path, const char *model_buf);
  STATUS InitWeightManagerByBuf(const char *model_buf);
  void DeleteSavedModelPtr(LiteModel *delete_model);
  STATUS StoreLiteModel(const char *model_buf, const Model *model);
  void *GetTensorData(const LiteModel *model, const SchemaTensorWrapper *origin_tensor, size_t tensor_index);
  std::pair<PackStatus, void *> GetPackedTensor(const Tensor *tensor, const size_t size);

 private:
  PackWeightManager() = default;
  std::pair<PackStatus, void *> FindPackedTensor(ModelConstWeight *weight, const Tensor *tensor, const size_t size);
  void FreePackedWeight(ModelConstWeight *weight);

  std::map<const std::string, ModelConstWeight *> path_model_weight_;
  std::map<const std::string, ModelConstWeight *> buf_model_weight_;
  std::map<const std::string, std::vector<const void *>> path_model_buf_;
  std::mutex mtx_weight_;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_PACK_WEIGHT_MANAGER_H_
