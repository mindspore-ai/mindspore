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

#ifndef MINDSPORE_LITE_SRC_PACK_WEIGHT_MANAGER_H_
#define MINDSPORE_LITE_SRC_PACK_WEIGHT_MANAGER_H_
#ifdef USING_SERVING
#include <map>
#include <string>
#include <algorithm>
#include <utility>
#include <vector>
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
  std::vector<const LiteSession *> lite_sessions;
};
enum PackStatus { NOTPACK = 1, PACKED = 2, MALLOC = 3 };

class PackWeightManager {
 public:
  static PackWeightManager *GetInstance();
  virtual ~PackWeightManager();

  void DeleteSavedModelPtr(LiteModel *delete_model);
  void DeleteSavedSessionPtr(LiteSession *delete_session);
  void FreePathModelWeight();
  void FreeBufModelWeight();

  void InitWeightManagerByBuf(const char *model_buf, const LiteSession *lite_session);
  void InitWeightManagerByPath(const std::string &model_path, const char *model_buf,
                               const LiteSession *session = nullptr);
  STATUS StoreLiteModel(const char *model_buf, const Model *model);

  void StoreOriginTensor(const LiteModel *model, const SchemaTensorWrapper *origin_tensor, size_t tensor_index);
  void *GetTensorData(const LiteModel *model, size_t tensor_index);
  std::pair<PackStatus, void *> GetPackedTensor(const Tensor *tensor, const size_t size);
  void FreePackedWeight(ModelConstWeight *weight);

 private:
  PackWeightManager() = default;
  std::pair<PackStatus, void *> FindPackedTensor(PackedWeight *packed_weights, const OriginWeight &origin_weithts,
                                                 const Tensor *tensor, const size_t size);
  std::map<const char *, ModelConstWeight *> buf_model_weight_;
  std::map<const std::string, std::vector<const void *>> path_model_buf_;
  // path: model_buf
  std::map<const std::string, ModelConstWeight *> path_model_weight_;
};
}  // namespace mindspore::lite
#endif
#endif  // MINDSPORE_LITE_SRC_PACK_WEIGHT_MANAGER_H_
