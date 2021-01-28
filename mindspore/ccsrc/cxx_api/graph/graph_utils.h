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
#ifndef MINDSPORE_CCSRC_CXX_API_GRAPH_GRAPH_UTILS_H
#define MINDSPORE_CCSRC_CXX_API_GRAPH_GRAPH_UTILS_H
#include <map>
#include <vector>
#include "include/api/types.h"
#include "ir/dtype/type_id.h"
#include "utils/log_adapter.h"

namespace mindspore::api {
class GraphUtils {
 public:
  static DataType TransTypeId2InferDataType(TypeId type_id) {
    const std::map<TypeId, api::DataType> id2type_map{
      {TypeId::kNumberTypeBegin, api::kMsUnknown},   {TypeId::kNumberTypeBool, api::kMsBool},
      {TypeId::kNumberTypeFloat64, api::kMsFloat64}, {TypeId::kNumberTypeInt8, api::kMsInt8},
      {TypeId::kNumberTypeUInt8, api::kMsUint8},     {TypeId::kNumberTypeInt16, api::kMsInt16},
      {TypeId::kNumberTypeUInt16, api::kMsUint16},   {TypeId::kNumberTypeInt32, api::kMsInt32},
      {TypeId::kNumberTypeUInt32, api::kMsUint32},   {TypeId::kNumberTypeInt64, api::kMsInt64},
      {TypeId::kNumberTypeUInt64, api::kMsUint64},   {TypeId::kNumberTypeFloat16, api::kMsFloat16},
      {TypeId::kNumberTypeFloat32, api::kMsFloat32},
    };

    auto it = id2type_map.find(type_id);
    if (it != id2type_map.end()) {
      return it->second;
    }

    MS_LOG(WARNING) << "Unsupported data id " << type_id;
    return api::kMsUnknown;
  }

  template <class T>
  inline static void ClearIfNotNull(T *vec) {
    if (vec != nullptr) {
      vec->clear();
    }
  }

  template <class T, class U>
  inline static void PushbackIfNotNull(U *vec, T &&item) {
    if (vec != nullptr) {
      vec->emplace_back(item);
    }
  }
};
}  // namespace mindspore::api

#endif  // MINDSPORE_CCSRC_CXX_API_GRAPH_GRAPH_UTILS_H
