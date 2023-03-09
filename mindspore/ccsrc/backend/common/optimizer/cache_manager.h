/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_COMMON_CACHE_MANAGER_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_COMMON_CACHE_MANAGER_H_

#include <map>
#include "ir/anf.h"
#include "mindapi/base/type_id.h"
#include "mindapi/base/shape_vector.h"

namespace mindspore::opt {
class CacheManager {
 public:
  CacheManager() {}
  ~CacheManager() = default;
  void Update(const AnfNodePtr &node);
  TypeId GetOutputType(const AnfNodePtr &node, size_t index);
  ShapeVector GetOutputShape(const AnfNodePtr &node, size_t index);

 private:
  std::map<AnfNodePtr, std::map<size_t, TypeId>> type_map_;
  std::map<AnfNodePtr, std::map<size_t, ShapeVector>> shape_map_;
};
}  // namespace mindspore::opt
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_COMMON_CACHE_MANAGER_H
