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
#include "backend/common/optimizer/cache_manager.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/anf_runtime_algorithm.h"

namespace mindspore::opt {
void CacheManager::Update(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto type_iter = type_map_.find(node);
  auto shape_iter = shape_map_.find(node);
  if (type_iter != type_map_.cend()) {
    (void)type_map_.erase(type_iter);
  }
  if (shape_iter != shape_map_.cend()) {
    (void)shape_map_.erase(shape_iter);
  }
}

TypeId CacheManager::GetOutputType(const AnfNodePtr &node, size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  auto iter = type_map_.find(node);
  if (iter != type_map_.cend()) {
    auto types = iter->second;
    auto type_iter = types.find(index);
    if (type_iter != types.cend()) {
      return type_iter->second;
    }
    return kTypeUnknown;
  }
  auto output_nums = AnfAlgo::GetOutputTensorNum(node);
  std::map<size_t, TypeId> index_to_types;
  TypeId result = kTypeUnknown;
  for (size_t i = 0; i < output_nums; i++) {
    auto output_type = common::AnfAlgo::GetOutputInferDataType(node, i);
    (void)index_to_types.emplace(i, output_type);
    if (index == i) {
      result = output_type;
    }
  }
  (void)type_map_.emplace(node, index_to_types);
  return result;
}

ShapeVector CacheManager::GetOutputShape(const AnfNodePtr &node, size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  auto iter = shape_map_.find(node);
  if (iter != shape_map_.cend()) {
    auto shapes = iter->second;
    auto shape_iter = shapes.find(index);
    if (shape_iter != shapes.cend()) {
      return shape_iter->second;
    }
    return {};
  }
  auto output_nums = AnfAlgo::GetOutputTensorNum(node);
  std::map<size_t, ShapeVector> index_to_shapes;
  ShapeVector result = {};
  for (size_t i = 0; i < output_nums; i++) {
    auto output_shape = common::AnfAlgo::GetOutputInferShape(node, i);
    (void)index_to_shapes.emplace(i, output_shape);
    if (index == i) {
      result = output_shape;
    }
  }
  (void)shape_map_.emplace(node, index_to_shapes);
  return result;
}
}  // namespace mindspore::opt
