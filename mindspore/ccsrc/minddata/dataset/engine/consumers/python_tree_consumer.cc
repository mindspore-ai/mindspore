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

#include <string>
#include <unordered_map>
#include <vector>
#include "minddata/dataset/engine/consumers/python_tree_consumer.h"

namespace mindspore::dataset {

Status PythonIteratorConsumer::GetNextAsList(py::list *out) {
  std::vector<TensorPtr> row;
  {
    py::gil_scoped_release gil_release;
    RETURN_IF_NOT_OK(GetNextAsVector(&row));
  }
  for (auto el : row) {
    (*out).append(el);
  }
  return Status::OK();
}
Status PythonIteratorConsumer::GetNextAsDict(py::dict *out) {
  std::unordered_map<std::string, TensorPtr> row;
  {
    py::gil_scoped_release gil_release;
    RETURN_IF_NOT_OK(GetNextAsMap(&row));
  }
  for (auto el : row) {
    (*out)[common::SafeCStr(el.first)] = el.second;
  }
  return Status::OK();
}
}  // namespace mindspore::dataset
