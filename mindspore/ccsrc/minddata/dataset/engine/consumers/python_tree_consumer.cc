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
  std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> vec;
  Status s;
  {
    py::gil_scoped_release gil_release;
    s = GetNextAsOrderedPair(&vec);
  }
  RETURN_IF_NOT_OK(s);
  // Generate Python dict, python dict maintains its insertion order
  for (const auto &pair : vec) {
    (*out)[common::SafeCStr(pair.first)] = pair.second;
  }
  return Status::OK();
}

Status PythonBuildVocabConsumer::Start() {
  py::gil_scoped_release gil_release;
  return BuildVocabConsumer::Start();
}

Status PythonSaveToDisk::Save() {
  py::gil_scoped_release gil_release;
  return SaveToDisk::Save();
}

PythonSaveToDisk::PythonSaveToDisk(const std::string &datasetPath, int32_t numFiles, const std::string &datasetType)
    : SaveToDisk(datasetPath, numFiles, datasetType) {}

Status PythonTreeGetters::GetRow(TensorRow *const r) {
  py::gil_scoped_release gil_release;
  return TreeGetters::GetRow(r);
}
Status PythonDatasetSizeGetter::GetRow(const std::shared_ptr<TreeAdapter> &tree_adapter, TensorRow *r) {
  py::gil_scoped_release gil_release;
  return DatasetSizeGetter::GetRow(tree_adapter, r);
}
}  // namespace mindspore::dataset
