/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "common.h"
#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

namespace UT {
#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

void DatasetOpTesting::SetUp() {
  std::string install_home = "data/dataset";
  datasets_root_path_ = install_home;
  mindrecord_root_path_ = "data/mindrecord";
}

std::vector<mindspore::dataset::TensorShape> DatasetOpTesting::ToTensorShapeVec(
  const std::vector<std::vector<int64_t>> &v) {
  std::vector<mindspore::dataset::TensorShape> ret_v;
  std::transform(v.begin(), v.end(), std::back_inserter(ret_v),
                 [](const auto &s) { return mindspore::dataset::TensorShape(s); });
  return ret_v;
}

std::vector<mindspore::dataset::DataType> DatasetOpTesting::ToDETypes(const std::vector<mindspore::DataType> &t) {
  std::vector<mindspore::dataset::DataType> ret_t;
  std::transform(t.begin(), t.end(), std::back_inserter(ret_t), [](const mindspore::DataType &t) {
    return mindspore::dataset::MSTypeToDEType(static_cast<mindspore::TypeId>(t));
  });
  return ret_t;
}

// Function to read a file into an MSTensor
// Note: This provides the analogous support for DETensor's CreateFromFile.
mindspore::MSTensor DatasetOpTesting::ReadFileToTensor(const std::string &file) {
  if (file.empty()) {
    MS_LOG(ERROR) << "Pointer file is nullptr; return an empty Tensor.";
    return mindspore::MSTensor();
  }
  std::ifstream ifs(file);
  if (!ifs.good()) {
    MS_LOG(ERROR) << "File: " << file << " does not exist; return an empty Tensor.";
    return mindspore::MSTensor();
  }
  if (!ifs.is_open()) {
    MS_LOG(ERROR) << "File: " << file << " open failed; return an empty Tensor.";
    return mindspore::MSTensor();
  }

  ifs.seekg(0, std::ios::end);
  size_t size = ifs.tellg();
  mindspore::MSTensor buf("file", mindspore::DataType::kNumberTypeUInt8, {static_cast<int64_t>(size)}, nullptr, size);

  ifs.seekg(0, std::ios::beg);
  ifs.read(reinterpret_cast<char *>(buf.MutableData()), size);
  ifs.close();

  return buf;
}

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
}  // namespace UT
