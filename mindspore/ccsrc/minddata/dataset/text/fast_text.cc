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

#include "minddata/dataset/text/fast_text.h"

#include "utils/file_utils.h"

namespace mindspore {
namespace dataset {
FastText::FastText(const std::unordered_map<std::string, std::vector<float>> &map, int32_t dim) : Vectors(map, dim) {}

Status CheckFastText(const std::string &file_path) {
  Path path = Path(file_path);
  if (path.Exists() && !path.IsDirectory()) {
    std::string basename = path.Basename();
    size_t dot = basename.rfind('.');
    std::string suffix = basename.substr(dot + 1);
    if (suffix != "vec") {
      RETURN_STATUS_UNEXPECTED("FastText: invalid file, can not find file '*.vec', but got: " + file_path);
    }
    return Status::OK();
  } else {
    RETURN_STATUS_UNEXPECTED("FastText: invalid file, failed to open FastText file.");
  }
}

Status FastText::BuildFromFile(std::shared_ptr<FastText> *fast_text, const std::string &path, int32_t max_vectors) {
  RETURN_UNEXPECTED_IF_NULL(fast_text);
  RETURN_IF_NOT_OK(CheckFastText(path));
  std::unordered_map<std::string, std::vector<float>> map;
  int vector_dim = -1;
  RETURN_IF_NOT_OK(Load(path, max_vectors, &map, &vector_dim));
  *fast_text = std::make_shared<FastText>(std::move(map), vector_dim);
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
