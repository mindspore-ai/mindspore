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

#include "minddata/dataset/text/vectors.h"

#include "utils/file_utils.h"

namespace mindspore {
namespace dataset {
Status Vectors::InferShape(const std::string &path, int32_t max_vectors, int32_t *num_lines, int32_t *header_num_lines,
                           int32_t *vector_dim) {
  RETURN_UNEXPECTED_IF_NULL(num_lines);
  RETURN_UNEXPECTED_IF_NULL(header_num_lines);
  RETURN_UNEXPECTED_IF_NULL(vector_dim);

  std::ifstream file_reader;
  file_reader.open(path, std::ios::in);
  CHECK_FAIL_RETURN_UNEXPECTED(file_reader.is_open(), "Vectors: invalid file, failed to open vector file: " + path);

  *num_lines = 0, *header_num_lines = 0, *vector_dim = -1;
  std::string line, row;
  while (std::getline(file_reader, line)) {
    if (*vector_dim == -1) {
      std::vector<std::string> vec;
      std::istringstream line_reader(line);
      while (std::getline(line_reader, row, ' ')) {
        vec.push_back(row);
      }
      // The number of rows and dimensions can be obtained directly from the information header.
      const int kInfoHeaderSize = 2;
      if (vec.size() == kInfoHeaderSize) {
        (*header_num_lines)++;
      } else {
        *vector_dim = vec.size() - 1;
        (*num_lines)++;
      }
    } else {
      (*num_lines)++;
    }
  }
  file_reader.close();
  CHECK_FAIL_RETURN_UNEXPECTED(*num_lines > 0, "Vectors: invalid file, file is empty.");

  if (max_vectors > 0) {
    *num_lines = std::min(max_vectors, *num_lines);  // Determine the true rows.
  }
  return Status::OK();
}

Status Vectors::Load(const std::string &path, int32_t max_vectors,
                     std::unordered_map<std::string, std::vector<float>> *map, int32_t *vector_dim) {
  RETURN_UNEXPECTED_IF_NULL(map);
  RETURN_UNEXPECTED_IF_NULL(vector_dim);
  auto realpath = FileUtils::GetRealPath(common::SafeCStr(path));
  CHECK_FAIL_RETURN_UNEXPECTED(realpath.has_value(), "Vectors: get real path failed, path: " + path);
  auto file_path = realpath.value();

  CHECK_FAIL_RETURN_UNEXPECTED(max_vectors >= 0,
                               "Vectors: max_vectors must be non negative, but got: " + std::to_string(max_vectors));

  int num_lines = 0, header_num_lines = 0;
  RETURN_IF_NOT_OK(InferShape(file_path, max_vectors, &num_lines, &header_num_lines, vector_dim));

  std::fstream file_reader;
  file_reader.open(file_path, std::ios::in);
  CHECK_FAIL_RETURN_UNEXPECTED(file_reader.is_open(),
                               "Vectors: invalid file, failed to open vector file: " + file_path);

  while (header_num_lines > 0) {
    file_reader.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    header_num_lines--;
  }

  std::string line, token, vector_value;
  for (auto i = 0; i < num_lines; ++i) {
    std::getline(file_reader, line);
    std::istringstream line_reader(line);
    std::getline(line_reader, token, ' ');
    std::vector<float> vector_values;
    int dim = 0;
    while (line_reader >> vector_value) {
      dim++;
      vector_values.push_back(atof(vector_value.c_str()));
    }
    if (dim <= 1) {
      file_reader.close();
      RETURN_STATUS_UNEXPECTED("Vectors: token with 1-dimensional vector.");
    }
    if (dim != *vector_dim) {
      file_reader.close();
      RETURN_STATUS_UNEXPECTED("Vectors: all vectors must have the same number of dimensions, but got dim " +
                               std::to_string(dim) + " while expecting " + std::to_string(*vector_dim));
    }

    auto token_index = map->find(token);
    if (token_index == map->end()) {
      (*map)[token] = vector_values;
    }
  }
  file_reader.close();
  return Status::OK();
}

Vectors::Vectors(const std::unordered_map<std::string, std::vector<float>> &map, int32_t dim) {
  map_ = map;
  dim_ = dim;
}

Status Vectors::BuildFromFile(std::shared_ptr<Vectors> *vectors, const std::string &path, int32_t max_vectors) {
  RETURN_UNEXPECTED_IF_NULL(vectors);
  std::unordered_map<std::string, std::vector<float>> map;
  int vector_dim = -1;
  RETURN_IF_NOT_OK(Load(path, max_vectors, &map, &vector_dim));
  *vectors = std::make_shared<Vectors>(std::move(map), vector_dim);
  return Status::OK();
}

std::vector<float> Vectors::Lookup(const std::string &token, const std::vector<float> &unk_init,
                                   bool lower_case_backup) {
  std::vector<float> init_vec(dim_, 0);
  if (!unk_init.empty()) {
    if (unk_init.size() != dim_) {
      MS_LOG(WARNING) << "Vectors: size of unk_init is not the same as vectors, will initialize with zero vectors.";
    } else {
      init_vec = unk_init;
    }
  }
  std::string lower_token = token;
  if (lower_case_backup) {
    transform(lower_token.begin(), lower_token.end(), lower_token.begin(), ::tolower);
  }
  auto str_index = map_.find(lower_token);
  if (str_index == map_.end()) {
    return init_vec;
  } else {
    return str_index->second;
  }
}
}  // namespace dataset
}  // namespace mindspore
