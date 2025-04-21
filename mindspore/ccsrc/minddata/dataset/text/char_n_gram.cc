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

#include "minddata/dataset/text/char_n_gram.h"

#include "utils/file_utils.h"

namespace mindspore {
namespace dataset {
CharNGram::CharNGram(const std::unordered_map<std::string, std::vector<float>> &map, int32_t dim) : Vectors(map, dim) {}

Status CharNGram::BuildFromFile(std::shared_ptr<CharNGram> *char_n_gram, const std::string &path, int32_t max_vectors) {
  RETURN_UNEXPECTED_IF_NULL(char_n_gram);
  std::unordered_map<std::string, std::vector<float>> map;
  int vector_dim = -1;
  RETURN_IF_NOT_OK(CharNGram::Load(path, max_vectors, &map, &vector_dim));
  *char_n_gram = std::make_shared<CharNGram>(std::move(map), vector_dim);
  return Status::OK();
}

std::vector<float> CharNGram::Lookup(const std::string &token, const std::vector<float> &unk_init,
                                     bool lower_case_backup) {
  std::vector<float> init_vec(dim_, 0);
  if (!unk_init.empty()) {
    if (unk_init.size() != dim_) {
      MS_LOG(WARNING) << "CharNGram: size of unk_init is not the same as vectors, will initialize with zero vectors.";
    } else {
      init_vec = unk_init;
    }
  }
  std::string lower_token = token;
  if (lower_case_backup) {
    std::transform(lower_token.begin(), lower_token.end(), lower_token.begin(), ::tolower);
  }

  std::vector<std::string> chars;
  chars.push_back("#BEGIN#");
  for (int i = 0; i < lower_token.length(); i++) {
    std::string s;
    s.push_back(lower_token[i]);  // Convert a char type letter to a string type.
    chars.push_back(s);
  }
  chars.push_back("#END#");

  int len = chars.size();
  int num_vectors = 0;
  std::vector<float> vector_value_sum(dim_, 0);
  std::vector<float> vector_value_temp;
  // The length of meaningful characters in the pre-training file is 2, 3, 4.
  const int slice_len[3] = {2, 3, 4};
  const int slice_len_size = sizeof(slice_len) / sizeof(slice_len[0]);
  for (int i = 0; i < slice_len_size; i++) {
    int end = len - slice_len[i] + 1;
    for (int pos = 0; pos < end; pos++) {
      std::vector<std::string> gram_vec;
      std::vector<std::string>::const_iterator first = chars.begin() + pos;
      std::vector<std::string>::const_iterator second = first + slice_len[i];
      gram_vec.assign(first, second);
      std::string c = "";
      std::string gram = std::accumulate(gram_vec.begin(), gram_vec.end(), c);
      std::string gram_key = std::to_string(slice_len[i]) + "gram-" + gram;
      auto str_index = map_.find(gram_key);
      if (str_index == map_.end()) {
        vector_value_temp = init_vec;
      } else {
        vector_value_temp = str_index->second;
      }
      if (vector_value_temp != init_vec) {
        std::transform(vector_value_temp.begin(), vector_value_temp.end(), vector_value_sum.begin(),
                       vector_value_sum.begin(), std::plus<float>());
        num_vectors++;
      }
    }
  }
  std::vector<float> vector_value(dim_, 0);
  if (num_vectors > 0) {
    std::transform(vector_value_sum.begin(), vector_value_sum.end(), vector_value.begin(),
                   [&num_vectors](float value) -> float { return value / num_vectors; });
    return vector_value;
  } else {
    return init_vec;
  }
}
}  // namespace dataset
}  // namespace mindspore
