/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "fl/compression/encode_executor.h"

#include <arpa/inet.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <algorithm>
#include <regex>
#include <map>
#include <utility>
#include <vector>
#include "fl/server/common.h"

namespace mindspore {
namespace fl {
namespace compression {
bool CompressExecutor::EnableCompressWeight(const schema::CompressType compressType) {
  return kCompressTypeMap.count(compressType) > 0;
}

bool CompressExecutor::construct_compress_weight(std::map<std::string, CompressWeight> *compressWeights,
                                                 std::map<std::string, std::vector<float>> feature_maps,
                                                 const schema::CompressType compressType) {
  if (compressType == schema::CompressType_QUANT) {
    return quant_min_max(compressWeights, feature_maps, kCompressTypeMap.at(compressType));
  }
  return false;
}

bool CompressExecutor::quant_min_max(std::map<std::string, CompressWeight> *compressWeights,
                                     std::map<std::string, std::vector<float>> feature_maps, size_t num_bits) {
  auto temp1 = static_cast<float>(1 << num_bits) - 1.0f;
  auto temp2 = static_cast<float>(1 << (num_bits - 1));
  for (const auto &feature_map : feature_maps) {
    std::string weight_name = feature_map.first;
    float min_value = 1e10f;
    float max_value = -min_value;
    for (const auto &feature : feature_map.second) {
      if (feature > max_value) {
        max_value = feature;
      }
      if (feature < min_value) {
        min_value = feature;
      }
    }
    float scale_value = (max_value - min_value) / temp1 + 1e-10f;
    size_t size = feature_map.second.size();
    if (size == 0) {
      MS_LOG(WARNING) << "The size of parameters is zero.";
      return false;
    }
    CompressWeight compressWeight;
    for (size_t i = 0; i < size; ++i) {
      auto round_data = round((feature_map.second[i] - min_value) / scale_value - temp2);
      // bit pack can be implemented here in the future
      auto int8_data = int8_t(round_data);
      compressWeight.compress_data.emplace_back(int8_data);
    }
    compressWeight.min_val = min_value;
    compressWeight.max_val = max_value;
    compressWeight.compress_data_len = size;

    (*compressWeights)[weight_name] = compressWeight;
  }
  return true;
}

schema::CompressType CompressExecutor::GetCompressType(const flatbuffers::Vector<int8_t> *download_compress_types) {
  schema::CompressType compressType = schema::CompressType_NO_COMPRESS;
  if (download_compress_types == nullptr) {
    MS_LOG(DEBUG) << "The client does not support current download compress type.";
  } else {
    for (size_t i = 0; i < download_compress_types->size(); ++i) {
      auto download_compress_type = download_compress_types->Get(i);
      if (download_compress_type == schema::CompressType_QUANT) {
        compressType = schema::CompressType_QUANT;
        break;
      }
    }
  }
  return compressType;
}
}  // namespace compression
}  // namespace fl
}  // namespace mindspore
