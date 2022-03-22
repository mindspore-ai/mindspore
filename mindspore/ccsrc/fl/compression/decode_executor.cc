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

#include "fl/compression/decode_executor.h"

namespace mindspore {
namespace fl {
namespace compression {
std::vector<int> DecodeExecutor::ConstructMaskArray(int seed, float upload_sparse_rate, size_t param_num) {
  static int multiplier = 2147483647;
  static double increment = 4294967294.0;
  static int modulo = 48271;
  size_t retain_num = size_t(static_cast<float>(param_num) * upload_sparse_rate);
  if (retain_num == 0) {
    MS_LOG(WARNING) << "The retain_num is 0, and upload_sparse_rate is too small.";
  }
  std::vector<int> mask_array(param_num, 0);
  for (size_t i = 0; i < retain_num; ++i) {
    mask_array[i] = 1;
  }

  seed = ((seed + multiplier) * modulo) % multiplier;
  for (size_t i = 0; i < param_num; ++i) {
    // generate random number in (0, 1)
    double rand = static_cast<double>(seed) / increment + 0.5;
    // update seed
    seed = (seed * modulo) % multiplier;
    size_t j = size_t(rand * static_cast<double>(param_num - i)) + i;
    int temp = mask_array[i];
    mask_array[i] = mask_array[j];
    mask_array[j] = temp;
  }
  return mask_array;
}

bool DecodeExecutor::DeQuantSparseDiff(std::map<std::string, std::vector<float>> *weight_map,
                                       const std::vector<CompressFeatureMap> &compress_feature_maps, size_t num_bits,
                                       float upload_sparse_rate, int seed, const std::vector<std::string> &name_vec,
                                       size_t data_size) {
  std::vector<std::vector<float>> decompress_feature_maps;

  // origin parameters
  std::vector<size_t> shape_vec;
  size_t param_num = 0;
  const auto &iter_to_model = mindspore::fl::server::ModelStore::GetInstance().iteration_to_model();
  size_t latest_iter_num = iter_to_model.rbegin()->first;
  std::map<std::string, AddressPtr> feature_maps =
    mindspore::fl::server::ModelStore::GetInstance().GetModelByIterNum(latest_iter_num);
  // get shape vector and number of upload parameters
  for (const auto &name : name_vec) {
    size_t shape = feature_maps[name]->size / sizeof(float);
    shape_vec.emplace_back(shape);
    param_num += shape;
  }
  MS_LOG(DEBUG) << "Compression get last weights success!";

  // quant decode
  auto temp1 = static_cast<float>(1 << num_bits) - 1.0f;
  auto temp2 = static_cast<float>(1 << (num_bits - 1));
  std::vector<float> de_min_max_feature_map;
  for (auto compress_feature_map : compress_feature_maps) {
    float min_val = compress_feature_map.min_val;
    float max_val = compress_feature_map.max_val;
    float scale_val = static_cast<float>(max_val - min_val) / temp1 + 1e-10f;
    size_t size = compress_feature_map.compress_data.size();
    for (size_t i = 0; i < size; ++i) {
      de_min_max_feature_map.emplace_back(
        (static_cast<float>(compress_feature_map.compress_data[i]) + temp2) * scale_val + min_val);
    }
  }
  MS_LOG(DEBUG) << "Compression quant decode success!";

  // sparse decode
  std::vector<int> mask_array = ConstructMaskArray(seed, upload_sparse_rate, param_num);
  size_t index = 0;
  size_t de_min_max_feature_map_index = 0;
  for (const auto &shape : shape_vec) {
    std::vector<float> feature_map(shape);
    for (size_t i = 0; i < shape; ++i) {
      if (index >= mask_array.size()) {
        MS_LOG(WARNING) << "The mask_array and parameter shape is not matched.";
        return false;
      }
      if (mask_array[index] == 1) {
        if (de_min_max_feature_map_index >= de_min_max_feature_map.size()) {
          MS_LOG(WARNING) << "The number of upload parameters is too small.";
          return false;
        }
        feature_map[i] = de_min_max_feature_map[de_min_max_feature_map_index];
        de_min_max_feature_map_index += 1;
      } else {
        feature_map[i] = 0.0f;
      }
      index += 1;
    }
    decompress_feature_maps.emplace_back(feature_map);
  }
  MS_LOG(DEBUG) << "Compression sparse decode success!";

  // difference decode
  for (size_t i = 0; i < decompress_feature_maps.size(); ++i) {
    size_t feature_size = decompress_feature_maps[i].size();
    std::string name = name_vec[i];
    float *weight_data = reinterpret_cast<float *>(feature_maps[name]->addr);
    auto &weight_item = (*weight_map)[name];
    weight_item.resize(feature_size);
    for (size_t j = 0; j < feature_size; ++j) {
      weight_item[j] = decompress_feature_maps[i][j] + data_size * weight_data[j];
    }
  }
  MS_LOG(DEBUG) << "Compression difference decode success!";

  return true;
}

bool DecodeExecutor::Decode(std::map<std::string, std::vector<float>> *weight_map,
                            const std::vector<CompressFeatureMap> &compress_feature_maps,
                            schema::CompressType upload_compress_type, float upload_sparse_rate, int seed,
                            const std::vector<std::string> &name_vec, size_t data_size) {
  if (upload_compress_type == schema::CompressType_DIFF_SPARSE_QUANT) {
    return DeQuantSparseDiff(weight_map, compress_feature_maps, 8, upload_sparse_rate, seed, name_vec, data_size);
  }
  return false;
}

schema::CompressType DecodeExecutor::GetCompressType(schema::CompressType upload_compress_type) {
  if (upload_compress_type == schema::CompressType_DIFF_SPARSE_QUANT) {
    MS_LOG(DEBUG) << "This upload compress type is DIFF_SPARSE_QUANT.";
    return schema::CompressType_DIFF_SPARSE_QUANT;
  }

  MS_LOG(DEBUG) << "This upload compress type is NO_COMPRESS.";
  return schema::CompressType_NO_COMPRESS;
}
}  // namespace compression
}  // namespace fl
}  // namespace mindspore
