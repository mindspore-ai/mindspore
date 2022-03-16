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

#ifndef MINDSPORE_CCSRC_FL_COMPRESSION_DECODE_EXECUTOR_H_
#define MINDSPORE_CCSRC_FL_COMPRESSION_DECODE_EXECUTOR_H_

#include <memory>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <algorithm>
#include <regex>
#include <map>
#include <utility>
#include "proto/comm.pb.h"
#include "schema/fl_job_generated.h"
#include "schema/cipher_generated.h"
#include "fl/server/model_store.h"
#include "fl/server/common.h"
#include "ps/ps_context.h"

namespace mindspore {
namespace fl {
namespace compression {
struct CompressFeatureMap {
  std::string weight_fullname;
  std::vector<int8_t> compress_data;
  float min_val;
  float max_val;
};

class DecodeExecutor {
 public:
  static DecodeExecutor &GetInstance() {
    static DecodeExecutor instance;
    return instance;
  }

  // construct mask array for random sparse
  std::vector<int> ConstructMaskArray(int seed, float upload_sparse_rate, size_t param_num);

  // decode min_max quantization and random sparse and parameter difference
  bool DeQuantSparseDiff(std::map<std::string, std::vector<float>> *weight_map,
                         const std::vector<CompressFeatureMap> &compress_feature_maps, size_t num_bits,
                         float upload_sparse_rate, int seed, const std::vector<std::string> &name_vec,
                         size_t data_size);

  // decode
  bool Decode(std::map<std::string, std::vector<float>> *weight_map,
              const std::vector<CompressFeatureMap> &compress_feature_maps, schema::CompressType upload_compress_type,
              float upload_sparse_rate, int seed, const std::vector<std::string> &name_vec, size_t data_size);

  schema::CompressType GetCompressType(schema::CompressType upload_compress_type);
};
}  // namespace compression
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_COMPRESSION_DECODE_EXECUTOR_H_
