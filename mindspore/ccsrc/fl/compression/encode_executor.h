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

#ifndef MINDSPORE_CCSRC_FL_COMPRESSION_ENCODE_EXECUTOR_H_
#define MINDSPORE_CCSRC_FL_COMPRESSION_ENCODE_EXECUTOR_H_

#include <memory>
#include <string>
#include <vector>
#include <map>
#include "proto/comm.pb.h"
#include "schema/fl_job_generated.h"
#include "schema/cipher_generated.h"
#include "fl/armour/secure_protocol/key_agreement.h"
#include "ps/ps_context.h"
#include "ps/core/worker_node.h"
#include "ps/core/cluster_metadata.h"
#include "ps/core/communicator/tcp_communicator.h"
#include "fl/server/common.h"

namespace mindspore {
namespace fl {
namespace compression {
// compress type map: schema::CompressType -> num bits
const std::map<schema::CompressType, size_t> kCompressTypeMap = {{schema::CompressType_QUANT, 8}};

struct CompressWeight {
  std::vector<int8_t> compress_data;
  size_t compress_data_len;
  float min_val;
  float max_val;
};

class CompressExecutor {
 public:
  static CompressExecutor &GetInstance() {
    static CompressExecutor instance;
    return instance;
  }

  bool EnableCompressWeight(const schema::CompressType compressType);

  bool construct_compress_weight(std::map<std::string, CompressWeight> *compressWeights,
                                 std::map<std::string, std::vector<float>> feature_maps,
                                 const schema::CompressType compressType);

  bool quant_min_max(std::map<std::string, CompressWeight> *compressWeights,
                     std::map<std::string, std::vector<float>> feature_maps, size_t num_bits);

  schema::CompressType GetCompressType(const flatbuffers::Vector<int8_t> *download_compress_types);
};
}  // namespace compression
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_COMPRESSION_ENCODE_EXECUTOR_H_
