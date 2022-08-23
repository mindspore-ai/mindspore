/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_DEVICE_DUMP_DUMP_DATA_BUILDER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_DEVICE_DUMP_DUMP_DATA_BUILDER_H_
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include "utils/log_adapter.h"
#include "proto/dump_data.pb.h"
#include "toolchain/adx_datadump_callback.h"

using Adx::DumpChunk;

// This class is for building dump data receiving from adx server. Tensor Data for each kernel will be divided in pieces
// and each piece would be wrapped into DumpChunk struct. This class provides function to merge dump chunks and
// construct dump data object.
class DumpDataBuilder {
 public:
  DumpDataBuilder() {}

  ~DumpDataBuilder() = default;

  /*
   * Feature group: Dump.
   * Target device group: Ascend.
   * Runtime category: Old runtime, MindRT.
   * Description: This function is for A+M dump only. In each callback, allocate memory and copy the dump chunk from
   * adx. Return false if OOM.
   */
  bool CopyDumpChunk(const DumpChunk *dump_chunk) {
    try {
      uint32_t buf_sz = dump_chunk->bufLen;
      std::string buffer_str(reinterpret_cast<const char *>(dump_chunk->dataBuf), buf_sz);
      chunk_list_.push_back(buffer_str);
      total_sz_ += buf_sz;
    } catch (std::bad_alloc &err) {
      MS_LOG(ERROR) << "Failed to allocate memory for " << dump_chunk->fileName << ", reason: " << err.what();
      return false;
    }
    return true;
  }

  /*
   * Feature group: Dump.
   * Target device group: Ascend.
   * Runtime category: Old runtime, MindRT.
   * Description: This function is for A+M dump only. When receiving the last chunk of the node (is_last_chunk = true),
   * parse and construct the dump data for dumping. It does the these steps: 1) merge all chunks for the node; 2)
   * parse header and protobuf string; 3) memcpy tensor data to contiguous memory segment.
   */
  bool ConstructDumpData(debugger::dump::DumpData *dump_data_proto, std::vector<char> *data_ptr) {
    if (chunk_list_.empty()) {
      return false;
    }
    // merge several chunks into one piece.
    std::string dump_proto_str;
    dump_proto_str.reserve(total_sz_);
    for (auto item : chunk_list_) {
      dump_proto_str += item;
    }
    chunk_list_.clear();

    const int8_t header_len_offset = 8;
    uint64_t header_len = *reinterpret_cast<const uint64_t *>(dump_proto_str.c_str());
    std::string header = dump_proto_str.substr(header_len_offset, header_len);
    if (!(*dump_data_proto).ParseFromString(header)) {
      MS_LOG(ERROR) << "Failed to parse dump proto file.";
      return false;
    }
    auto data_sz = total_sz_ - header_len_offset - header_len;
    data_ptr->resize(data_sz);
    // The security memory copy function 'memcpy_s' has a size limit (SECUREC_MEM_MAX_LEN). If the data size is greater
    // than that, it should be cut into segments to copy. Otherwise, memcpy_s will fail.
    int ret;
    if (data_sz < SECUREC_MEM_MAX_LEN) {
      ret = memcpy_s(data_ptr->data(), data_sz, dump_proto_str.c_str() + header_len_offset + header_len, data_sz);
    } else {
      size_t mem_cpy_len;
      for (size_t pos = 0; pos < data_sz; pos += SECUREC_MEM_MAX_LEN) {
        mem_cpy_len = std::min(data_sz - pos, SECUREC_MEM_MAX_LEN);
        ret = memcpy_s(data_ptr->data() + pos, mem_cpy_len,
                       dump_proto_str.c_str() + header_len_offset + header_len + pos, mem_cpy_len);
        if (ret != 0) {
          break;
        }
      }
    }
    if (ret != EOK) {
      MS_LOG(ERROR) << "Failed to memcpy: error code (" << ret << ").";
      return false;
    }
    return true;
  }

 private:
  std::vector<std::string> chunk_list_;
  uint64_t total_sz_{0};
};
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_DEVICE_DUMP_DUMP_DATA_BUILDER_H_
