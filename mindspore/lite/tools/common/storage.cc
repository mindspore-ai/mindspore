/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "tools/common/storage.h"
#include <sys/stat.h>
#ifndef _MSC_VER
#include <unistd.h>
#endif
#include "flatbuffers/flatbuffers.h"
#include "src/common/log_adapter.h"
#include "src/common/file_utils.h"

namespace mindspore {
namespace lite {
schema::MetaGraphT *Storage::Load(const std::string &inputPath) {
  size_t size = 0;
  std::string filename = inputPath;
  if (filename.substr(filename.find_last_of(".") + 1) != "ms") {
    filename = filename + ".ms";
  }
  auto buf = ReadFile(filename.c_str(), &size);
  if (buf == nullptr) {
    MS_LOG(ERROR) << "the file buffer is nullptr";
    return nullptr;
  }

  flatbuffers::Verifier verify((const uint8_t *)buf, size);
  if (!schema::VerifyMetaGraphBuffer(verify)) {
    MS_LOG(ERROR) << "the buffer is invalid and fail to create meta graph";
    return nullptr;
  }

  auto graphDefT = schema::UnPackMetaGraph(buf);
  return graphDefT.release();
}
}  // namespace lite
}  // namespace mindspore
