/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "flatbuffers/flatbuffers.h"
#include "utils/log_adapter.h"
#include "src/common/file_utils.h"

namespace mindspore {
namespace lite {
int Storage::Save(const schema::MetaGraphT &graph, const std::string &outputPath) {
  flatbuffers::FlatBufferBuilder builder(1024);
  auto offset = schema::MetaGraph::Pack(builder, &graph);
  builder.Finish(offset);
  int size = builder.GetSize();
  auto content = builder.GetBufferPointer();
  if (content == nullptr) {
    MS_LOG(ERROR) << "GetBufferPointer nullptr";
    return RET_ERROR;
  }

  std::ofstream output(outputPath + ".ms", std::ofstream::binary);
  if (!output.is_open()) {
    MS_LOG(ERROR) << "ofstream open failed";
    return RET_ERROR;
  }

  output.write((const char *)content, size);
  output.close();
  return RET_OK;
}

schema::MetaGraphT *Storage::Load(const std::string &inputPath) {
  size_t size;
  auto buf = ReadFile(inputPath.c_str(), &size);
  if (buf == nullptr) {
    // MS_LOG(ERROR)("the file buffer is nullptr");
    return nullptr;
  }

  flatbuffers::Verifier verify((const uint8_t *)buf, size);
  // if (false == VerifyGraphDefBuffer(verify)) {
  //   //MS_LOG(ERROR)("the buffer is invalid and fail to create graph");
  //   return nullptr;
  // }

  auto graphDefT = schema::UnPackMetaGraph(buf);
  return graphDefT.release();
}
}  // namespace lite
}  // namespace mindspore

