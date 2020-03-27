/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "common/storage.h"
#include "flatbuffers/flatbuffers.h"
#include "common/mslog.h"
#include "common/file_utils.h"

namespace mindspore {
namespace predict {
int Storage::Save(const GraphDefT &graph, const std::string &outputPath) {
  flatbuffers::FlatBufferBuilder builder(flatSize);
  auto offset = GraphDef::Pack(builder, &graph);
  builder.Finish(offset);
  int size = builder.GetSize();
  auto content = builder.GetBufferPointer();
  if (content == nullptr) {
    MS_LOGE("GetBufferPointer nullptr");
    return RET_ERROR;
  }
  std::string realPath = RealPath(outputPath.c_str());
  if (realPath.empty()) {
    MS_LOGE("Output file path '%s' is not valid", outputPath.c_str());
    return RET_ERROR;
  }

  std::ofstream output(realPath, std::ofstream::binary);
  if (!output.is_open()) {
    MS_LOGE("ofstream open failed");
    return RET_ERROR;
  }
  output.write((const char *)content, size);
  output.close();
  return RET_OK;
}
}  // namespace predict
}  // namespace mindspore
