/**
 * Copyright 2020 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#include "backend/optimizer/somas/somas_stream.h"

namespace mindspore {
namespace somas {
void SomasStream::ComputeAncestorStreams() {
  // (Naive) algorithm: for a given stream, compute its ancestors assuming only distance 1 ancestors are known (handles
  // cycles between streams)
  std::set<SomasStreamPtr> current_level, temp_level, already_visited;
  auto thisPtr = std::make_shared<SomasStream>(id_);
  already_visited.insert(thisPtr);
  // Initialize current level to distance 2 ancestors
  for (auto stream1 : ancestor_streams_) {
    already_visited.insert(stream1);
    for (auto stream2 : stream1->ancestor_streams_) {
      if (std::find(already_visited.begin(), already_visited.end(), stream2) == already_visited.end())
        current_level.insert(stream2);
    }
  }

  while (!current_level.empty()) {
    // Push current level into ancestors
    for (auto stream1 : current_level) {
      ancestor_streams_.insert(stream1);
      already_visited.insert(stream1);
      // Keep next level of this ancestor
      for (auto stream2 : stream1->ancestor_streams_) {
        if (std::find(already_visited.begin(), already_visited.end(), stream2) == already_visited.end())
          temp_level.insert(stream2);
      }
    }
    current_level.clear();
    current_level = temp_level;
    temp_level.clear();
  }
}
}  // namespace somas
}  // namespace mindspore
