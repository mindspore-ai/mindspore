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
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <climits>
#include <string>
#include <iostream>
#include <memory>
#include <fstream>
#include "common/common_test.h"
#include "src/common/log_adapter.h"
#include "mindspore/lite/include/lite_session.h"
#include "mindspore/lite/src/executor.h"
#include "mindspore/lite/schema/inner/anf_ir_generated.h"

namespace mindspore {
class TestLiteInference : public mindspore::CommonTest {
 public:
  TestLiteInference() {}
};

std::string RealPath(const char *path) {
  if (path == nullptr) {
    return "";
  }
  if ((strlen(path)) >= PATH_MAX) {
    return "";
  }

  std::shared_ptr<char> resolvedPath(new (std::nothrow) char[PATH_MAX]{0});
  if (resolvedPath == nullptr) {
    return "";
  }

  auto ret = realpath(path, resolvedPath.get());
  if (ret == nullptr) {
    return "";
  }
  return resolvedPath.get();
}

char *ReadModelFile(const char *file, size_t *size) {
  if (file == nullptr) {
    return nullptr;
  }
  MS_ASSERT(size != nullptr);
  std::ifstream ifs(RealPath(file));
  if (!ifs.good()) {
    return nullptr;
  }

  if (!ifs.is_open()) {
    return nullptr;
  }

  ifs.seekg(0, std::ios::end);
  *size = ifs.tellg();
  std::unique_ptr<char> buf(new (std::nothrow) char[*size]);
  if (buf == nullptr) {
    ifs.close();
    return nullptr;
  }

  ifs.seekg(0, std::ios::beg);
  ifs.read(buf.get(), *size);
  ifs.close();

  return buf.release();
}
}  // namespace mindspore
