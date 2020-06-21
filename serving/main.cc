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
#include "core/server.h"
#include "core/util/option_parser.h"

using mindspore::serving::Options;

int main(int argc, char **argv) {
  auto flag = Options::Instance().ParseCommandLine(argc, argv);
  if (!flag) {
    return 0;
  }
  mindspore::serving::Server server;
  server.BuildAndStart();
  return 0;
}
