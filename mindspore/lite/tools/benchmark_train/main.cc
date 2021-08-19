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

#include <malloc.h>
#include <unistd.h>
#include <fstream>
#include "tools/benchmark_train/net_train.h"
#include "include/version.h"

void PrintMem() {
  std::string proc_file = "/proc/" + std::to_string(getpid()) + "/status";
  std::ifstream infile(proc_file);
  if (infile.good()) {
    std::string line;
    while (std::getline(infile, line)) {
      if (line.find("VmHWM") != std::string::npos) {
        std::cout << line << std::endl;
      }
    }
    infile.close();
    struct mallinfo info = mallinfo();
    std::cout << "Arena allocation: " << info.arena + info.hblkhd << std::endl;
    // process pair (a,b)
  }
}

int main(int argc, const char **argv) {
  MS_LOG(INFO) << mindspore::lite::Version();
  int res = mindspore::lite::RunNetTrain(argc, argv);
  PrintMem();
  return res;
}
