/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "coder/generator/component/parallel_component.h"
#include <string>
#include "coder/opcoders/parallel.h"

namespace mindspore::lite::micro {

void CodeCreateThreadPool(std::ofstream &ofs) {
  ofs << "  int thread_num = 4;\n"
         "  BindMode bind_mode = NO_BIND_MODE;\n"
         "  if (argc >= 6) {\n"
         "    thread_num = atoi(argv[4]);\n"
         "    bind_mode = atoi(argv[5]);\n"
         "  }\n"
         "  struct ThreadPool *thread_pool = CreateThreadPool(thread_num, bind_mode);\n"
         "  if (thread_pool == NULL) {\n"
         "    MICRO_ERROR(\"create thread pool failed\");\n"
         "    return RET_ERROR;\n"
         "  }\n"
      << "  ret = "
      << "SetThreadPool(thread_pool);\n"
      << "  if (ret != RET_OK) {\n"
         "    MICRO_ERROR(\"set global thread pool failed\");\n"
         "    return RET_ERROR;\n"
         "  }\n"
         "  printf(\"config: ThreadNum: %d, BindMode: %d\\n\", thread_num, bind_mode);\n";
}

void CodeDestroyThreadPool(std::ofstream &ofs) { ofs << "  DestroyThreadPool(thread_pool);\n"; }

void CodeSetGlobalThreadPoolState(std::ofstream &ofs) {
  ofs << "/*\n"
         " * set global thread pool, which is created by user\n"
         " */\n"
      << "int "
      << "SetThreadPool(struct ThreadPool *thread_pool);\n\n";
}

void CodeSetGlobalThreadPoolImplement(std::ofstream &ofs) {
  ofs << "struct ThreadPool *" << gThreadPool << " = NULL;\n"
      << "int "
      << "SetThreadPool(struct ThreadPool *thread_pool) {\n"
      << "  if (thread_pool == NULL) {\n"
         "    return RET_ERROR;\n"
         "  }\n"
      << gThreadPool
      << " = thread_pool;\n"
         "  return RET_OK;\n"
         "}\n";
}
}  // namespace mindspore::lite::micro
