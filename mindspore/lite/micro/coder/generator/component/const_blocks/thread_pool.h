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

#ifndef MINDSPORE_LITE_MICRO_CODER_GENERATOR_CONST_BLOCKS_THREAD_POOL_H_
#define MINDSPORE_LITE_MICRO_CODER_GENERATOR_CONST_BLOCKS_THREAD_POOL_H_

namespace mindspore::lite::micro {

const char *thread_pool_h =
  "/**\n"
  " * Copyright 2021 Huawei Technologies Co., Ltd\n"
  " *\n"
  " * Licensed under the Apache License, Version 2.0 (the \"License\");\n"
  " * you may not use this file except in compliance with the License.\n"
  " * You may obtain a copy of the License at\n"
  " *\n"
  " * http://www.apache.org/licenses/LICENSE-2.0\n"
  " *\n"
  " * Unless required by applicable law or agreed to in writing, software\n"
  " * distributed under the License is distributed on an \"AS IS\" BASIS,\n"
  " * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
  " * See the License for the specific language governing permissions and\n"
  " * limitations under the License.\n"
  " */\n"
  "\n"
  "#ifndef MINDSPORE_LITE_SRC_RUNTIME_THREAD_POOL_H_\n"
  "#define MINDSPORE_LITE_SRC_RUNTIME_THREAD_POOL_H_\n"
  "\n"
  "#include <stdbool.h>\n"
  "\n"
  "#define MAX_TASK_NUM (2)\n"
  "\n"
  "/// \\brief BindMode defined for holding bind cpu strategy argument.\n"
  "typedef enum {\n"
  "  NO_BIND_MODE = 0, /**< no bind */\n"
  "  HIGHER_MODE = 1,  /**< bind higher cpu first */\n"
  "  MID_MODE = 2      /**< bind middle cpu first */\n"
  "} BindMode;\n"
  "\n"
  "struct ThreadPool;\n"
  "\n"
  "struct ThreadPool *CreateThreadPool(int thread_num, int mode);\n"
  "\n"
  "/**\n"
  " *\n"
  " * @param session_index, support multi session\n"
  " * @param job\n"
  " * @param content\n"
  " * @param task_num\n"
  " */\n"
  "int ParallelLaunch(struct ThreadPool *thread_pool, int (*job)(void *, int), void *content, int task_num);\n"
  "\n"
  "/**\n"
  " * bind each thread to specified cpu core\n"
  " * @param is_bind\n"
  " * @param mode\n"
  " */\n"
  "int BindThreads(struct ThreadPool *thread_pool, bool is_bind, int mode);\n"
  "\n"
  "/**\n"
  " * activate the thread pool\n"
  " * @param thread_pool_id\n"
  " */\n"
  "void ActivateThreadPool(struct ThreadPool *thread_pool);\n"
  "\n"
  "/**\n"
  " * deactivate the thread pool\n"
  " * @param thread_pool_id\n"
  " */\n"
  "void DeactivateThreadPool(struct ThreadPool *thread_pool);\n"
  "\n"
  "/**\n"
  " *\n"
  " * @return current thread num\n"
  " */\n"
  "int GetCurrentThreadNum(struct ThreadPool *thread_pool);\n"
  "\n"
  "/**\n"
  " * destroy thread pool, and release resource\n"
  " */\n"
  "void DestroyThreadPool(struct ThreadPool *thread_pool);\n"
  "\n"
  "#endif  // MINDSPORE_LITE_SRC_RUNTIME_THREAD_POOL_H_\n";
}  // namespace mindspore::lite::micro

#endif  // MINDSPORE_LITE_MICRO_CODER_GENERATOR_CONST_BLOCKS_THREAD_POOL_H_
