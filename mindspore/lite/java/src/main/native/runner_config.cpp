/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include <jni.h>
#include "common/ms_log.h"
#include "include/api/model_parallel_runner.h"

extern "C" JNIEXPORT jlong JNICALL Java_com_mindspore_config_RunnerConfig_createRunnerConfig(JNIEnv *env, jobject thiz,
                                                                                             jlong context_ptr) {
  auto runner_config = new (std::nothrow) mindspore::RunnerConfig();
  if (runner_config == nullptr) {
    MS_LOGE("new RunnerConfig fail!");
    return (jlong) nullptr;
  }
  auto *c_context_ptr = reinterpret_cast<mindspore::Context *>(context_ptr);
  if (c_context_ptr == nullptr) {
    delete runner_config;
    MS_LOGE("Context pointer from java is nullptr");
    return (jlong) nullptr;
  }
  auto context = std::make_shared<mindspore::Context>();
  if (context == nullptr) {
    delete runner_config;
    MS_LOGE("Make context failed");
    return (jlong) nullptr;
  }
  context.reset(c_context_ptr);
  runner_config->model_ctx = context;
  return (jlong)runner_config;
}

extern "C" JNIEXPORT void JNICALL Java_com_mindspore_config_RunnerConfig_setNumModel(JNIEnv *env, jobject thiz,
  jstring runner_config_ptr,
jint num_model) {
auto *pointer = reinterpret_cast<mindspore::RunnerConfig *>(runner_config_ptr);
if (pointer == nullptr) {
MS_LOGE("runner config pointer from java is nullptr");
return;
}
pointer->num_model = num_model;
}
