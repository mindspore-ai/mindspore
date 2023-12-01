/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "src/common/log.h"
#include "c_api/context_c.h"

extern "C" JNIEXPORT jlong JNICALL Java_com_mindspore_micro_MSContext_createDefaultMSContext(JNIEnv *env,
                                                                                             jobject thiz) {
  auto micro_context = MSContextCreate();
  if (micro_context == nullptr) {
    MS_LOG(ERROR) << "Micro new Context fail!";
    return (jlong) nullptr;
  }
  return (jlong)micro_context;
}

extern "C" JNIEXPORT void JNICALL Java_com_mindspore_micro_MSContext_free(JNIEnv *env, jobject thiz,
                                                                          jlong ms_context_handle) {
  auto c_context_ptr = static_cast<MSContextHandle>(reinterpret_cast<void *>(ms_context_handle));
  MSContextDestroy(&c_context_ptr);
}
