/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#include <jni.h>
#include "common/ms_log.h"
#include "include/model.h"

extern "C" JNIEXPORT jlong JNICALL Java_com_mindspore_lite_Model_loadModel(JNIEnv *env, jobject thiz, jobject buffer) {
  MS_LOGD("Start getting buffer from java");
  if (buffer == nullptr) {
    MS_LOGE("Buffer from java is nullptr");
    return reinterpret_cast<jlong>(nullptr);
  }
  jlong buffer_len = env->GetDirectBufferCapacity(buffer);
  auto *model_buffer = static_cast<char *>(env->GetDirectBufferAddress(buffer));

  MS_LOGD("Start Loading model");
  auto model = mindspore::lite::Model::Import(model_buffer, buffer_len);
  //    env->DeleteLocalRef(*(jobject *)model_buffer);
  if (model == nullptr) {
    MS_LOGE("Import model failed");
    return reinterpret_cast<jlong>(nullptr);
  }
  return reinterpret_cast<jlong>(model);
}

extern "C" JNIEXPORT void JNICALL Java_com_mindspore_lite_Model_free(JNIEnv *env, jobject thiz, jlong model_ptr) {
  auto *pointer = reinterpret_cast<void *>(model_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Model pointer from java is nullptr");
    return;
  }
  auto *lite_model_ptr = static_cast<mindspore::lite::Model *>(pointer);
  delete (lite_model_ptr);
}
