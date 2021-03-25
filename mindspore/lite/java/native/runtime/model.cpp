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

#include <jni.h>
#include <fstream>
#include "common/ms_log.h"
#include "include/model.h"

extern "C" JNIEXPORT jlong JNICALL Java_com_mindspore_lite_Model_loadModel(JNIEnv *env, jobject thiz, jobject buffer) {
  if (buffer == nullptr) {
    MS_LOGE("Buffer from java is nullptr");
    return reinterpret_cast<jlong>(nullptr);
  }
  jlong buffer_len = env->GetDirectBufferCapacity(buffer);
  auto *model_buffer = static_cast<char *>(env->GetDirectBufferAddress(buffer));

  auto model = mindspore::lite::Model::Import(model_buffer, buffer_len);
  if (model == nullptr) {
    MS_LOGE("Import model failed");
    return reinterpret_cast<jlong>(nullptr);
  }
  return reinterpret_cast<jlong>(model);
}

extern "C" JNIEXPORT jlong JNICALL Java_com_mindspore_lite_Model_loadModelByPath(JNIEnv *env, jobject thiz,
                                                                                 jstring model_path) {
  auto model_path_char = env->GetStringUTFChars(model_path, JNI_FALSE);
  if (nullptr == model_path_char) {
    MS_LOGE("model_path_char is nullptr");
    return reinterpret_cast<jlong>(nullptr);
  }
  std::ifstream ifs(model_path_char);
  if (!ifs.good()) {
    MS_LOGE("file: %s is not exist", model_path_char);
    return reinterpret_cast<jlong>(nullptr);
  }

  if (!ifs.is_open()) {
    MS_LOGE("file: %s open failed", model_path_char);
    return reinterpret_cast<jlong>(nullptr);
  }

  ifs.seekg(0, std::ios::end);
  auto size = ifs.tellg();
  auto buf = new (std::nothrow) char[size];
  if (buf == nullptr) {
    MS_LOGE("malloc buf failed, file: %s", model_path_char);
    ifs.close();
    return reinterpret_cast<jlong>(nullptr);
  }

  ifs.seekg(0, std::ios::beg);
  ifs.read(buf, size);
  ifs.close();
  auto model = mindspore::lite::Model::Import(buf, size);
  delete[] buf;
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

extern "C" JNIEXPORT void JNICALL Java_com_mindspore_lite_Model_freeBuffer(JNIEnv *env, jobject thiz, jlong model_ptr) {
  auto *pointer = reinterpret_cast<void *>(model_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Model pointer from java is nullptr");
    return;
  }
  auto *lite_model_ptr = static_cast<mindspore::lite::Model *>(pointer);
  lite_model_ptr->Free();
}
