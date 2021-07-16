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
#include "common/ms_log.h"
#include "include/train/train_session.h"
#include "include/train/train_cfg.h"
#include "include/errorcode.h"

extern "C" JNIEXPORT jlong JNICALL Java_com_mindspore_lite_TrainSession_createTrainSession(JNIEnv *env, jobject thiz,
                                                                                          jstring file_name,
                                                                                          jlong ms_context_ptr,
                                                                                          jboolean train_mode,
                                                                                          jlong train_config_ptr) {
  auto *pointer = reinterpret_cast<void *>(ms_context_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Context pointer from java is nullptr");
    return jlong(nullptr);
  }
  auto *lite_context_ptr = static_cast<mindspore::lite::Context *>(pointer);

  auto session = mindspore::session::TrainSession::CreateTrainSession(env->GetStringUTFChars(file_name, JNI_FALSE),
                                                                     lite_context_ptr, train_mode, nullptr);
  if (session == nullptr) {
    MS_LOGE("CreateTrainSession failed");
    return jlong(nullptr);
  }
  return jlong(session);
}
