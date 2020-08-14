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
#include "common/jni_utils.h"
#include "include/lite_session.h"
#include "include/errorcode.h"

extern "C" JNIEXPORT jlong JNICALL Java_com_mindspore_lite_LiteSession_createSession(JNIEnv *env, jobject thiz,
                                                                                      jlong context_ptr) {
  auto *pointer = reinterpret_cast<void *>(context_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Context pointer from java is nullptr");
    return jlong(nullptr);
  }
  auto *lite_context_ptr = static_cast<mindspore::lite::Context *>(pointer);
  auto session = mindspore::session::LiteSession::CreateSession(lite_context_ptr);
  if (session == nullptr) {
    MS_LOGE("CreateSession failed");
    return jlong(nullptr);
  }
  return jlong(session);
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_lite_LiteSession_compileGraph(JNIEnv *env, jobject thiz,
                                                                                        jlong session_ptr,
                                                                                        jlong model_ptr) {
  auto *session_pointer = reinterpret_cast<void *>(session_ptr);
  if (session_pointer == nullptr) {
    MS_LOGE("Session pointer from java is nullptr");
    return (jboolean) false;
  }
  auto *lite_session_ptr = static_cast<mindspore::session::LiteSession *>(session_pointer);
  auto *model_pointer = reinterpret_cast<void *>(model_ptr);
  if (model_pointer == nullptr) {
    MS_LOGE("Model pointer from java is nullptr");
    return (jboolean) false;
  }
  auto *lite_model_ptr = static_cast<mindspore::lite::Model *>(model_pointer);

  auto ret = lite_session_ptr->CompileGraph(lite_model_ptr);
  return (jboolean)(ret == mindspore::lite::RET_OK);
}

extern "C" JNIEXPORT void JNICALL Java_com_mindspore_lite_LiteSession_bindThread(JNIEnv *env, jobject thiz,
                                                                                  jlong session_ptr, jboolean if_bind) {
  auto *pointer = reinterpret_cast<void *>(session_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Session pointer from java is nullptr");
    return;
  }
  auto *lite_session_ptr = static_cast<mindspore::session::LiteSession *>(pointer);
  lite_session_ptr->BindThread(if_bind);
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_lite_LiteSession_runGraph(JNIEnv *env, jobject thiz,
                                                                                    jlong session_ptr) {
  auto *pointer = reinterpret_cast<void *>(session_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Session pointer from java is nullptr");
    return (jboolean) false;
  }
  auto *lite_session_ptr = static_cast<mindspore::session::LiteSession *>(pointer);
  auto ret = lite_session_ptr->RunGraph();
  return (jboolean)(ret == mindspore::lite::RET_OK);
}

extern "C" JNIEXPORT jobject JNICALL Java_com_mindspore_lite_LiteSession_getInputs(JNIEnv *env, jobject thiz,
                                                                                    jlong session_ptr) {
  jclass array_list = env->FindClass("java/util/ArrayList");
  jmethodID array_list_construct = env->GetMethodID(array_list, "<init>", "()V");
  jobject ret = env->NewObject(array_list, array_list_construct);
  jmethodID array_list_add = env->GetMethodID(array_list, "add", "(Ljava/lang/Object;)Z");

  jclass long_object = env->FindClass("java/lang/Long");
  jmethodID long_object_construct = env->GetMethodID(long_object, "<init>", "(J)V");
  auto *pointer = reinterpret_cast<void *>(session_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Session pointer from java is nullptr");
    return ret;
  }
  auto *lite_session_ptr = static_cast<mindspore::session::LiteSession *>(pointer);
  auto inputs = lite_session_ptr->GetInputs();
  for (auto input : inputs) {
    jobject tensor_addr = env->NewObject(long_object, long_object_construct, jlong(input));
    env->CallBooleanMethod(ret, array_list_add, tensor_addr);
  }
  return ret;
}

extern "C" JNIEXPORT jobject JNICALL Java_com_mindspore_lite_LiteSession_getInputsByName(JNIEnv *env, jobject thiz,
                                                                                          jlong session_ptr,
                                                                                          jstring node_name) {
  jclass array_list = env->FindClass("java/util/ArrayList");
  jmethodID array_list_construct = env->GetMethodID(array_list, "<init>", "()V");
  jobject ret = env->NewObject(array_list, array_list_construct);
  jmethodID array_list_add = env->GetMethodID(array_list, "add", "(Ljava/lang/Object;)Z");

  jclass long_object = env->FindClass("java/lang/Long");
  jmethodID long_object_construct = env->GetMethodID(long_object, "<init>", "(J)V");
  auto *pointer = reinterpret_cast<void *>(session_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Session pointer from java is nullptr");
    return ret;
  }
  auto *lite_session_ptr = static_cast<mindspore::session::LiteSession *>(pointer);
  auto inputs = lite_session_ptr->GetInputsByName(JstringToChar(env, node_name));
  for (auto input : inputs) {
    jobject tensor_addr = env->NewObject(long_object, long_object_construct, jlong(input));
    env->CallBooleanMethod(ret, array_list_add, tensor_addr);
  }
  return ret;
}

extern "C" JNIEXPORT jobject JNICALL Java_com_mindspore_lite_LiteSession_getOutputs(JNIEnv *env, jobject thiz,
                                                                                     jlong session_ptr) {
  jclass hash_map_clazz = env->FindClass("java/util/HashMap");
  jmethodID hash_map_construct = env->GetMethodID(hash_map_clazz, "<init>", "()V");
  jobject hash_map = env->NewObject(hash_map_clazz, hash_map_construct);
  jmethodID hash_map_put =
    env->GetMethodID(hash_map_clazz, "put", "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
  auto *pointer = reinterpret_cast<void *>(session_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Session pointer from java is nullptr");
    return hash_map;
  }
  auto *lite_session_ptr = static_cast<mindspore::session::LiteSession *>(pointer);
  auto outputs = lite_session_ptr->GetOutputs();
  jclass long_object = env->FindClass("java/lang/Long");
  jmethodID long_object_construct = env->GetMethodID(long_object, "<init>", "(J)V");
  jclass array_list = env->FindClass("java/util/ArrayList");
  jmethodID array_list_construct = env->GetMethodID(array_list, "<init>", "()V");
  jmethodID array_list_add = env->GetMethodID(array_list, "add", "(Ljava/lang/Object;)Z");
  for (auto output_iter : outputs) {
    auto node_name = output_iter.first;
    auto ms_tensors = output_iter.second;
    jobject vec = env->NewObject(array_list, array_list_construct);
    for (auto ms_tensor : ms_tensors) {
      jobject tensor_addr = env->NewObject(long_object, long_object_construct, jlong(ms_tensor));
      env->CallBooleanMethod(vec, array_list_add, tensor_addr);
    }
    env->CallObjectMethod(hash_map, hash_map_put, env->NewStringUTF(node_name.c_str()), vec);
  }
  return hash_map;
}

extern "C" JNIEXPORT jobject JNICALL Java_com_mindspore_lite_LiteSession_getOutputsByName(JNIEnv *env, jobject thiz,
                                                                                           jlong session_ptr,
                                                                                           jstring node_name) {
  jclass array_list = env->FindClass("java/util/ArrayList");
  jmethodID array_list_construct = env->GetMethodID(array_list, "<init>", "()V");
  jobject ret = env->NewObject(array_list, array_list_construct);
  jmethodID array_list_add = env->GetMethodID(array_list, "add", "(Ljava/lang/Object;)Z");

  jclass long_object = env->FindClass("java/lang/Long");
  jmethodID long_object_construct = env->GetMethodID(long_object, "<init>", "(J)V");
  auto *pointer = reinterpret_cast<void *>(session_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Session pointer from java is nullptr");
    return ret;
  }
  auto *lite_session_ptr = static_cast<mindspore::session::LiteSession *>(pointer);
  auto inputs = lite_session_ptr->GetOutputsByName(JstringToChar(env, node_name));
  for (auto input : inputs) {
    jobject tensor_addr = env->NewObject(long_object, long_object_construct, jlong(input));
    env->CallBooleanMethod(ret, array_list_add, tensor_addr);
  }
  return ret;
}

extern "C" JNIEXPORT void JNICALL Java_com_mindspore_lite_LiteSession_free(JNIEnv *env, jobject thiz,
                                                                            jlong session_ptr) {
  auto *pointer = reinterpret_cast<void *>(session_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Session pointer from java is nullptr");
    return;
  }
  auto *lite_session_ptr = static_cast<mindspore::session::LiteSession *>(pointer);
  delete (lite_session_ptr);
}
