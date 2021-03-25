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
#include "include/lite_session.h"
#include "include/errorcode.h"

extern "C" JNIEXPORT jlong JNICALL Java_com_mindspore_lite_LiteSession_createSession(JNIEnv *env, jobject thiz,
                                                                                     jlong ms_config_ptr) {
  auto *pointer = reinterpret_cast<void *>(ms_config_ptr);
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

extern "C" JNIEXPORT jlong JNICALL Java_com_mindspore_lite_LiteSession_getInputsByTensorName(JNIEnv *env, jobject thiz,
                                                                                             jlong session_ptr,
                                                                                             jstring tensor_name) {
  auto *pointer = reinterpret_cast<void *>(session_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Session pointer from java is nullptr");
    return jlong(nullptr);
  }
  auto *lite_session_ptr = static_cast<mindspore::session::LiteSession *>(pointer);
  auto input = lite_session_ptr->GetInputsByTensorName(env->GetStringUTFChars(tensor_name, JNI_FALSE));
  return jlong(input);
}

extern "C" JNIEXPORT jobject JNICALL Java_com_mindspore_lite_LiteSession_getOutputsByNodeName(JNIEnv *env, jobject thiz,
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
  auto inputs = lite_session_ptr->GetOutputsByNodeName(env->GetStringUTFChars(node_name, JNI_FALSE));
  for (auto input : inputs) {
    jobject tensor_addr = env->NewObject(long_object, long_object_construct, jlong(input));
    env->CallBooleanMethod(ret, array_list_add, tensor_addr);
    env->DeleteLocalRef(tensor_addr);
  }
  return ret;
}

extern "C" JNIEXPORT jobject JNICALL Java_com_mindspore_lite_LiteSession_getOutputMapByTensor(JNIEnv *env, jobject thiz,
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
  for (const auto &output_iter : outputs) {
    auto node_name = output_iter.first;
    auto ms_tensor = output_iter.second;
    jobject tensor_addr = env->NewObject(long_object, long_object_construct, jlong(ms_tensor));
    env->CallObjectMethod(hash_map, hash_map_put, env->NewStringUTF(node_name.c_str()), tensor_addr);
    env->DeleteLocalRef(tensor_addr);
  }
  return hash_map;
}

extern "C" JNIEXPORT jobject JNICALL Java_com_mindspore_lite_LiteSession_getOutputTensorNames(JNIEnv *env, jobject thiz,
                                                                                              jlong session_ptr) {
  jclass array_list = env->FindClass("java/util/ArrayList");
  jmethodID array_list_construct = env->GetMethodID(array_list, "<init>", "()V");
  jobject ret = env->NewObject(array_list, array_list_construct);
  jmethodID array_list_add = env->GetMethodID(array_list, "add", "(Ljava/lang/Object;)Z");

  auto *pointer = reinterpret_cast<void *>(session_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Session pointer from java is nullptr");
    return ret;
  }
  auto *lite_session_ptr = static_cast<mindspore::session::LiteSession *>(pointer);
  auto output_names = lite_session_ptr->GetOutputTensorNames();
  for (const auto &output_name : output_names) {
    env->CallBooleanMethod(ret, array_list_add, env->NewStringUTF(output_name.c_str()));
  }
  return ret;
}

extern "C" JNIEXPORT jlong JNICALL Java_com_mindspore_lite_LiteSession_getOutputByTensorName(JNIEnv *env, jobject thiz,
                                                                                             jlong session_ptr,
                                                                                             jstring tensor_name) {
  auto *pointer = reinterpret_cast<void *>(session_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Session pointer from java is nullptr");
    return jlong(nullptr);
  }
  auto *lite_session_ptr = static_cast<mindspore::session::LiteSession *>(pointer);
  auto output = lite_session_ptr->GetOutputByTensorName(env->GetStringUTFChars(tensor_name, JNI_FALSE));
  return jlong(output);
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

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_lite_LiteSession_resize(JNIEnv *env, jobject thiz,
                                                                                 jlong session_ptr, jlongArray inputs,
                                                                                 jobjectArray dims) {
  std::vector<std::vector<int>> c_dims;
  auto *pointer = reinterpret_cast<void *>(session_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Session pointer from java is nullptr");
    return false;
  }
  auto *lite_session_ptr = static_cast<mindspore::session::LiteSession *>(pointer);

  auto input_size = static_cast<int>(env->GetArrayLength(inputs));
  jlong *input_data = env->GetLongArrayElements(inputs, nullptr);
  std::vector<mindspore::tensor::MSTensor *> c_inputs;
  for (int i = 0; i < input_size; i++) {
    auto *tensor_pointer = reinterpret_cast<void *>(input_data[i]);
    if (tensor_pointer == nullptr) {
      MS_LOGE("Tensor pointer from java is nullptr");
      return false;
    }
    auto *ms_tensor_ptr = static_cast<mindspore::tensor::MSTensor *>(tensor_pointer);
    c_inputs.push_back(ms_tensor_ptr);
  }
  auto tensor_size = static_cast<int>(env->GetArrayLength(dims));
  for (int i = 0; i < tensor_size; i++) {
    auto array = static_cast<jintArray>(env->GetObjectArrayElement(dims, i));
    auto dim_size = static_cast<int>(env->GetArrayLength(array));
    jint *dim_data = env->GetIntArrayElements(array, nullptr);
    std::vector<int> tensor_dims(dim_size);
    for (int j = 0; j < dim_size; j++) {
      tensor_dims[j] = dim_data[j];
    }
    c_dims.push_back(tensor_dims);
    env->ReleaseIntArrayElements(array, dim_data, JNI_ABORT);
    env->DeleteLocalRef(array);
  }
  int ret = lite_session_ptr->Resize(c_inputs, c_dims);
  return (jboolean)(ret == mindspore::lite::RET_OK);
}
