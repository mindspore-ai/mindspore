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
#include "common/jni_utils.h"
#include "include/train/train_session.h"
#include "include/errorcode.h"

extern "C" JNIEXPORT jlong JNICALL Java_com_mindspore_lite_TrainSession_createSession(JNIEnv *env, jobject thiz,
                                                                                      jstring model_file_name,
                                                                                      jlong ms_config_ptr) {
  auto *pointer = reinterpret_cast<void *>(ms_config_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Context pointer from java is nullptr");
    return jlong(nullptr);
  }
  auto *lite_context_ptr = static_cast<mindspore::lite::Context *>(pointer);
  auto session = mindspore::session::TrainSession::CreateSession(JstringToChar(env, model_file_name), lite_context_ptr);
  if (session == nullptr) {
    MS_LOGE("CreateSession failed");
    return jlong(nullptr);
  }
  return jlong(session);
}

extern "C" JNIEXPORT void JNICALL Java_com_mindspore_lite_TrainSession_bindThread(JNIEnv *env, jobject thiz,
                                                                                  jlong session_ptr, jboolean if_bind) {
  auto *pointer = reinterpret_cast<void *>(session_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Session pointer from java is nullptr");
    return;
  }
  auto *train_session_ptr = static_cast<mindspore::session::TrainSession *>(pointer);
  train_session_ptr->BindThread(if_bind);
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_lite_TrainSession_runGraph(JNIEnv *env, jobject thiz,
                                                                                    jlong session_ptr) {
  auto *pointer = reinterpret_cast<void *>(session_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Session pointer from java is nullptr");
    return (jboolean) false;
  }
  auto *train_session_ptr = static_cast<mindspore::session::TrainSession *>(pointer);
  auto ret = train_session_ptr->RunGraph();
  return (jboolean)(ret == mindspore::lite::RET_OK);
}

extern "C" JNIEXPORT jobject JNICALL Java_com_mindspore_lite_TrainSession_getInputs(JNIEnv *env, jobject thiz,
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
  auto *train_session_ptr = static_cast<mindspore::session::TrainSession *>(pointer);
  auto inputs = train_session_ptr->GetInputs();
  for (auto input : inputs) {
    jobject tensor_addr = env->NewObject(long_object, long_object_construct, jlong(input));
    env->CallBooleanMethod(ret, array_list_add, tensor_addr);
  }
  return ret;
}

extern "C" JNIEXPORT jlong JNICALL Java_com_mindspore_lite_TrainSession_getInputsByTensorName(JNIEnv *env, jobject thiz,
                                                                                              jlong session_ptr,
                                                                                              jstring tensor_name) {
  auto *pointer = reinterpret_cast<void *>(session_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Session pointer from java is nullptr");
    return jlong(nullptr);
  }
  auto *train_session_ptr = static_cast<mindspore::session::TrainSession *>(pointer);
  auto input = train_session_ptr->GetInputsByTensorName(JstringToChar(env, tensor_name));
  return jlong(input);
}

extern "C" JNIEXPORT jobject JNICALL Java_com_mindspore_lite_TrainSession_getOutputsByNodeName(JNIEnv *env,
                                                                                               jobject thiz,
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
  auto *train_session_ptr = static_cast<mindspore::session::TrainSession *>(pointer);
  auto inputs = train_session_ptr->GetOutputsByNodeName(JstringToChar(env, node_name));
  for (auto input : inputs) {
    jobject tensor_addr = env->NewObject(long_object, long_object_construct, jlong(input));
    env->CallBooleanMethod(ret, array_list_add, tensor_addr);
  }
  return ret;
}

extern "C" JNIEXPORT jobject JNICALL Java_com_mindspore_lite_TrainSession_getOutputMapByTensor(JNIEnv *env,
                                                                                               jobject thiz,
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
  auto *train_session_ptr = static_cast<mindspore::session::TrainSession *>(pointer);
  auto outputs = train_session_ptr->GetOutputs();
  jclass long_object = env->FindClass("java/lang/Long");
  jmethodID long_object_construct = env->GetMethodID(long_object, "<init>", "(J)V");
  for (auto output_iter : outputs) {
    auto node_name = output_iter.first;
    auto ms_tensor = output_iter.second;
    jobject tensor_addr = env->NewObject(long_object, long_object_construct, jlong(ms_tensor));
    env->CallObjectMethod(hash_map, hash_map_put, env->NewStringUTF(node_name.c_str()), tensor_addr);
  }
  return hash_map;
}

extern "C" JNIEXPORT jobject JNICALL Java_com_mindspore_lite_TrainSession_getOutputTensorNames(JNIEnv *env,
                                                                                               jobject thiz,
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
  auto *train_session_ptr = static_cast<mindspore::session::TrainSession *>(pointer);
  auto output_names = train_session_ptr->GetOutputTensorNames();
  for (auto output_name : output_names) {
    env->CallBooleanMethod(ret, array_list_add, env->NewStringUTF(output_name.c_str()));
  }
  return ret;
}

extern "C" JNIEXPORT jlong JNICALL Java_com_mindspore_lite_TrainSession_getOutputByTensorName(JNIEnv *env, jobject thiz,
                                                                                              jlong session_ptr,
                                                                                              jstring tensor_name) {
  auto *pointer = reinterpret_cast<void *>(session_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Session pointer from java is nullptr");
    return jlong(nullptr);
  }
  auto *train_session_ptr = static_cast<mindspore::session::TrainSession *>(pointer);
  auto output = train_session_ptr->GetOutputByTensorName(JstringToChar(env, tensor_name));
  return jlong(output);
}

extern "C" JNIEXPORT void JNICALL Java_com_mindspore_lite_TrainSession_free(JNIEnv *env, jobject thiz,
                                                                            jlong session_ptr) {
  auto *pointer = reinterpret_cast<void *>(session_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Session pointer from java is nullptr");
    return;
  }
  auto *train_session_ptr = static_cast<mindspore::session::TrainSession *>(pointer);
  delete (train_session_ptr);
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_lite_TrainSession_resize(JNIEnv *env, jobject thiz,
                                                                                  jlong session_ptr, jlongArray inputs,
                                                                                  jobjectArray dims) {
  std::vector<std::vector<int>> c_dims;
  auto *pointer = reinterpret_cast<void *>(session_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Session pointer from java is nullptr");
    return false;
  }
  auto *train_session_ptr = static_cast<mindspore::session::TrainSession *>(pointer);

  jsize input_size = static_cast<int>(env->GetArrayLength(inputs));
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
  jsize tensor_size = static_cast<int>(env->GetArrayLength(dims));
  for (int i = 0; i < tensor_size; i++) {
    jintArray array = static_cast<jintArray>(env->GetObjectArrayElement(dims, i));
    jsize dim_size = static_cast<int>(env->GetArrayLength(array));
    jint *dim_data = env->GetIntArrayElements(array, nullptr);
    std::vector<int> tensor_dims;
    for (int j = 0; j < dim_size; j++) {
      tensor_dims.push_back(dim_data[j]);
    }
    c_dims.push_back(tensor_dims);
  }
  int ret = train_session_ptr->Resize(c_inputs, c_dims);
  return (jboolean)(ret == mindspore::lite::RET_OK);
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_lite_TrainSession_saveToFile(JNIEnv *env, jobject thiz,
                                                                                      jlong session_ptr,
                                                                                      jstring model_file_name) {
  auto *session_pointer = reinterpret_cast<void *>(session_ptr);
  if (session_pointer == nullptr) {
    MS_LOGE("Session pointer from java is nullptr");
    return (jboolean) false;
  }
  auto *train_session_ptr = static_cast<mindspore::session::TrainSession *>(session_pointer);
  auto ret = train_session_ptr->SaveToFile(JstringToChar(env, model_file_name));
  return (jboolean)(ret == 0);
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_lite_TrainSession_train(JNIEnv *env, jobject thiz,
                                                                                 jlong session_ptr) {
  auto *session_pointer = reinterpret_cast<void *>(session_ptr);
  if (session_pointer == nullptr) {
    MS_LOGE("Session pointer from java is nullptr");
    return (jboolean) false;
  }
  auto *train_session_ptr = static_cast<mindspore::session::TrainSession *>(session_pointer);
  auto ret = train_session_ptr->Train();
  return (jboolean)(ret == mindspore::lite::RET_OK);
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_lite_TrainSession_eval(JNIEnv *env, jobject thiz,
                                                                                jlong session_ptr) {
  auto *session_pointer = reinterpret_cast<void *>(session_ptr);
  if (session_pointer == nullptr) {
    MS_LOGE("Session pointer from java is nullptr");
    return (jboolean) false;
  }
  auto *train_session_ptr = static_cast<mindspore::session::TrainSession *>(session_pointer);
  auto ret = train_session_ptr->Eval();
  return (jboolean)(ret == mindspore::lite::RET_OK);
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_lite_TrainSession_isTrain(JNIEnv *env, jobject thiz,
                                                                                   jlong session_ptr) {
  auto *session_pointer = reinterpret_cast<void *>(session_ptr);
  if (session_pointer == nullptr) {
    MS_LOGE("Session pointer from java is nullptr");
    return (jboolean) false;
  }
  auto *train_session_ptr = static_cast<mindspore::session::TrainSession *>(session_pointer);
  auto ret = train_session_ptr->IsTrain();
  return (jboolean)(ret);
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_lite_TrainSession_isEval(JNIEnv *env, jobject thiz,
                                                                                  jlong session_ptr) {
  auto *session_pointer = reinterpret_cast<void *>(session_ptr);
  if (session_pointer == nullptr) {
    MS_LOGE("Session pointer from java is nullptr");
    return (jboolean) false;
  }
  auto *train_session_ptr = static_cast<mindspore::session::TrainSession *>(session_pointer);
  auto ret = train_session_ptr->IsEval();
  return (jboolean)(ret);
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_lite_TrainSession_setLearningRate(JNIEnv *env, jobject thiz,
                                                                                           jlong session_ptr,
                                                                                           jfloat learning_rate) {
  auto *session_pointer = reinterpret_cast<void *>(session_ptr);
  if (session_pointer == nullptr) {
    MS_LOGE("Session pointer from java is nullptr");
    return (jboolean) false;
  }
  auto *train_session_ptr = static_cast<mindspore::session::TrainSession *>(session_pointer);
  auto ret = train_session_ptr->SetLearningRate(learning_rate);
  return (jboolean)(ret == mindspore::lite::RET_OK);
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_lite_TrainSession_setupVirtualBatch(JNIEnv *env, jobject thiz,
                                                                                           jlong session_ptr,
                                                                                           jint virtualBatchMultiplier,
                                                                                           jfloat learningRate,
                                                                                           jfloat momentum) {
  auto *session_pointer = reinterpret_cast<void *>(session_ptr);
  if (session_pointer == nullptr) {
    MS_LOGE("Session pointer from java is nullptr");
    return (jboolean) false;
  }
  auto *train_session_ptr = static_cast<mindspore::session::TrainSession *>(session_pointer);
  auto ret = train_session_ptr->SetupVirtualBatch(virtualBatchMultiplier, learningRate, momentum);
  return (jboolean)(ret == mindspore::lite::RET_OK);
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_lite_TrainSession_setLossName(JNIEnv *env, jobject thiz,
                                                                                       jlong session_ptr,
                                                                                       jstring lossName) {
  auto *session_pointer = reinterpret_cast<void *>(session_ptr);
  if (session_pointer == nullptr) {
    MS_LOGE("Session pointer from java is nullptr");
    return (jboolean) false;
  }
  auto *train_session_ptr = static_cast<mindspore::session::TrainSession *>(session_pointer);
  auto ret = train_session_ptr->SetLossName(JstringToChar(env, lossName));
  return (jboolean)(ret == mindspore::lite::RET_OK);
}
