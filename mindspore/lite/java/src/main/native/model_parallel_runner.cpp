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

extern "C" JNIEXPORT jlong JNICALL Java_com_mindspore_ModelParallelRunner_init(JNIEnv *env, jobject thiz,
                                                                               jstring model_path,
                                                                               jlong runner_config_ptr) {
  auto runner = new (std::nothrow) mindspore::ModelParallelRunner();
  if (runner == nullptr) {
    MS_LOGE("Make ModelParallelRunner failed");
    return (jlong) nullptr;
  }
  auto model_path_str = env->GetStringUTFChars(model_path, JNI_FALSE);
  if (runner_config_ptr == 0L) {
    runner->Init(model_path_str);
  } else {
    auto *c_runner_config = reinterpret_cast<mindspore::RunnerConfig *>(runner_config_ptr);
    auto runner_config = std::make_shared<mindspore::RunnerConfig>();
    if (runner_config == nullptr) {
      delete runner;
      MS_LOGE("Make RunnerConfig failed");
      return (jlong) nullptr;
    }
    runner_config.reset(c_runner_config);
    runner->Init(model_path_str, runner_config);
  }
  return (jlong)runner;
}

jobject GetParallelInOrOutTensors(JNIEnv *env, jobject thiz, jlong model_parallel_runner_ptr, bool is_input) {
  jclass array_list = env->FindClass("java/util/ArrayList");
  jmethodID array_list_construct = env->GetMethodID(array_list, "<init>", "()V");
  jobject ret = env->NewObject(array_list, array_list_construct);
  jmethodID array_list_add = env->GetMethodID(array_list, "add", "(Ljava/lang/Object;)Z");

  jclass long_object = env->FindClass("java/lang/Long");
  jmethodID long_object_construct = env->GetMethodID(long_object, "<init>", "(J)V");
  auto *pointer = reinterpret_cast<mindspore::ModelParallelRunner *>(model_parallel_runner_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Model pointer from java is nullptr");
    return ret;
  }
  std::vector<mindspore::MSTensor> tensors;
  if (is_input) {
    tensors = pointer->GetInputs();
  } else {
    tensors = pointer->GetOutputs();
  }
  for (auto &tensor : tensors) {
    auto tensor_ptr = std::make_unique<mindspore::MSTensor>(tensor);
    if (tensor_ptr == nullptr) {
      MS_LOGE("Make ms tensor failed");
      return ret;
    }
    jobject tensor_addr = env->NewObject(long_object, long_object_construct, jlong(tensor_ptr.release()));
    env->CallBooleanMethod(ret, array_list_add, tensor_addr);
  }
  return ret;
}

extern "C" JNIEXPORT jobject JNICALL Java_com_mindspore_ModelParallelRunner_getInputs(JNIEnv *env, jobject thiz,
                                                                                      jlong model_parallel_runner_ptr) {
  return GetParallelInOrOutTensors(env, thiz, model_parallel_runner_ptr, true);
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_mindspore_ModelParallelRunner_getOutputs(JNIEnv *env, jobject thiz, jlong model_parallel_runner_ptr) {
  return GetParallelInOrOutTensors(env, thiz, model_parallel_runner_ptr, false);
}

extern "C" JNIEXPORT jobject JNICALL Java_com_mindspore_ModelParallelRunner_predict(JNIEnv *env, jobject thiz,
                                                                                    jlong model_parallel_runner_ptr,
                                                                                    jlongArray inputs) {
  jclass array_list = env->FindClass("java/util/ArrayList");
  jmethodID array_list_construct = env->GetMethodID(array_list, "<init>", "()V");
  jobject ret = env->NewObject(array_list, array_list_construct);
  jmethodID array_list_add = env->GetMethodID(array_list, "add", "(Ljava/lang/Object;)Z");
  auto *pointer = reinterpret_cast<mindspore::ModelParallelRunner *>(model_parallel_runner_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Model pointer from java is nullptr");
    return ret;
  }

  auto input_size = static_cast<int>(env->GetArrayLength(inputs));
  jlong *input_data = env->GetLongArrayElements(inputs, nullptr);
  std::vector<mindspore::MSTensor> c_inputs;
  for (int i = 0; i < input_size; i++) {
    auto *tensor_pointer = reinterpret_cast<void *>(input_data[i]);
    if (tensor_pointer == nullptr) {
      MS_LOGE("Tensor pointer from java is nullptr");
      return ret;
    }
    auto &ms_tensor = *static_cast<mindspore::MSTensor *>(tensor_pointer);
    c_inputs.push_back(ms_tensor);
  }
  std::vector<mindspore::MSTensor> outputs;
  pointer->Predict(c_inputs, &outputs);
  for (auto &tensor : outputs) {
    auto tensor_ptr = std::make_unique<mindspore::MSTensor>(tensor);
    if (tensor_ptr == nullptr) {
      MS_LOGE("Make ms tensor failed");
      return ret;
    }
    jclass long_object = env->FindClass("java/lang/Long");
    jmethodID long_object_construct = env->GetMethodID(long_object, "<init>", "(J)V");
    jobject tensor_addr = env->NewObject(long_object, long_object_construct, jlong(tensor_ptr.release()));
    env->CallBooleanMethod(ret, array_list_add, tensor_addr);
  }
  return ret;
}

extern "C" JNIEXPORT void JNICALL Java_com_mindspore_ModelParallelRunner_free(JNIEnv *env, jobject thiz,
  jlong model_parallel_runner_ptr) {
auto *pointer = reinterpret_cast<mindspore::ModelParallelRunner *>(model_parallel_runner_ptr);
if (pointer == nullptr) {
MS_LOGE("ModelParallelRunner pointer from java is nullptr");
return;
}
delete pointer;
}
