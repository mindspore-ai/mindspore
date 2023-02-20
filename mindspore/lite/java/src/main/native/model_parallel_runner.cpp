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
#include "common/log_adapter.h"
#include "include/api/model_parallel_runner.h"
#include "include/api/context.h"

extern "C" JNIEXPORT jlong JNICALL Java_com_mindspore_ModelParallelRunner_init(JNIEnv *env, jobject thiz,
                                                                               jstring model_path,
                                                                               jlong runner_config_ptr) {
  auto runner = new (std::nothrow) mindspore::ModelParallelRunner();
  if (runner == nullptr) {
    MS_LOG(ERROR) << "Make ModelParallelRunner failed";
    return (jlong) nullptr;
  }
  const char *c_model_path = env->GetStringUTFChars(model_path, nullptr);
  std::string str_model_path(c_model_path, env->GetStringLength(model_path));
  env->ReleaseStringUTFChars(model_path, c_model_path);
  if (runner_config_ptr == 0L) {
    auto ret = runner->Init(str_model_path);
    if (ret != mindspore::kSuccess) {
      delete runner;
      return (jlong) nullptr;
    }
  } else {
    auto *c_runner_config = reinterpret_cast<mindspore::RunnerConfig *>(runner_config_ptr);
    auto origin_context = c_runner_config->GetContext();
    if (origin_context == nullptr) {
      auto ret = runner->Init(str_model_path);
      if (ret != mindspore::kSuccess) {
        delete runner;
        return (jlong) nullptr;
      }
      return (jlong)runner;
    }
    auto copy_context = std::make_shared<mindspore::Context>();
    if (copy_context == nullptr) {
      MS_LOG(ERROR) << "new context is nullptr.";
      delete runner;
      return (jlong) nullptr;
    }
    copy_context->SetThreadNum(origin_context->GetThreadNum());
    copy_context->SetInterOpParallelNum(origin_context->GetInterOpParallelNum());
    copy_context->SetThreadAffinity(origin_context->GetThreadAffinityMode());
    copy_context->SetThreadAffinity(origin_context->GetThreadAffinityCoreList());
    auto &copy_device_list = copy_context->MutableDeviceInfo();
    auto origin_device_list = origin_context->MutableDeviceInfo();
    for (size_t i = 0; i < origin_device_list.size(); i++) {
      auto origin_device = origin_device_list[i];
      if (origin_device->GetDeviceType() == mindspore::kCPU) {
        auto origin_device_info = origin_device->Cast<mindspore::CPUDeviceInfo>();
        auto copy_device_info = std::make_shared<mindspore::CPUDeviceInfo>();
        if (copy_device_info == nullptr) {
          MS_LOG(ERROR) << "new copy_device_info is nullptr.";
          delete runner;
          return (jlong) nullptr;
        }
        auto enable_fp16 = origin_device_info->GetEnableFP16();
        copy_device_info->SetEnableFP16(enable_fp16);
        copy_device_list.push_back(copy_device_info);
      } else if (origin_device->GetDeviceType() == mindspore::kGPU) {
        auto origin_device_info = origin_device->Cast<mindspore::GPUDeviceInfo>();
        auto copy_device_info = std::make_shared<mindspore::GPUDeviceInfo>();
        if (copy_device_info == nullptr) {
          MS_LOG(ERROR) << "new copy_device_info is nullptr.";
          delete runner;
          return (jlong) nullptr;
        }
        copy_device_info->SetEnableFP16(origin_device_info->GetEnableFP16());
        copy_device_info->SetDeviceID(origin_device_info->GetDeviceID());
        copy_device_list.push_back(copy_device_info);
      } else if (origin_device->GetDeviceType() == mindspore::kAscend) {
        auto origin_device_info = origin_device->Cast<mindspore::AscendDeviceInfo>();
        auto copy_device_info = std::make_shared<mindspore::AscendDeviceInfo>();
        if (copy_device_info == nullptr) {
          delete runner;
          return (jlong) nullptr;
        }
        copy_device_info->SetDeviceID(origin_device_info->GetDeviceID());
        copy_device_info->SetInsertOpConfigPath(origin_device_info->GetInsertOpConfigPath());
        copy_device_info->SetInputFormat(origin_device_info->GetInputFormat());
        copy_device_list.push_back(copy_device_info);
      }
    }
    auto runner_config = std::make_shared<mindspore::RunnerConfig>();
    if (runner_config == nullptr) {
      delete runner;
      MS_LOG(ERROR) << "Make RunnerConfig failed";
      return (jlong) nullptr;
    }
    runner_config->SetContext(copy_context);
    runner_config->SetWorkersNum(c_runner_config->GetWorkersNum());
    runner_config->SetConfigPath(c_runner_config->GetConfigPath());
    auto config_info = c_runner_config->GetConfigInfo();
    for (auto &item : config_info) {
      runner_config->SetConfigInfo(item.first, item.second);
    }
    auto ret = runner->Init(str_model_path, runner_config);
    if (ret != mindspore::kSuccess) {
      delete runner;
      return (jlong) nullptr;
    }
  }
  return (jlong)runner;
}

jobject GetParallelInOrOutTensors(JNIEnv *env, jobject thiz, jlong model_parallel_runner_ptr, bool is_input) {
  jclass array_list = env->FindClass("java/util/ArrayList");
  jmethodID array_list_construct = env->GetMethodID(array_list, "<init>", "()V");
  jobject ret = env->NewObject(array_list, array_list_construct);
  jmethodID array_list_add = env->GetMethodID(array_list, "add", "(Ljava/lang/Object;)Z");
  env->DeleteLocalRef(array_list);

  jclass long_object = env->FindClass("java/lang/Long");
  jmethodID long_object_construct = env->GetMethodID(long_object, "<init>", "(J)V");
  auto *pointer = reinterpret_cast<mindspore::ModelParallelRunner *>(model_parallel_runner_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Model pointer from java is nullptr";
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
      MS_LOG(ERROR) << "Make ms tensor failed";
      return ret;
    }
    jobject tensor_addr = env->NewObject(long_object, long_object_construct, jlong(tensor_ptr.release()));
    env->CallBooleanMethod(ret, array_list_add, tensor_addr);
    env->DeleteLocalRef(tensor_addr);
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

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_ModelParallelRunner_predictWithOutput(
  JNIEnv *env, jobject thiz, jlong model_parallel_runner_ptr, jlongArray inputs, jlongArray outputs) {
  auto *pointer = reinterpret_cast<mindspore::ModelParallelRunner *>(model_parallel_runner_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Model runner pointer from java is nullptr";
    return false;
  }

  auto input_size = static_cast<int>(env->GetArrayLength(inputs));
  jlong *input_data = env->GetLongArrayElements(inputs, nullptr);
  std::vector<mindspore::MSTensor> c_inputs;
  for (int i = 0; i < input_size; i++) {
    auto *tensor_pointer = reinterpret_cast<void *>(input_data[i]);
    if (tensor_pointer == nullptr) {
      MS_LOG(ERROR) << "input tensor pointer from java is nullptr";
      return false;
    }
    auto &ms_tensor = *static_cast<mindspore::MSTensor *>(tensor_pointer);
    c_inputs.push_back(ms_tensor);
  }
  env->ReleaseLongArrayElements(inputs, input_data, JNI_ABORT);

  auto output_size = static_cast<int>(env->GetArrayLength(outputs));
  jlong *output_data = env->GetLongArrayElements(outputs, nullptr);
  std::vector<mindspore::MSTensor> c_outputs;
  for (int i = 0; i < output_size; i++) {
    auto *tensor_pointer = reinterpret_cast<void *>(output_data[i]);
    if (tensor_pointer == nullptr) {
      MS_LOG(ERROR) << "output tensor pointer from java is nullptr";
      return false;
    }
    auto &ms_tensor = *static_cast<mindspore::MSTensor *>(tensor_pointer);
    c_outputs.push_back(ms_tensor);
  }
  env->ReleaseLongArrayElements(outputs, output_data, JNI_ABORT);

  auto ret = pointer->Predict(c_inputs, &c_outputs);
  if (ret != mindspore::kSuccess) {
    MS_LOG(ERROR) << "predict failed.";
    return false;
  }
  return true;
}

extern "C" JNIEXPORT jobject JNICALL Java_com_mindspore_ModelParallelRunner_predict(JNIEnv *env, jobject thiz,
                                                                                    jlong model_parallel_runner_ptr,
                                                                                    jlongArray inputs) {
  jclass array_list = env->FindClass("java/util/ArrayList");
  jmethodID array_list_construct = env->GetMethodID(array_list, "<init>", "()V");
  jobject ret = env->NewObject(array_list, array_list_construct);
  jmethodID array_list_add = env->GetMethodID(array_list, "add", "(Ljava/lang/Object;)Z");
  env->DeleteLocalRef(array_list);
  auto *pointer = reinterpret_cast<mindspore::ModelParallelRunner *>(model_parallel_runner_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Model pointer from java is nullptr";
    return ret;
  }

  auto input_size = static_cast<int>(env->GetArrayLength(inputs));
  jlong *input_data = env->GetLongArrayElements(inputs, nullptr);
  std::vector<mindspore::MSTensor> c_inputs;
  for (int i = 0; i < input_size; i++) {
    auto *tensor_pointer = reinterpret_cast<void *>(input_data[i]);
    if (tensor_pointer == nullptr) {
      MS_LOG(ERROR) << "Tensor pointer from java is nullptr";
      return ret;
    }
    auto &ms_tensor = *static_cast<mindspore::MSTensor *>(tensor_pointer);
    c_inputs.push_back(ms_tensor);
  }
  env->ReleaseLongArrayElements(inputs, input_data, JNI_ABORT);
  std::vector<mindspore::MSTensor> outputs;
  pointer->Predict(c_inputs, &outputs);
  for (auto &tensor : outputs) {
    auto tensor_ptr = std::make_unique<mindspore::MSTensor>(tensor);
    if (tensor_ptr == nullptr) {
      MS_LOG(ERROR) << "Make ms tensor failed";
      return ret;
    }
    jclass long_object = env->FindClass("java/lang/Long");
    jmethodID long_object_construct = env->GetMethodID(long_object, "<init>", "(J)V");
    jobject tensor_addr = env->NewObject(long_object, long_object_construct, jlong(tensor_ptr.release()));
    env->DeleteLocalRef(long_object);
    env->CallBooleanMethod(ret, array_list_add, tensor_addr);
    env->DeleteLocalRef(tensor_addr);
  }
  return ret;
}

extern "C" JNIEXPORT void JNICALL Java_com_mindspore_ModelParallelRunner_free(JNIEnv *env, jobject thiz,
                                                                              jlong model_parallel_runner_ptr) {
  auto *pointer = reinterpret_cast<mindspore::ModelParallelRunner *>(model_parallel_runner_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "ModelParallelRunner pointer from java is nullptr";
    return;
  }
  delete pointer;
}
