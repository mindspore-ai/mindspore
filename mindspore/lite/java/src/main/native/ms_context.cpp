/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "include/api/context.h"

constexpr const int kNpuDevice = 2;
extern "C" JNIEXPORT jlong JNICALL Java_com_mindspore_config_MSContext_createMSContext(JNIEnv *env, jobject thiz,
                                                                                       jint thread_num,
                                                                                       jint cpu_bind_mode,
                                                                                       jboolean enable_parallel) {
  auto *context = new (std::nothrow) mindspore::Context();
  if (context == nullptr) {
    MS_LOGE("new Context fail!");
    return (jlong) nullptr;
  }
  context->SetThreadNum(thread_num);
  context->SetEnableParallel(enable_parallel);
  context->SetThreadAffinity(cpu_bind_mode);
  return (jlong)context;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_config_MSContext_addDeviceInfo(
  JNIEnv *env, jobject thiz, jlong context_ptr, jint device_type, jboolean enable_fp16, jint npu_freq) {
  auto *pointer = reinterpret_cast<void *>(context_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Context pointer from java is nullptr");
    return (jboolean) false;
  }
  auto *c_context_ptr = static_cast<mindspore::Context *>(pointer);
  auto &device_list = c_context_ptr->MutableDeviceInfo();
  switch (device_type) {
    case 0: {
      auto cpu_device_info = std::make_shared<mindspore::CPUDeviceInfo>();
      if (cpu_device_info == nullptr) {
        MS_LOGE("cpu device info is nullptr");
        delete (c_context_ptr);
        return (jboolean) false;
      }
      cpu_device_info->SetEnableFP16(enable_fp16);
      device_list.push_back(cpu_device_info);
      break;
    }
    case 1:  // DT_GPU
    {
      auto gpu_device_info = std::make_shared<mindspore::GPUDeviceInfo>();
      if (gpu_device_info == nullptr) {
        MS_LOGE("gpu device info is nullptr");
        delete (c_context_ptr);
        return (jboolean) false;
      }
      gpu_device_info->SetEnableFP16(enable_fp16);
      device_list.push_back(gpu_device_info);
      break;
    }
    case kNpuDevice:  // DT_NPU
    {
      auto npu_device_info = std::make_shared<mindspore::KirinNPUDeviceInfo>();
      if (npu_device_info == nullptr) {
        MS_LOGE("npu device info is nullptr");
        delete (c_context_ptr);
        return (jboolean) false;
      }
      npu_device_info->SetFrequency(npu_freq);
      device_list.push_back(npu_device_info);
      break;
    }
    default:
      MS_LOGE("Invalid device_type : %d", device_type);
      delete (c_context_ptr);
      return (jboolean) false;
  }
  return (jboolean) true;
}

extern "C" JNIEXPORT void JNICALL Java_com_mindspore_config_MSContext_free(JNIEnv *env, jobject thiz,
                                                                           jlong context_ptr) {
  auto *pointer = reinterpret_cast<void *>(context_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Context pointer from java is nullptr");
    return;
  }
  auto *c_context_ptr = static_cast<mindspore::Context *>(pointer);
  delete (c_context_ptr);
}
