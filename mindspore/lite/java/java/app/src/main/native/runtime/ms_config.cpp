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
#include "include/context.h"

extern "C" JNIEXPORT jlong JNICALL Java_com_mindspore_lite_config_MSConfig_createMSConfig(
  JNIEnv *env, jobject thiz, jint device_type, jint thread_num, jint cpu_bind_mode, jboolean enable_float16) {
  auto *context = new (std::nothrow) mindspore::lite::Context();
  if (context == nullptr) {
    MS_LOGE("new Context fail!");
    return (jlong)context;
  }

  auto &cpu_device_ctx = context->device_list_[0];
  switch (device_type) {
    case 0:
      context->device_list_[0].device_type_ = mindspore::lite::DT_CPU;
      break;
    case 1:  // DT_GPU
    {
      mindspore::lite::DeviceContext gpu_device_ctx{mindspore::lite::DT_GPU, {false}};
      gpu_device_ctx.device_info_.gpu_device_info_.enable_float16_ = enable_float16;
      context->device_list_.push_back(gpu_device_ctx);
      break;
    }
    case 2:  // DT_NPU
      MS_LOGE("We only support CPU and GPU now.");
      return (jlong)context;
      break;
    default:
      MS_LOGE("Invalid device_type : %d", device_type);
      return (jlong)context;
  }

  switch (cpu_bind_mode) {
    case 0:
      cpu_device_ctx.device_info_.cpu_device_info_.cpu_bind_mode_ = mindspore::lite::NO_BIND;
      break;
    case 1:
      cpu_device_ctx.device_info_.cpu_device_info_.cpu_bind_mode_ = mindspore::lite::HIGHER_CPU;
      break;
    case 2:
      cpu_device_ctx.device_info_.cpu_device_info_.cpu_bind_mode_ = mindspore::lite::MID_CPU;
      break;
    default:
      MS_LOGE("Invalid cpu_bind_mode : %d", cpu_bind_mode);
      return (jlong)context;
  }

  cpu_device_ctx.device_info_.cpu_device_info_.enable_float16_ = enable_float16;

  context->thread_num_ = thread_num;
  return (jlong)context;
}

extern "C" JNIEXPORT void JNICALL Java_com_mindspore_lite_config_MSConfig_free(JNIEnv *env, jobject thiz,
                                                                               jlong context_ptr) {
  auto *pointer = reinterpret_cast<void *>(context_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Context pointer from java is nullptr");
    return;
  }
  auto *lite_context_ptr = static_cast<mindspore::lite::Context *>(pointer);
  delete (lite_context_ptr);
}
