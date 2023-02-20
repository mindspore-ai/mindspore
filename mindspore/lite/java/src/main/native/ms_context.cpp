/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "include/api/context.h"
#include "common/jni_utils.h"

constexpr const int kNpuDevice = 2;
constexpr const int kAscendDevice = 3;
extern "C" JNIEXPORT jlong JNICALL Java_com_mindspore_config_MSContext_createMSContext(JNIEnv *env, jobject thiz,
                                                                                       jint thread_num,
                                                                                       jint cpu_bind_mode,
                                                                                       jboolean enable_parallel) {
  auto *context = new (std::nothrow) mindspore::Context();
  if (context == nullptr) {
    MS_LOG(ERROR) << "new Context fail!";
    return (jlong) nullptr;
  }
  context->SetThreadNum(thread_num);
  context->SetEnableParallel(enable_parallel);
  context->SetThreadAffinity(cpu_bind_mode);
  return (jlong)context;
}

extern "C" JNIEXPORT jlong JNICALL Java_com_mindspore_config_MSContext_createDefaultMSContext(JNIEnv *env,
                                                                                              jobject thiz) {
  auto *context = new (std::nothrow) mindspore::Context();
  if (context == nullptr) {
    MS_LOG(ERROR) << "new Context fail!";
    return (jlong) nullptr;
  }
  return (jlong)context;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_config_MSContext_addDeviceInfo(
  JNIEnv *env, jobject thiz, jlong context_ptr, jint device_type, jboolean enable_fp16, jint npu_freq) {
  auto *pointer = reinterpret_cast<void *>(context_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Context pointer from java is nullptr";
    return (jboolean) false;
  }
  auto *c_context_ptr = static_cast<mindspore::Context *>(pointer);
  auto &device_list = c_context_ptr->MutableDeviceInfo();
  switch (device_type) {
    case 0: {
      auto cpu_device_info = std::make_shared<mindspore::CPUDeviceInfo>();
      if (cpu_device_info == nullptr) {
        MS_LOG(ERROR) << "cpu device info is nullptr";
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
        MS_LOG(ERROR) << "gpu device info is nullptr";
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
        MS_LOG(ERROR) << "npu device info is nullptr";
        delete (c_context_ptr);
        return (jboolean) false;
      }
      npu_device_info->SetFrequency(npu_freq);
      device_list.push_back(npu_device_info);
      break;
    }
    case kAscendDevice:  // DT_ASCEND
    {
      auto ascend_device_info = std::make_shared<mindspore::AscendDeviceInfo>();
      if (ascend_device_info == nullptr) {
        MS_LOG(ERROR) << "ascend device info is nullptr";
        delete (c_context_ptr);
        return (jboolean) false;
      }
      ascend_device_info->SetDeviceID(0);
      device_list.push_back(ascend_device_info);
      break;
    }
    default:
      MS_LOG(ERROR) << "Invalid device_type : " << device_type;
      delete (c_context_ptr);
      return (jboolean) false;
  }
  return (jboolean) true;
}

/*
 * Class:     com_mindspore_config_MSContext
 * Method:    SetThreadNum
 * Signature: (JI)V
 */
extern "C" JNIEXPORT void JNICALL Java_com_mindspore_config_MSContext_setThreadNum(JNIEnv *env, jobject thiz,
                                                                                   jlong context_ptr, jint thread_num) {
  auto *c_context_ptr = static_cast<mindspore::Context *>(reinterpret_cast<void *>(context_ptr));
  if (c_context_ptr == nullptr) {
    MS_LOG(ERROR) << "Context pointer from java is nullptr";
    return;
  }
  c_context_ptr->SetThreadNum(thread_num);
}

/*
 * Class:     com_mindspore_config_MSContext
 * Method:    GetThreadNum
 * Signature: (J)I
 */
extern "C" JNIEXPORT jint JNICALL Java_com_mindspore_config_MSContext_getThreadNum(JNIEnv *env, jobject thiz,
                                                                                   jlong context_ptr) {
  auto *c_context_ptr = static_cast<mindspore::Context *>(reinterpret_cast<void *>(context_ptr));
  if (c_context_ptr == nullptr) {
    MS_LOG(ERROR) << "Context pointer from java is nullptr";
    return 0;
  }
  int32_t thread_num = c_context_ptr->GetThreadNum();
  return thread_num;
}

/*
 * Class:     com_mindspore_config_MSContext
 * Method:    SetInterOpParallelNum
 * Signature: (JI)V
 */
extern "C" JNIEXPORT void JNICALL Java_com_mindspore_config_MSContext_setInterOpParallelNum(JNIEnv *env, jobject thiz,
                                                                                            jlong context_ptr,
                                                                                            jint op_parallel_num) {
  auto *c_context_ptr = static_cast<mindspore::Context *>(reinterpret_cast<void *>(context_ptr));
  if (c_context_ptr == nullptr) {
    MS_LOG(ERROR) << "Context pointer from java is nullptr";
    return;
  }
  c_context_ptr->SetInterOpParallelNum((int32_t)op_parallel_num);
}

/*
 * Class:     com_mindspore_config_MSContext
 * Method:    GetInterOpParallelNum
 * Signature: (J)I
 */
extern "C" JNIEXPORT jint JNICALL Java_com_mindspore_config_MSContext_getInterOpParallelNum(JNIEnv *env, jobject thiz,
                                                                                            jlong context_ptr) {
  auto *c_context_ptr = static_cast<mindspore::Context *>(reinterpret_cast<void *>(context_ptr));
  if (c_context_ptr == nullptr) {
    MS_LOG(ERROR) << "Context pointer from java is nullptr";
    return 0;
  }
  auto inter_op_parallel_num = c_context_ptr->GetInterOpParallelNum();
  return inter_op_parallel_num;
}

/*
 * Class:     com_mindspore_config_MSContext
 * Method:    SetThreadAffinity
 * Signature: (JI)V
 */
extern "C" JNIEXPORT void JNICALL Java_com_mindspore_config_MSContext_setThreadAffinity__JI(JNIEnv *env, jobject thiz,
                                                                                            jlong context_ptr,
                                                                                            jint thread_affinity) {
  auto *c_context_ptr = static_cast<mindspore::Context *>(reinterpret_cast<void *>(context_ptr));
  if (c_context_ptr == nullptr) {
    MS_LOG(ERROR) << "Context pointer from java is nullptr";
    return;
  }
  c_context_ptr->SetThreadAffinity(thread_affinity);
}

/*
 * Class:     com_mindspore_config_MSContext
 * Method:    GetThreadAffinityMode
 * Signature: (J)I
 */
extern "C" JNIEXPORT jint JNICALL Java_com_mindspore_config_MSContext_getThreadAffinityMode(JNIEnv *env, jobject thiz,
                                                                                            jlong context_ptr) {
  auto *c_context_ptr = static_cast<mindspore::Context *>(reinterpret_cast<void *>(context_ptr));
  if (c_context_ptr == nullptr) {
    MS_LOG(ERROR) << "Context pointer from java is nullptr";
    return 0;
  }
  auto thread_affinity_mode = c_context_ptr->GetThreadAffinityMode();
  return thread_affinity_mode;
}

/*
 * Class:     com_mindspore_config_MSContext
 * Method:    SetThreadAffinity
 * Signature: (J[I)V
 */
extern "C" JNIEXPORT void JNICALL Java_com_mindspore_config_MSContext_setThreadAffinity__J_3I(JNIEnv *env, jobject thiz,
                                                                                              jlong context_ptr,
                                                                                              jintArray core_list) {
  if (core_list == nullptr) {
    MS_LOG(ERROR) << "core_list from java is nullptr";
    return;
  }
  auto *c_context_ptr = static_cast<mindspore::Context *>(reinterpret_cast<void *>(context_ptr));
  if (c_context_ptr == nullptr) {
    MS_LOG(ERROR) << "Context pointer from java is nullptr";
    return;
  }
  int32_t array_len = env->GetArrayLength(core_list);
  jboolean is_copy = JNI_FALSE;
  int *core_value = env->GetIntArrayElements(core_list, &is_copy);
  std::vector<int> c_core_list(core_value, core_value + array_len);
  c_context_ptr->SetThreadAffinity(c_core_list);
  env->ReleaseIntArrayElements(core_list, core_value, 0);
  env->DeleteLocalRef(core_list);
}

/*
 * Class:     com_mindspore_config_MSContext
 * Method:    GetThreadAffinityCoreList
 * Signature: (J)Ljava/util/ArrayList;
 */
extern "C" JNIEXPORT jobject JNICALL Java_com_mindspore_config_MSContext_getThreadAffinityCoreList(JNIEnv *env,
                                                                                                   jobject thiz,
                                                                                                   jlong context_ptr) {
  auto *c_context_ptr = static_cast<mindspore::Context *>(reinterpret_cast<void *>(context_ptr));
  if (c_context_ptr == nullptr) {
    MS_LOG(ERROR) << "Context pointer from java is nullptr";
    return nullptr;
  }
  std::vector<int32_t> core_list_tmp = c_context_ptr->GetThreadAffinityCoreList();
  jobject core_list = newObjectArrayList<int32_t>(env, core_list_tmp, "java/lang/Integer", "(I)V");
  return core_list;
}

/*
 * Class:     com_mindspore_config_MSContext
 * Method:    SetEnableParallel
 * Signature: (JZ)V
 */
extern "C" JNIEXPORT void JNICALL Java_com_mindspore_config_MSContext_setEnableParallel(JNIEnv *env, jobject thiz,
                                                                                        jlong context_ptr,
                                                                                        jboolean is_parallel) {
  auto *c_context_ptr = static_cast<mindspore::Context *>(reinterpret_cast<void *>(context_ptr));
  if (c_context_ptr == nullptr) {
    MS_LOG(ERROR) << "Context pointer from java is nullptr";
    return;
  }
  c_context_ptr->SetEnableParallel(static_cast<bool>(is_parallel));
}

/*
 * Class:     com_mindspore_config_MSContext
 * Method:    GetEnableParallel
 * Signature: (J)Z
 */
extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_config_MSContext_getEnableParallel(JNIEnv *env, jobject thiz,
                                                                                            jlong context_ptr) {
  auto *c_context_ptr = static_cast<mindspore::Context *>(reinterpret_cast<void *>(context_ptr));
  if (c_context_ptr == nullptr) {
    MS_LOG(ERROR) << "Context pointer from java is nullptr";
    return (jboolean) false;
  }
  bool is_parallel = c_context_ptr->GetEnableParallel();
  return (jboolean)is_parallel;
}

extern "C" JNIEXPORT void JNICALL Java_com_mindspore_config_MSContext_free(JNIEnv *env, jobject thiz,
                                                                           jlong context_ptr) {
  auto *c_context_ptr = static_cast<mindspore::Context *>(reinterpret_cast<void *>(context_ptr));
  if (c_context_ptr == nullptr) {
    MS_LOG(ERROR) << "Context pointer from java is nullptr";
    return;
  }
  delete (c_context_ptr);
}
