/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

constexpr const int kCpuDevice = 0;
constexpr const int kGpuDevice = 1;
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
  if (static_cast<bool>(enable_parallel)) {
    context->SetEnableParallel(enable_parallel);
  }
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

/*
 * Parse provider from AscendDeviceInfo.
 */
std::string ParseAscendDevInfoProvider(JNIEnv *env, jclass clazz_ascend_device_info, jobject ascend_device_info_obj) {
  jmethodID methodID_getProvider = env->GetMethodID(clazz_ascend_device_info, "getProvider", "()Ljava/lang/String;");
  if (methodID_getProvider == nullptr) {
    MS_LOG(ERROR) << "AscendDeviceInfo method getProvider from java is nullptr";
    return "";
  }
  jstring rv = (jstring)(env->CallObjectMethod(ascend_device_info_obj, methodID_getProvider, 0));
  if (rv == nullptr) {
    return "";
  }
  const char *c_provider = env->GetStringUTFChars(rv, nullptr);
  std::string str_provider(c_provider, env->GetStringLength(rv));
  env->ReleaseStringUTFChars(rv, c_provider);
  return str_provider;
}

/*
 * Parse device id from AscendDeviceInfo.
 */
int ParseAscendDevInfoDeviceID(JNIEnv *env, jclass clazz_ascend_device_info, jobject ascend_device_info_obj) {
  jmethodID methodID_getDeviceID = env->GetMethodID(clazz_ascend_device_info, "getDeviceID", "()I");
  if (methodID_getDeviceID == nullptr) {
    MS_LOG(ERROR) << "AscendDeviceInfo method getDeviceID from java is nullptr";
    return 0;
  }
  jint rv = (jint)(env->CallIntMethod(ascend_device_info_obj, methodID_getDeviceID, 0));
  return static_cast<int>(rv);
}

/*
 * Parse rank id from AscendDeviceInfo.
 */
int ParseAscendDevInfoRankID(JNIEnv *env, jclass clazz_ascend_device_info, jobject ascend_device_info_obj) {
  jmethodID methodID_getRankID = env->GetMethodID(clazz_ascend_device_info, "getRankID", "()I");
  if (methodID_getRankID == nullptr) {
    MS_LOG(ERROR) << "AscendDeviceInfo method getRankID from java is nullptr";
    return 0;
  }
  jint rv = (jint)(env->CallIntMethod(ascend_device_info_obj, methodID_getRankID, 0));
  return static_cast<int>(rv);
}

/*
 * Parse insert op config path from AscendDeviceInfo.
 */
std::string ParseAscendDevInfoInsrtOpConfigPath(JNIEnv *env, jclass clazz_ascend_device_info,
                                                jobject ascend_device_info_obj) {
  jmethodID methodID_getInsertOpConfigPath =
    env->GetMethodID(clazz_ascend_device_info, "getInsertOpConfigPath", "()Ljava/lang/String;");
  if (methodID_getInsertOpConfigPath == nullptr) {
    MS_LOG(ERROR) << "AscendDeviceInfo method getInsertOpConfigPath from java is nullptr";
    return "";
  }
  jstring rv = (jstring)(env->CallObjectMethod(ascend_device_info_obj, methodID_getInsertOpConfigPath, 0));
  if (rv == nullptr) {
    return "";
  }
  const char *c_path = env->GetStringUTFChars(rv, nullptr);
  std::string str_config(c_path, env->GetStringLength(rv));
  env->ReleaseStringUTFChars(rv, c_path);
  return str_config;
}

/*
 * Parse input format from AscendDeviceInfo.
 */
std::string ParseAscendDevInfoInputFormat(JNIEnv *env, jclass clazz_ascend_device_info,
                                          jobject ascend_device_info_obj) {
  jmethodID methodID_getInputFormat =
    env->GetMethodID(clazz_ascend_device_info, "getInputFormat", "()Ljava/lang/String;");
  if (methodID_getInputFormat == nullptr) {
    MS_LOG(ERROR) << "AscendDeviceInfo method getInputFormat from java is nullptr";
    return "";
  }
  jstring rv = (jstring)(env->CallObjectMethod(ascend_device_info_obj, methodID_getInputFormat, 0));
  if (rv == nullptr) {
    return "";
  }
  const char *c_format = env->GetStringUTFChars(rv, nullptr);
  std::string str_format(c_format, env->GetStringLength(rv));
  env->ReleaseStringUTFChars(rv, c_format);
  return str_format;
}

/*
 * Parse input shape from AscendDeviceInfo.
 */
std::string ParseAscendDevInfoInputShape(JNIEnv *env, jclass clazz_ascend_device_info, jobject ascend_device_info_obj) {
  jmethodID methodID_getInputShape =
    env->GetMethodID(clazz_ascend_device_info, "getInputShape", "()Ljava/lang/String;");
  if (methodID_getInputShape == nullptr) {
    MS_LOG(ERROR) << "AscendDeviceInfo method getInputShape from java is nullptr";
    return "";
  }
  jstring rv = (jstring)(env->CallObjectMethod(ascend_device_info_obj, methodID_getInputShape, 0));
  if (rv == nullptr) {
    return "";
  }
  const char *c_shape = env->GetStringUTFChars(rv, nullptr);
  std::string str_shape(c_shape, env->GetStringLength(rv));
  env->ReleaseStringUTFChars(rv, c_shape);
  return str_shape;
}

/*
 * Parse dynamic image size from AscendDeviceInfo.
 */
std::string ParseAscendDevInfoDynImageSize(JNIEnv *env, jclass clazz_ascend_device_info,
                                           jobject ascend_device_info_obj) {
  jmethodID methodID_getDynamicImageSize =
    env->GetMethodID(clazz_ascend_device_info, "getDynamicImageSize", "()Ljava/lang/String;");
  if (methodID_getDynamicImageSize == nullptr) {
    MS_LOG(ERROR) << "AscendDeviceInfo method getDynamicImageSize from java is nullptr";
    return "";
  }
  jstring rv = (jstring)(env->CallObjectMethod(ascend_device_info_obj, methodID_getDynamicImageSize, 0));
  if (rv == nullptr) {
    return "";
  }
  const char *c_img_size = env->GetStringUTFChars(rv, nullptr);
  std::string str_img_size(c_img_size, env->GetStringLength(rv));
  env->ReleaseStringUTFChars(rv, c_img_size);
  return str_img_size;
}

/*
 * Parse output type from AscendDeviceInfo.
 */
int ParseAscendDevInfoOutputType(JNIEnv *env, jclass clazz_ascend_device_info, jobject ascend_device_info_obj) {
  jmethodID methodID_getOutputType = env->GetMethodID(clazz_ascend_device_info, "getOutputType", "()I");
  if (methodID_getOutputType == nullptr) {
    MS_LOG(ERROR) << "AscendDeviceInfo method getOutputType from java is nullptr";
    return 0;
  }
  jint rv = (jint)(env->CallIntMethod(ascend_device_info_obj, methodID_getOutputType, 0));
  return static_cast<int>(rv);
}

/*
 * Parse precision mode from AscendDeviceInfo.
 */
std::string ParseAscendDevInfoPrecisionMode(JNIEnv *env, jclass clazz_ascend_device_info,
                                            jobject ascend_device_info_obj) {
  jmethodID methodID_getPrecisionMode =
    env->GetMethodID(clazz_ascend_device_info, "getPrecisionMode", "()Ljava/lang/String;");
  if (methodID_getPrecisionMode == nullptr) {
    MS_LOG(ERROR) << "AscendDeviceInfo method getPrecisionMode from java is nullptr";
    return "";
  }
  jstring rv = (jstring)(env->CallObjectMethod(ascend_device_info_obj, methodID_getPrecisionMode, 0));
  if (rv == nullptr) {
    return "";
  }
  const char *c_precision_mode = env->GetStringUTFChars(rv, nullptr);
  std::string str_precision_mode(c_precision_mode, env->GetStringLength(rv));
  env->ReleaseStringUTFChars(rv, c_precision_mode);
  return str_precision_mode;
}

/*
 * Parse op select impl mode from AscendDeviceInfo.
 */
std::string ParseAscendDevInfoOpSelectImplMode(JNIEnv *env, jclass clazz_ascend_device_info,
                                               jobject ascend_device_info_obj) {
  jmethodID methodID_getOpSelectImplMode =
    env->GetMethodID(clazz_ascend_device_info, "getOpSelectImplMode", "()Ljava/lang/String;");
  if (methodID_getOpSelectImplMode == nullptr) {
    MS_LOG(ERROR) << "AscendDeviceInfo method getOpSelectImplMode from java is nullptr";
    return "";
  }
  jstring rv = (jstring)(env->CallObjectMethod(ascend_device_info_obj, methodID_getOpSelectImplMode, 0));
  if (rv == nullptr) {
    return "";
  }
  const char *c_op_select_mode = env->GetStringUTFChars(rv, nullptr);
  std::string str_op_select_mode(c_op_select_mode, env->GetStringLength(rv));
  env->ReleaseStringUTFChars(rv, c_op_select_mode);
  return str_op_select_mode;
}

/*
 * Parse fusion switch config path from AscendDeviceInfo.
 */
std::string ParseAscendDevInfoFusionSwitchConfigPath(JNIEnv *env, jclass clazz_ascend_device_info,
                                                     jobject ascend_device_info_obj) {
  jmethodID methodID_getFusionSwitchConfigPath =
    env->GetMethodID(clazz_ascend_device_info, "getFusionSwitchConfigPath", "()Ljava/lang/String;");
  if (methodID_getFusionSwitchConfigPath == nullptr) {
    MS_LOG(ERROR) << "AscendDeviceInfo method getFusionSwitchConfigPath from java is nullptr";
    return "";
  }
  jstring rv = (jstring)(env->CallObjectMethod(ascend_device_info_obj, methodID_getFusionSwitchConfigPath, 0));
  if (rv == nullptr) {
    return "";
  }
  const char *c_fusion_switch_path = env->GetStringUTFChars(rv, nullptr);
  std::string str_fusion_switch_path(c_fusion_switch_path, env->GetStringLength(rv));
  env->ReleaseStringUTFChars(rv, c_fusion_switch_path);
  return str_fusion_switch_path;
}

/*
 * Parse buffer optimize mode from AscendDeviceInfo.
 */
std::string ParseAscendDevInfoBufferOptimizeMode(JNIEnv *env, jclass clazz_ascend_device_info,
                                                 jobject ascend_device_info_obj) {
  jmethodID methodID_getBufferOptimizeMode =
    env->GetMethodID(clazz_ascend_device_info, "getBufferOptimizeMode", "()Ljava/lang/String;");
  if (methodID_getBufferOptimizeMode == nullptr) {
    MS_LOG(ERROR) << "AscendDeviceInfo method getBufferOptimizeMode from java is nullptr";
    return "";
  }
  jstring rv = (jstring)(env->CallObjectMethod(ascend_device_info_obj, methodID_getBufferOptimizeMode, 0));
  if (rv == nullptr) {
    return "";
  }
  const char *c_buffer_mode = env->GetStringUTFChars(rv, nullptr);
  std::string str_buffer_mode(c_buffer_mode, env->GetStringLength(rv));
  env->ReleaseStringUTFChars(rv, c_buffer_mode);
  return str_buffer_mode;
}

/*
 * Parse dynamic batch sizes from AscendDeviceInfo.
 */
std::vector<size_t> ParseAscendDevInfoDynBatchSize(JNIEnv *env, jclass clazz_ascend_device_info,
                                                   jobject ascend_device_info_obj) {
  jmethodID methodID_getDynamicBatchSize =
    env->GetMethodID(clazz_ascend_device_info, "getDynamicBatchSize", "()Ljava/util/ArrayList;");
  if (methodID_getDynamicBatchSize == nullptr) {
    MS_LOG(ERROR) << "AscendDeviceInfo method getDynamicBatchSize from java is nullptr";
    return {};
  }
  jobject object_dyn_batch_list = env->CallObjectMethod(ascend_device_info_obj, methodID_getDynamicBatchSize);
  if (object_dyn_batch_list == nullptr) {
    MS_LOG(ERROR) << "Dynamic batch size list is nullptr.";
    return {};
  }
  std::vector<size_t> batch_vals = {};
  jclass clazz_array_list = env->FindClass("java/util/ArrayList");
  jclass clazz_integer = env->FindClass("java/lang/Integer");

  jmethodID methodID_intValue = env->GetMethodID(clazz_integer, "intValue", "()I");
  jmethodID methodID_get = env->GetMethodID(clazz_array_list, "get", "(I)Ljava/lang/Object;");
  jmethodID methodID_size = env->GetMethodID(clazz_array_list, "size", "()I");

  int batch_vec_size = static_cast<int>(env->CallIntMethod(object_dyn_batch_list, methodID_size));
  if (batch_vec_size < 1) {
    return {};
  }

  for (int i = 0; i < batch_vec_size; i++) {
    jobject elem = env->CallObjectMethod(object_dyn_batch_list, methodID_get, i);
    batch_vals.push_back(static_cast<size_t>(env->CallIntMethod(elem, methodID_intValue)));
  }

  return batch_vals;
}

/*
 * Parse input shape map from AscendDeviceInfo.
 */
std::map<int, std::vector<int>> ParseAscendDevInfoInputShapeMap(JNIEnv *env, jclass clazz_ascend_device_info,
                                                                jobject ascend_device_info_obj) {
  jmethodID methodID_getInputShapeMap =
    env->GetMethodID(clazz_ascend_device_info, "getInputShapeMap", "()Ljava/util/HashMap;");
  if (methodID_getInputShapeMap == nullptr) {
    MS_LOG(ERROR) << "AscendDeviceInfo method getInputShapeMap from java is nullptr";
    return {};
  }
  jobject object_in_shape_map = env->CallObjectMethod(ascend_device_info_obj, methodID_getInputShapeMap);
  if (object_in_shape_map == nullptr) {
    MS_LOG(ERROR) << "Input shape map is nullptr.";
    return {};
  }
  std::map<int, std::vector<int>> res;

  jclass clazz_array_list = env->FindClass("java/util/ArrayList");
  jclass clazz_hash_map = env->FindClass("java/util/HashMap");
  jclass clazz_set = env->FindClass("java/util/Set");
  jclass clazz_integer = env->FindClass("java/lang/Integer");
  jclass clazz_iterator = env->FindClass("java/util/Iterator");
  jclass clazz_map_entry = env->FindClass("java/util/Map$Entry");

  jmethodID methodID_intValue = env->GetMethodID(clazz_integer, "intValue", "()I");
  jmethodID methodID_get = env->GetMethodID(clazz_array_list, "get", "(I)Ljava/lang/Object;");
  jmethodID methodID_size = env->GetMethodID(clazz_array_list, "size", "()I");
  jmethodID methodID_entry_set = env->GetMethodID(clazz_hash_map, "entrySet", "()Ljava/util/Set;");
  jmethodID methodID_iterator = env->GetMethodID(clazz_set, "iterator", "()Ljava/util/Iterator;");
  jmethodID methodID_next = env->GetMethodID(clazz_iterator, "next", "()Ljava/lang/Object;");
  jmethodID methodID_has_next = env->GetMethodID(clazz_iterator, "hasNext", "()Z");
  jmethodID methodID_get_key = env->GetMethodID(clazz_map_entry, "getKey", "()Ljava/lang/Object;");
  jmethodID methodID_get_value = env->GetMethodID(clazz_map_entry, "getValue", "()Ljava/lang/Object;");

  jobject object_set = env->CallObjectMethod(object_in_shape_map, methodID_entry_set);
  jobject object_iterator = env->CallObjectMethod(object_set, methodID_iterator);

  while (env->CallBooleanMethod(object_iterator, methodID_has_next)) {
    jobject object_entry = env->CallObjectMethod(object_iterator, methodID_next);
    jobject object_key = env->CallObjectMethod(object_entry, methodID_get_key);
    int input_index = static_cast<int>(env->CallIntMethod(object_key, methodID_intValue));

    jobject object_shapes = env->CallObjectMethod(object_entry, methodID_get_value);
    if (object_shapes == nullptr) {
      continue;
    }
    int shape_vec_size = static_cast<int>(env->CallIntMethod(object_shapes, methodID_size));
    if (shape_vec_size < 1) {
      continue;
    }

    std::vector<int> vec;
    for (int i = 0; i < shape_vec_size; i++) {
      jobject elem = env->CallObjectMethod(object_shapes, methodID_get, i);
      vec.push_back(static_cast<int>(env->CallIntMethod(elem, methodID_intValue)));
      env->DeleteLocalRef(elem);
    }
    res[input_index] = vec;
    env->DeleteLocalRef(object_key);
    env->DeleteLocalRef(object_shapes);
    env->DeleteLocalRef(object_entry);
  }

  return res;
}

/*
 * Class:     com_mindspore_config_MSContext
 * Method:    addDeviceInfo
 * Signature: (JLcom/mindspore/config/AscendDeviceInfo;)Z
 */
extern "C" JNIEXPORT jboolean JNICALL
Java_com_mindspore_config_MSContext_addDeviceInfo__JLcom_mindspore_config_AscendDeviceInfo_2(
  JNIEnv *env, jobject thiz, jlong context_ptr, jobject ascend_device_info_obj) {
  jclass clazz_ascend_device_info = env->GetObjectClass(ascend_device_info_obj);
  if (clazz_ascend_device_info == nullptr) {
    MS_LOG(ERROR) << "AscendDeviceInfo class from java is nullptr";
    return (jboolean) false;
  }

  auto *pointer = reinterpret_cast<void *>(context_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Context pointer from java is nullptr";
    return (jboolean) false;
  }
  auto *c_context_ptr = static_cast<mindspore::Context *>(pointer);
  auto &device_list = c_context_ptr->MutableDeviceInfo();

  auto ascend_device_info = std::make_shared<mindspore::AscendDeviceInfo>();
  if (ascend_device_info == nullptr) {
    MS_LOG(ERROR) << "ascend device info is nullptr";
    delete (c_context_ptr);
    return (jboolean) false;
  }

  // Parse provider option from AscendDeviceInfo
  std::string str_provider = ParseAscendDevInfoProvider(env, clazz_ascend_device_info, ascend_device_info_obj);
  if (!str_provider.empty()) {
    ascend_device_info->SetProvider(str_provider);
  }

  // Parse device id option from AscendDeviceInfo
  int device_id = ParseAscendDevInfoDeviceID(env, clazz_ascend_device_info, ascend_device_info_obj);
  ascend_device_info->SetDeviceID(device_id);

  // Parse rank id option from AscendDeviceInfo
  int rank_id = ParseAscendDevInfoRankID(env, clazz_ascend_device_info, ascend_device_info_obj);
  ascend_device_info->SetRankID(rank_id);

  // Parse insert op config path from AscendDeviceInfo
  std::string insert_op_config_path =
    ParseAscendDevInfoInsrtOpConfigPath(env, clazz_ascend_device_info, ascend_device_info_obj);
  ascend_device_info->SetInsertOpConfigPath(insert_op_config_path);

  // Parse input format from AscendDeviceInfo
  std::string input_format = ParseAscendDevInfoInputFormat(env, clazz_ascend_device_info, ascend_device_info_obj);
  ascend_device_info->SetInputFormat(input_format);

  // Parse input shape from AscendDeviceInfo
  std::string input_shape = ParseAscendDevInfoInputShape(env, clazz_ascend_device_info, ascend_device_info_obj);
  ascend_device_info->SetInputShape(input_shape);

  // Parse dynamic image size from AscendDeviceInfo
  std::string dyn_image_size = ParseAscendDevInfoDynImageSize(env, clazz_ascend_device_info, ascend_device_info_obj);
  ascend_device_info->SetDynamicImageSize(dyn_image_size);

  // Parse output type from AscendDeviceInfo
  int output_type = ParseAscendDevInfoOutputType(env, clazz_ascend_device_info, ascend_device_info_obj);
  ascend_device_info->SetOutputType(static_cast<mindspore::DataType>(output_type));

  // Parse precision mode from AscendDeviceInfo
  std::string precision_mode = ParseAscendDevInfoPrecisionMode(env, clazz_ascend_device_info, ascend_device_info_obj);
  ascend_device_info->SetPrecisionMode(precision_mode);

  // Parse op select impl mode from AscendDeviceInfo
  std::string op_select_mode =
    ParseAscendDevInfoOpSelectImplMode(env, clazz_ascend_device_info, ascend_device_info_obj);
  ascend_device_info->SetOpSelectImplMode(op_select_mode);

  // Parse fusion switch config path from AscendDeviceInfo
  std::string fusion_switch_path =
    ParseAscendDevInfoFusionSwitchConfigPath(env, clazz_ascend_device_info, ascend_device_info_obj);
  ascend_device_info->SetFusionSwitchConfigPath(fusion_switch_path);

  // Parse fusion switch config path from AscendDeviceInfo
  std::string buffer_mode = ParseAscendDevInfoBufferOptimizeMode(env, clazz_ascend_device_info, ascend_device_info_obj);
  ascend_device_info->SetBufferOptimizeMode(buffer_mode);

  // Parse dynamic batch sizes from AscendDeviceInfo
  std::vector<size_t> dyn_batch_sizes =
    ParseAscendDevInfoDynBatchSize(env, clazz_ascend_device_info, ascend_device_info_obj);
  ascend_device_info->SetDynamicBatchSize(dyn_batch_sizes);

  // Parse input shape map from AscendDeviceInfo
  std::map<int, std::vector<int>> input_shape_map =
    ParseAscendDevInfoInputShapeMap(env, clazz_ascend_device_info, ascend_device_info_obj);
  ascend_device_info->SetInputShapeMap(input_shape_map);

  device_list.push_back(ascend_device_info);

  return (jboolean) true;
}

/*
 * Class:     com_mindspore_config_MSContext
 * Method:    addDeviceInfo
 * Signature: (JIZI)Z
 */
extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_config_MSContext_addDeviceInfo__JIZI(
  JNIEnv *env, jobject thiz, jlong context_ptr, jint device_type, jboolean enable_fp16, jint npu_freq) {
  auto *pointer = reinterpret_cast<void *>(context_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Context pointer from java is nullptr";
    return (jboolean) false;
  }
  auto *c_context_ptr = static_cast<mindspore::Context *>(pointer);
  auto &device_list = c_context_ptr->MutableDeviceInfo();
  switch (device_type) {
    case kCpuDevice: {
      auto cpu_device_info = std::make_shared<mindspore::CPUDeviceInfo>();
      if (cpu_device_info == nullptr) {
        MS_LOG(ERROR) << "cpu device info is nullptr";
        return (jboolean) false;
      }
      cpu_device_info->SetEnableFP16(enable_fp16);
      device_list.push_back(cpu_device_info);
      break;
    }
    case kGpuDevice:  // DT_GPU
    {
      auto gpu_device_info = std::make_shared<mindspore::GPUDeviceInfo>();
      if (gpu_device_info == nullptr) {
        MS_LOG(ERROR) << "gpu device info is nullptr";
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
        return (jboolean) false;
      }
      ascend_device_info->SetDeviceID(0);
      device_list.push_back(ascend_device_info);
      break;
    }
    default:
      MS_LOG(ERROR) << "Invalid device_type : " << device_type;
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
