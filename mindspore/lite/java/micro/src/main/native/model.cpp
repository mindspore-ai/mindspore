/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include <cstring>
#include <cstdlib>
#include "src/common/log.h"
#include "c_api/model_c.h"

void *ReadWeightFile(const char *c_weight_file, int *size) {
  if (!strstr(c_weight_file, ".bin") && !strstr(c_weight_file, ".net")) {
    MS_LOG(ERROR) << "Weight file name should end with .bin or .net";
    return nullptr;
  }
  FILE *file;
  file = fopen(c_weight_file, "rb");
  if (!file) {
    MS_LOG(ERROR) << "Can not find weight file";
    return nullptr;
  }
  int curr_file_posi = static_cast<int>(ftell(file));
  fseek(file, 0, SEEK_END);
  *size = static_cast<int>(ftell(file));
  auto *model_buffer = static_cast<unsigned char *>(malloc(*size));
  (void)memset(model_buffer, 0, *size);
  fseek(file, curr_file_posi, SEEK_SET);
  int read_size = static_cast<int>(fread(model_buffer, 1, *size, file));
  fclose(file);
  if (read_size != *size) {
    MS_LOG(ERROR) << "Read weight file failed, total file size: " << size << ", read_size: " << read_size;
    free(model_buffer);
    return nullptr;
  }
  return reinterpret_cast<void *>(model_buffer);
}

jobject GetInOrOutTensors(JNIEnv *env, jobject thiz, jlong model_ptr, bool is_input) {
  jclass array_list = env->FindClass("java/util/ArrayList");
  jmethodID array_list_construct = env->GetMethodID(array_list, "<init>", "()V");
  jobject ret = env->NewObject(array_list, array_list_construct);
  jmethodID array_list_add = env->GetMethodID(array_list, "add", "(Ljava/lang/Object;)Z");
  jclass long_object = env->FindClass("java/lang/Long");
  jmethodID long_object_construct = env->GetMethodID(long_object, "<init>", "(J)V");

  auto *pointer = reinterpret_cast<MSModelHandle *>(model_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Model pointer from java is nullptr";
    return ret;
  }
  MSTensorHandleArray tensor_handels;
  if (is_input) {
    tensor_handels = MSModelGetInputs(pointer);
  } else {
    tensor_handels = MSModelGetOutputs(pointer);
  }

  size_t data_num = tensor_handels.handle_num;
  for (size_t i = 0; i < data_num; i++) {
    MSTensorHandle tensor_handel = tensor_handels.handle_list[i];
    jobject tensor_addr = env->NewObject(long_object, long_object_construct, jlong(tensor_handel));
    env->CallBooleanMethod(ret, array_list_add, tensor_addr);
  }
  return ret;
}

void ConvertFP16ToFP32(MSTensorHandleArray tensors) {
  for (size_t i = 0; i < tensors.handle_num; i++) {
    if (MSTensorGetDataType(tensors.handle_list[i]) == kMSDataTypeNumberTypeFloat16) {
      MSTensorSetDataType(tensors.handle_list[i], kMSDataTypeNumberTypeFloat32);
    }
  }
}

extern "C" JNIEXPORT jlong JNICALL Java_com_mindspore_micro_Model_buildByWeight(JNIEnv *env, jobject thiz,
                                                                                jstring weight_file,
                                                                                jlong context_ptr) {
  auto c_context_ptr = reinterpret_cast<MSContextHandle>(context_ptr);
  if (c_context_ptr == nullptr) {
    MS_LOG(ERROR) << "Context pointer from java is nullptr";
    return jlong(nullptr);
  }
  auto c_weight_file = env->GetStringUTFChars(weight_file, JNI_FALSE);
  int model_size = 0;
  void *model_buffer = ReadWeightFile(c_weight_file, &model_size);
  if (model_buffer == nullptr) {
    MS_LOG(ERROR) << "Read weight file failed";
    return jlong(nullptr);
  }
  MSModelHandle model_handle = MSModelCreate();
  MSStatus status = MSModelBuild(model_handle, model_buffer, model_size, kMSModelTypeMindIR, c_context_ptr);
  if (model_buffer) {
    free(model_buffer);
  }
  if (status != kMSStatusSuccess) {
    MS_LOG(ERROR) << "[Micro] model build failed, status: " << status;
    return jlong(nullptr);
  }
  // if enable fp16, float inputs and outputs must be recognizable (fp32) by Java
  ConvertFP16ToFP32(MSModelGetInputs(model_handle));
  ConvertFP16ToFP32(MSModelGetOutputs(model_handle));
  return jlong(model_handle);
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_micro_Model_runStep(JNIEnv *env, jobject thiz,
                                                                             jlong model_ptr) {
  auto *pointer = reinterpret_cast<void *>(model_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Model pointer from java is nullptr";
    return (jboolean) false;
  }
  auto *model_handle = static_cast<MSModelHandle *>(pointer);
  MSTensorHandleArray inputs_handle = MSModelGetInputs(model_handle);
  for (size_t i = 0; i < inputs_handle.handle_num; i++) {
    if (MSTensorGetData(inputs_handle.handle_list[i]) == nullptr) {
      MS_LOG(ERROR) << "[Micro] model input[" << i << "] is not set";
      return (jboolean) false;
    }
  }
  MSTensorHandleArray outputs_handle = MSModelGetOutputs(model_handle);
  MSStatus status = MSModelPredict(model_handle, inputs_handle, &outputs_handle, nullptr, nullptr);
  if (status != kMSStatusSuccess) {
    MS_LOG(ERROR) << "[Micro] model predict failed, status: " << status;
    return (jboolean) false;
  }
  return (jboolean) true;
}

extern "C" JNIEXPORT jobject JNICALL Java_com_mindspore_micro_Model_getInputs(JNIEnv *env, jobject thiz,
                                                                              jlong model_ptr) {
  return GetInOrOutTensors(env, thiz, model_ptr, true);
}

extern "C" JNIEXPORT jobject JNICALL Java_com_mindspore_micro_Model_getOutputs(JNIEnv *env, jobject thiz,
                                                                               jlong model_ptr) {
  return GetInOrOutTensors(env, thiz, model_ptr, false);
}

extern "C" JNIEXPORT void JNICALL Java_com_mindspore_micro_Model_free(JNIEnv *env, jobject thiz, jlong model_ptr) {
  auto pointer = reinterpret_cast<void *>(model_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Model pointer from java is nullptr";
    return;
  }
  auto model_handle = static_cast<MSModelHandle>(pointer);
  MSModelDestroy(&model_handle);
}
