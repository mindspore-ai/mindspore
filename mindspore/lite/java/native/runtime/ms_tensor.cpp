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
#include "include/ms_tensor.h"
#include "ir/dtype/type_id.h"

extern "C" JNIEXPORT jlong JNICALL Java_com_mindspore_lite_MSTensor_createMSTensor(JNIEnv *env, jobject thiz,
                                                                                    jint data_type, jintArray shape,
                                                                                    jint shape_len) {
  jboolean is_copy = false;
  jint *local_shape_arr = env->GetIntArrayElements(shape, &is_copy);
  std::vector<int> local_shape(shape_len);
  for (size_t i = 0; i < shape_len; i++) {
    local_shape[i] = local_shape_arr[i];
  }
  auto *ms_tensor = mindspore::tensor::MSTensor::CreateTensor(mindspore::TypeId(data_type), local_shape);
  env->ReleaseIntArrayElements(shape, local_shape_arr, JNI_ABORT);
  if (ms_tensor == nullptr) {
    MS_LOGE("CreateTensor failed");
    return reinterpret_cast<jlong>(nullptr);
  }
  return reinterpret_cast<jlong>(ms_tensor);
}

extern "C" JNIEXPORT jintArray JNICALL Java_com_mindspore_lite_MSTensor_getShape(JNIEnv *env, jobject thiz,
                                                                                  jlong tensor_ptr) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Tensor pointer from java is nullptr");
    return env->NewIntArray(0);
  }
  auto *ms_tensor_ptr = static_cast<mindspore::tensor::MSTensor *>(pointer);
  auto local_shape = ms_tensor_ptr->shape();
  auto shape_size = local_shape.size();
  jintArray shape = env->NewIntArray(shape_size);
  auto *tmp = new jint[shape_size];
  for (size_t i = 0; i < shape_size; i++) {
    tmp[i] = local_shape.at(i);
  }
  delete[](tmp);
  env->SetIntArrayRegion(shape, 0, shape_size, tmp);
  return shape;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_lite_MSTensor_setShape(JNIEnv *env, jobject thiz,
                                                                                 jlong tensor_ptr, jintArray shape,
                                                                                 jint shape_len) {
  jboolean is_copy = false;
  jint *local_shape_arr = env->GetIntArrayElements(shape, &is_copy);
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Tensor pointer from java is nullptr");
    return static_cast<jboolean>(false);
  }
  auto *ms_tensor_ptr = static_cast<mindspore::tensor::MSTensor *>(pointer);
  std::vector<int> local_shape(shape_len);
  for (size_t i = 0; i < shape_len; i++) {
    local_shape[i] = local_shape_arr[i];
  }
  auto ret = ms_tensor_ptr->set_shape(local_shape);
  return ret == shape_len;
}

extern "C" JNIEXPORT jint JNICALL Java_com_mindspore_lite_MSTensor_getDataType(JNIEnv *env, jobject thiz,
                                                                                jlong tensor_ptr) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Tensor pointer from java is nullptr");
    return static_cast<jboolean>(false);
  }
  auto *ms_tensor_ptr = static_cast<mindspore::tensor::MSTensor *>(pointer);
  return jint(ms_tensor_ptr->data_type());
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_lite_MSTensor_setDataType(JNIEnv *env, jobject thiz,
                                                                                    jlong tensor_ptr, jint data_type) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Tensor pointer from java is nullptr");
    return static_cast<jboolean>(false);
  }
  auto *ms_tensor_ptr = static_cast<mindspore::tensor::MSTensor *>(pointer);
  auto ret = ms_tensor_ptr->set_data_type(mindspore::TypeId(data_type));
  return ret == data_type;
}

extern "C" JNIEXPORT jbyteArray JNICALL Java_com_mindspore_lite_MSTensor_getData(JNIEnv *env, jobject thiz,
                                                                                  jlong tensor_ptr) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Tensor pointer from java is nullptr");
    return env->NewByteArray(0);
  }
  auto *ms_tensor_ptr = static_cast<mindspore::tensor::MSTensor *>(pointer);
  auto *local_data = static_cast<jbyte *>(ms_tensor_ptr->MutableData());
  if (local_data == nullptr) {
    MS_LOGD("Tensor has no data");
    return env->NewByteArray(0);
  }
  auto local_data_size = ms_tensor_ptr->Size();
  auto ret = env->NewByteArray(local_data_size);
  env->SetByteArrayRegion(ret, 0, local_data_size, local_data);
  return ret;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_lite_MSTensor_setData(JNIEnv *env, jobject thiz,
                                                                                jlong tensor_ptr, jbyteArray data,
                                                                                jlong data_len) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Tensor pointer from java is nullptr");
    return static_cast<jboolean>(false);
  }
  auto *ms_tensor_ptr = static_cast<mindspore::tensor::MSTensor *>(pointer);
  if (data_len != ms_tensor_ptr->Size()) {
    MS_LOGE("data_len(%ld) not equal to Size of ms_tensor(%zu)", data_len, ms_tensor_ptr->Size());
    return static_cast<jboolean>(false);
  }
  jboolean is_copy = false;
  auto *data_arr = env->GetByteArrayElements(data, &is_copy);
  auto *local_data = ms_tensor_ptr->MutableData();
  memcpy(local_data, data_arr, data_len);
  return static_cast<jboolean>(true);
}

extern "C" JNIEXPORT jlong JNICALL Java_com_mindspore_lite_MSTensor_size(JNIEnv *env, jobject thiz, jlong tensor_ptr) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Tensor pointer from java is nullptr");
    return 0;
  }
  auto *ms_tensor_ptr = static_cast<mindspore::tensor::MSTensor *>(pointer);
  return ms_tensor_ptr->Size();
}

extern "C" JNIEXPORT jint JNICALL Java_com_mindspore_lite_MSTensor_elementsNum(JNIEnv *env, jobject thiz,
                                                                                jlong tensor_ptr) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Tensor pointer from java is nullptr");
    return 0;
  }
  auto *ms_tensor_ptr = static_cast<mindspore::tensor::MSTensor *>(pointer);
  return ms_tensor_ptr->ElementsNum();
}

extern "C" JNIEXPORT void JNICALL Java_com_mindspore_lite_MSTensor_free(JNIEnv *env, jobject thiz, jlong tensor_ptr) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Tensor pointer from java is nullptr");
    return;
  }
  auto *ms_tensor_ptr = static_cast<mindspore::tensor::MSTensor *>(pointer);
  delete (ms_tensor_ptr);
}
