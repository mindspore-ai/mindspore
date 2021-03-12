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
#include <cstring>
#include "common/ms_log.h"
#include "include/ms_tensor.h"
#include "ir/dtype/type_id.h"

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
  env->SetIntArrayRegion(shape, 0, shape_size, tmp);
  delete[](tmp);
  return shape;
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

extern "C" JNIEXPORT jbyteArray JNICALL Java_com_mindspore_lite_MSTensor_getByteData(JNIEnv *env, jobject thiz,
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

  auto local_element_num = ms_tensor_ptr->ElementsNum();
  auto ret = env->NewByteArray(local_element_num);
  env->SetByteArrayRegion(ret, 0, local_element_num, local_data);
  return ret;
}

extern "C" JNIEXPORT jlongArray JNICALL Java_com_mindspore_lite_MSTensor_getLongData(JNIEnv *env, jobject thiz,
                                                                                     jlong tensor_ptr) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Tensor pointer from java is nullptr");
    return env->NewLongArray(0);
  }

  auto *ms_tensor_ptr = static_cast<mindspore::tensor::MSTensor *>(pointer);

  auto *local_data = static_cast<jlong *>(ms_tensor_ptr->MutableData());
  if (local_data == nullptr) {
    MS_LOGD("Tensor has no data");
    return env->NewLongArray(0);
  }

  if (ms_tensor_ptr->data_type() != mindspore::kNumberTypeInt64) {
    MS_LOGE("data type is error : %d", ms_tensor_ptr->data_type());
    return env->NewLongArray(0);
  }
  auto local_element_num = ms_tensor_ptr->ElementsNum();
  auto ret = env->NewLongArray(local_element_num);
  env->SetLongArrayRegion(ret, 0, local_element_num, local_data);
  return ret;
}

extern "C" JNIEXPORT jintArray JNICALL Java_com_mindspore_lite_MSTensor_getIntData(JNIEnv *env, jobject thiz,
                                                                                   jlong tensor_ptr) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Tensor pointer from java is nullptr");
    return env->NewIntArray(0);
  }

  auto *ms_tensor_ptr = static_cast<mindspore::tensor::MSTensor *>(pointer);

  auto *local_data = static_cast<jint *>(ms_tensor_ptr->MutableData());
  if (local_data == nullptr) {
    MS_LOGD("Tensor has no data");
    return env->NewIntArray(0);
  }

  if (ms_tensor_ptr->data_type() != mindspore::kNumberTypeInt32) {
    MS_LOGE("data type is error : %d", ms_tensor_ptr->data_type());
    return env->NewIntArray(0);
  }
  auto local_element_num = ms_tensor_ptr->ElementsNum();
  auto ret = env->NewIntArray(local_element_num);
  env->SetIntArrayRegion(ret, 0, local_element_num, local_data);
  return ret;
}

extern "C" JNIEXPORT jfloatArray JNICALL Java_com_mindspore_lite_MSTensor_getFloatData(JNIEnv *env, jobject thiz,
                                                                                       jlong tensor_ptr) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Tensor pointer from java is nullptr");
    return env->NewFloatArray(0);
  }

  auto *ms_tensor_ptr = static_cast<mindspore::tensor::MSTensor *>(pointer);

  auto *local_data = static_cast<jfloat *>(ms_tensor_ptr->MutableData());
  if (local_data == nullptr) {
    MS_LOGD("Tensor has no data");
    return env->NewFloatArray(0);
  }

  if (ms_tensor_ptr->data_type() != mindspore::kNumberTypeFloat32) {
    MS_LOGE("data type is error : %d", ms_tensor_ptr->data_type());
    return env->NewFloatArray(0);
  }
  auto local_element_num = ms_tensor_ptr->ElementsNum();
  auto ret = env->NewFloatArray(local_element_num);
  env->SetFloatArrayRegion(ret, 0, local_element_num, local_data);
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
#ifdef ENABLE_ARM32
    MS_LOGE("data_len(%lld) not equal to Size of ms_tensor(%zu)", data_len, ms_tensor_ptr->Size());
#else
    MS_LOGE("data_len(%ld) not equal to Size of ms_tensor(%zu)", data_len, ms_tensor_ptr->Size());
#endif
    return static_cast<jboolean>(false);
  }
  jboolean is_copy = false;
  auto *data_arr = env->GetByteArrayElements(data, &is_copy);
  auto *local_data = ms_tensor_ptr->MutableData();
  memcpy(local_data, data_arr, data_len);
  env->ReleaseByteArrayElements(data, data_arr, JNI_ABORT);
  return static_cast<jboolean>(true);
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_lite_MSTensor_setByteBufferData(JNIEnv *env, jobject thiz,
                                                                                         jlong tensor_ptr,
                                                                                         jobject buffer) {
  auto *p_data = reinterpret_cast<jbyte *>(env->GetDirectBufferAddress(buffer));  // get buffer pointer
  jlong data_len = env->GetDirectBufferCapacity(buffer);                          // get buffer capacity
  if (p_data == nullptr) {
    MS_LOGE("GetDirectBufferAddress return null");
    return false;
  }

  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Tensor pointer from java is nullptr");
    return static_cast<jboolean>(false);
  }

  auto *ms_tensor_ptr = static_cast<mindspore::tensor::MSTensor *>(pointer);
  if (data_len != ms_tensor_ptr->Size()) {
#ifdef ENABLE_ARM32
    MS_LOGE("data_len(%lld) not equal to Size of ms_tensor(%zu)", data_len, ms_tensor_ptr->Size());
#else
    MS_LOGE("data_len(%ld) not equal to Size of ms_tensor(%zu)", data_len, ms_tensor_ptr->Size());
#endif
    return static_cast<jboolean>(false);
  }
  auto *local_data = ms_tensor_ptr->MutableData();
  memcpy(local_data, p_data, data_len);
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

extern "C" JNIEXPORT jstring JNICALL Java_com_mindspore_lite_MSTensor_tensorName(JNIEnv *env, jobject thiz,
                                                                                 jlong tensor_ptr) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Tensor pointer from java is nullptr");
    return nullptr;
  }
  auto *ms_tensor_ptr = static_cast<mindspore::tensor::MSTensor *>(pointer);

  return env->NewStringUTF(ms_tensor_ptr->tensor_name().c_str());
}
