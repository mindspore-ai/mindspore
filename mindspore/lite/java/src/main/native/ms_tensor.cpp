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
#include <cstring>
#include "common/ms_log.h"
#include "include/api/types.h"

extern "C" JNIEXPORT jintArray JNICALL Java_com_mindspore_MSTensor_getShape(JNIEnv *env, jobject thiz,
                                                                            jlong tensor_ptr) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Tensor pointer from java is nullptr");
    return env->NewIntArray(0);
  }
  auto *ms_tensor_ptr = static_cast<mindspore::MSTensor *>(pointer);
  auto local_shape = ms_tensor_ptr->Shape();
  auto shape_size = local_shape.size();
  jintArray shape = env->NewIntArray(shape_size);
  auto *tmp = new jint[shape_size];
  for (size_t i = 0; i < shape_size; i++) {
    tmp[i] = static_cast<int>(local_shape.at(i));
  }
  env->SetIntArrayRegion(shape, 0, shape_size, tmp);
  delete[](tmp);
  return shape;
}

extern "C" JNIEXPORT jint JNICALL Java_com_mindspore_MSTensor_getDataType(JNIEnv *env, jobject thiz, jlong tensor_ptr) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Tensor pointer from java is nullptr");
    return static_cast<jboolean>(false);
  }
  auto *ms_tensor_ptr = static_cast<mindspore::MSTensor *>(pointer);
  return jint(ms_tensor_ptr->DataType());
}

extern "C" JNIEXPORT jbyteArray JNICALL Java_com_mindspore_MSTensor_getByteData(JNIEnv *env, jobject thiz,
                                                                                jlong tensor_ptr) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Tensor pointer from java is nullptr");
    return env->NewByteArray(0);
  }
  auto *ms_tensor_ptr = static_cast<mindspore::MSTensor *>(pointer);
  auto *local_data = static_cast<jbyte *>(ms_tensor_ptr->MutableData());
  if (local_data == nullptr) {
    MS_LOGD("Tensor has no data");
    return env->NewByteArray(0);
  }

  auto local_size = ms_tensor_ptr->DataSize();
  if (local_size <= 0) {
    MS_LOGE("Size of tensor is negative: %zu", local_size);
    return env->NewByteArray(0);
  }
  auto ret = env->NewByteArray(local_size);
  env->SetByteArrayRegion(ret, 0, local_size, local_data);
  return ret;
}

extern "C" JNIEXPORT jlongArray JNICALL Java_com_mindspore_MSTensor_getLongData(JNIEnv *env, jobject thiz,
                                                                                jlong tensor_ptr) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Tensor pointer from java is nullptr");
    return env->NewLongArray(0);
  }

  auto *ms_tensor_ptr = static_cast<mindspore::MSTensor *>(pointer);

  auto *local_data = static_cast<jlong *>(ms_tensor_ptr->MutableData());
  if (local_data == nullptr) {
    MS_LOGD("Tensor has no data");
    return env->NewLongArray(0);
  }

  if (ms_tensor_ptr->DataType() != mindspore::DataType::kNumberTypeInt64) {
    MS_LOGE("data type is error : %d", static_cast<int>(ms_tensor_ptr->DataType()));
    return env->NewLongArray(0);
  }
  auto local_element_num = ms_tensor_ptr->ElementNum();
  if (local_element_num <= 0) {
    MS_LOGE("ElementsNum of tensor is negative: %d", static_cast<int>(local_element_num));
    return env->NewLongArray(0);
  }
  auto ret = env->NewLongArray(local_element_num);
  env->SetLongArrayRegion(ret, 0, local_element_num, local_data);
  return ret;
}

extern "C" JNIEXPORT jintArray JNICALL Java_com_mindspore_MSTensor_getIntData(JNIEnv *env, jobject thiz,
                                                                              jlong tensor_ptr) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Tensor pointer from java is nullptr");
    return env->NewIntArray(0);
  }

  auto *ms_tensor_ptr = static_cast<mindspore::MSTensor *>(pointer);

  auto *local_data = static_cast<jint *>(ms_tensor_ptr->MutableData());
  if (local_data == nullptr) {
    MS_LOGD("Tensor has no data");
    return env->NewIntArray(0);
  }

  if (ms_tensor_ptr->DataType() != mindspore::DataType::kNumberTypeInt32) {
    MS_LOGE("data type is error : %d", static_cast<int>(ms_tensor_ptr->DataType()));
    return env->NewIntArray(0);
  }
  auto local_element_num = ms_tensor_ptr->ElementNum();
  if (local_element_num <= 0) {
    MS_LOGE("ElementsNum of tensor is negative: %d", static_cast<int>(local_element_num));
    return env->NewIntArray(0);
  }
  auto ret = env->NewIntArray(local_element_num);
  env->SetIntArrayRegion(ret, 0, local_element_num, local_data);
  return ret;
}

extern "C" JNIEXPORT jfloatArray JNICALL Java_com_mindspore_MSTensor_getFloatData(JNIEnv *env, jobject thiz,
                                                                                  jlong tensor_ptr) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Tensor pointer from java is nullptr");
    return env->NewFloatArray(0);
  }

  auto *ms_tensor_ptr = static_cast<mindspore::MSTensor *>(pointer);

  auto *local_data = static_cast<jfloat *>(ms_tensor_ptr->MutableData());
  if (local_data == nullptr) {
    MS_LOGD("Tensor has no data");
    return env->NewFloatArray(0);
  }

  if (ms_tensor_ptr->DataType() != mindspore::DataType::kNumberTypeFloat32) {
    MS_LOGE("data type is error : %d", static_cast<int>(ms_tensor_ptr->DataType()));
    return env->NewFloatArray(0);
  }
  auto local_element_num = ms_tensor_ptr->ElementNum();
  if (local_element_num <= 0) {
    MS_LOGE("ElementsNum of tensor is negative: %d", static_cast<int>(local_element_num));
    return env->NewFloatArray(0);
  }
  auto ret = env->NewFloatArray(local_element_num);
  env->SetFloatArrayRegion(ret, 0, local_element_num, local_data);
  return ret;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_MSTensor_setData(JNIEnv *env, jobject thiz, jlong tensor_ptr,
                                                                          jbyteArray data, jlong data_len) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Tensor pointer from java is nullptr");
    return static_cast<jboolean>(false);
  }
  auto *ms_tensor_ptr = static_cast<mindspore::MSTensor *>(pointer);
  if (static_cast<size_t>(data_len) != ms_tensor_ptr->DataSize()) {
#ifdef ENABLE_ARM32
    MS_LOGE("data_len(%lld) not equal to Size of ms_tensor(%zu)", data_len, ms_tensor_ptr->DataSize());
#else
    MS_LOGE("data_len(%ld) not equal to Size of ms_tensor(%zu)", data_len, ms_tensor_ptr->DataSize());
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

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_MSTensor_setByteBufferData(JNIEnv *env, jobject thiz,
                                                                                    jlong tensor_ptr, jobject buffer) {
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

  auto *ms_tensor_ptr = static_cast<mindspore::MSTensor *>(pointer);
  if (static_cast<size_t>(data_len) != ms_tensor_ptr->DataSize()) {
#ifdef ENABLE_ARM32
    MS_LOGE("data_len(%lld) not equal to Size of ms_tensor(%zu)", data_len, ms_tensor_ptr->DataSize());
#else
    MS_LOGE("data_len(%ld) not equal to Size of ms_tensor(%zu)", data_len, ms_tensor_ptr->DataSize());
#endif
    return static_cast<jboolean>(false);
  }
  auto *local_data = ms_tensor_ptr->MutableData();
  memcpy(local_data, p_data, data_len);
  return static_cast<jboolean>(true);
}

extern "C" JNIEXPORT jlong JNICALL Java_com_mindspore_MSTensor_size(JNIEnv *env, jobject thiz, jlong tensor_ptr) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Tensor pointer from java is nullptr");
    return 0;
  }
  auto *ms_tensor_ptr = static_cast<mindspore::MSTensor *>(pointer);
  return ms_tensor_ptr->DataSize();
}

extern "C" JNIEXPORT jint JNICALL Java_com_mindspore_MSTensor_elementsNum(JNIEnv *env, jobject thiz, jlong tensor_ptr) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Tensor pointer from java is nullptr");
    return 0;
  }
  auto *ms_tensor_ptr = static_cast<mindspore::MSTensor *>(pointer);
  return ms_tensor_ptr->ElementNum();
}

extern "C" JNIEXPORT void JNICALL Java_com_mindspore_MSTensor_free(JNIEnv *env, jobject thiz, jlong tensor_ptr) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Tensor pointer from java is nullptr");
    return;
  }
  auto *ms_tensor_ptr = static_cast<mindspore::MSTensor *>(pointer);
  delete (ms_tensor_ptr);
}

extern "C" JNIEXPORT jstring JNICALL Java_com_mindspore_MSTensor_tensorName(JNIEnv *env, jobject thiz,
                                                                            jlong tensor_ptr) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOGE("Tensor pointer from java is nullptr");
    return nullptr;
  }
  auto *ms_tensor_ptr = static_cast<mindspore::MSTensor *>(pointer);

  return env->NewStringUTF(ms_tensor_ptr->Name().c_str());
}

extern "C" JNIEXPORT jlong JNICALL Java_com_mindspore_MSTensor_createTensor(JNIEnv *env, jobject thiz,
                                                                            jstring tensor_name, jobject buffer) {
  auto *p_data = reinterpret_cast<jbyte *>(env->GetDirectBufferAddress(buffer));  // get buffer pointer
  jlong data_len = env->GetDirectBufferCapacity(buffer);                          // get buffer capacity
  if (p_data == nullptr) {
    MS_LOGE("GetDirectBufferAddress return null");
    return false;
  }
  char *tensor_data(new char[data_len]);
  memcpy(tensor_data, p_data, data_len);
  int tensor_size = static_cast<jint>(data_len / sizeof(float));
  std::vector<int64_t> shape = {tensor_size};
  auto tensor =
    mindspore::MSTensor::CreateTensor(env->GetStringUTFChars(tensor_name, JNI_FALSE),
                                      mindspore::DataType::kNumberTypeFloat32, shape, tensor_data, data_len);
  return jlong(tensor);
}
