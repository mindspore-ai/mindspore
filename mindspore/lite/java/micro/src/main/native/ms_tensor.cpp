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
#include "src/common/log.h"
#include "c_api/tensor_c.h"
#include "c_api/data_type_c.h"

extern "C" JNIEXPORT jintArray JNICALL Java_com_mindspore_micro_MSTensor_getShape(JNIEnv *env, jobject thiz,
                                                                                  jlong tensor_ptr) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Tensor pointer from java is nullptr";
    return env->NewIntArray(0);
  }
  auto *ms_tensor_ptr = static_cast<MSTensorHandle *>(pointer);
  size_t shape_num = 0;
  const int64_t *dims = MSTensorGetShape(ms_tensor_ptr, &shape_num);
  jintArray shape = env->NewIntArray((jsize)shape_num);
  auto *tmp = new jint[shape_num];
  for (size_t i = 0; i < shape_num; i++) {
    tmp[i] = static_cast<int>(dims[i]);
  }
  env->SetIntArrayRegion(shape, 0, (jsize)shape_num, tmp);
  delete[] (tmp);
  return shape;
}

extern "C" JNIEXPORT jint JNICALL Java_com_mindspore_micro_MSTensor_getDataType(JNIEnv *env, jobject thiz,
                                                                                jlong tensor_ptr) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Tensor pointer from java is nullptr";
    return static_cast<jboolean>(false);
  }
  auto *ms_tensor_ptr = static_cast<MSTensorHandle *>(pointer);
  return jint(MSTensorGetDataType(ms_tensor_ptr));
}

extern "C" JNIEXPORT jfloatArray JNICALL Java_com_mindspore_micro_MSTensor_getFloatData(JNIEnv *env, jobject thiz,
                                                                                        jlong tensor_ptr) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Tensor pointer from java is nullptr";
    return env->NewFloatArray(0);
  }
  auto *ms_tensor_ptr = static_cast<MSTensorHandle *>(pointer);

  auto *local_data = reinterpret_cast<jfloat *>(MSTensorGetMutableData(ms_tensor_ptr));
  if (local_data == nullptr) {
    MS_LOG(DEBUG) << "Tensor has no data";
    return env->NewFloatArray(0);
  }
  MSDataType data_type = MSTensorGetDataType(ms_tensor_ptr);
  if (data_type != MSDataType::kMSDataTypeNumberTypeFloat16 && data_type != MSDataType::kMSDataTypeNumberTypeFloat32) {
    MS_LOG(ERROR) << "data type is error: " << static_cast<int>(data_type);
    return env->NewFloatArray(0);
  }
  auto local_element_num = MSTensorGetElementNum(ms_tensor_ptr);
  if (local_element_num <= 0) {
    MS_LOG(ERROR) << "ElementsNum of tensor is negative: " << static_cast<int>(local_element_num);
    return env->NewFloatArray(0);
  }
  auto ret = env->NewFloatArray((jsize)local_element_num);
  env->SetFloatArrayRegion(ret, 0, (jsize)local_element_num, local_data);
  return ret;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_micro_MSTensor_setFloatData(JNIEnv *env, jobject thiz,
                                                                                     jlong tensor_ptr, jfloatArray data,
                                                                                     jlong data_len) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Tensor pointer from java is nullptr";
    return static_cast<jboolean>(false);
  }
  auto *ms_tensor_ptr = static_cast<MSTensorHandle *>(pointer);
  size_t tensor_size = MSTensorGetElementNum(ms_tensor_ptr);
  if (static_cast<size_t>(data_len) != tensor_size) {
    MS_LOG(ERROR) << "data_len(" << data_len << ") not equal to Size of ms_tensor(" << tensor_size << ")";
    return static_cast<jboolean>(false);
  }
  auto *local_data = reinterpret_cast<jfloat *>(MSTensorGetMutableData(ms_tensor_ptr));
  env->GetFloatArrayRegion(data, 0, static_cast<jsize>(data_len), local_data);
  return static_cast<jboolean>(true);
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_micro_MSTensor_setByteBufferData(JNIEnv *env, jobject thiz,
                                                                                          jlong tensor_ptr,
                                                                                          jobject buffer) {
  auto *p_data = reinterpret_cast<jbyte *>(env->GetDirectBufferAddress(buffer));  // get buffer pointer
  jlong data_len = env->GetDirectBufferCapacity(buffer);                          // get buffer capacity
  if (p_data == nullptr) {
    MS_LOG(ERROR) << "GetDirectBufferAddress return null";
    return false;
  }
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Tensor pointer from java is nullptr";
    return static_cast<jboolean>(false);
  }
  auto *ms_tensor_ptr = static_cast<MSTensorHandle *>(pointer);
  size_t tensor_size = MSTensorGetDataSize(ms_tensor_ptr);
  if (static_cast<size_t>(data_len) != tensor_size) {
    MS_LOG(ERROR) << "data_len(" << data_len << ") not equal to Size of ms_tensor(" << tensor_size << ")";
    return static_cast<jboolean>(false);
  }
  MSTensorSetData(ms_tensor_ptr, p_data);
  return static_cast<jboolean>(true);
}

extern "C" JNIEXPORT jlong JNICALL Java_com_mindspore_micro_MSTensor_size(JNIEnv *env, jobject thiz, jlong tensor_ptr) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Tensor pointer from java is nullptr";
    return 0;
  }
  auto *ms_tensor_ptr = static_cast<MSTensorHandle *>(pointer);
  return (jlong)MSTensorGetDataSize(ms_tensor_ptr);
}

extern "C" JNIEXPORT void JNICALL Java_com_mindspore_micro_MSTensor_free(JNIEnv *env, jobject thiz, jlong tensor_ptr) {
  auto pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Tensor pointer from java is nullptr";
    return;
  }
  auto ms_tensor_ptr = static_cast<MSTensorHandle>(pointer);
  MSTensorDestroy(&ms_tensor_ptr);
}
