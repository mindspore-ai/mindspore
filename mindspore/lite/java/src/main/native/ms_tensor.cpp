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
#include "common/log_adapter.h"
#include "include/api/types.h"

extern "C" JNIEXPORT jintArray JNICALL Java_com_mindspore_MSTensor_getShape(JNIEnv *env, jobject thiz,
                                                                            jlong tensor_ptr) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Tensor pointer from java is nullptr";
    return env->NewIntArray(0);
  }
  auto *ms_tensor_ptr = static_cast<mindspore::MSTensor *>(pointer);
  auto local_shape = ms_tensor_ptr->Shape();
  auto shape_size = local_shape.size();
  jintArray shape = env->NewIntArray(shape_size);
  if (shape == nullptr) {
    MS_LOG(ERROR) << "new intArray failed.";
    return env->NewIntArray(0);
  }
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
    MS_LOG(ERROR) << "Tensor pointer from java is nullptr";
    return static_cast<jboolean>(false);
  }
  auto *ms_tensor_ptr = static_cast<mindspore::MSTensor *>(pointer);
  return jint(ms_tensor_ptr->DataType());
}

extern "C" JNIEXPORT jbyteArray JNICALL Java_com_mindspore_MSTensor_getByteData(JNIEnv *env, jobject thiz,
                                                                                jlong tensor_ptr) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Tensor pointer from java is nullptr";
    return env->NewByteArray(0);
  }
  auto *ms_tensor_ptr = static_cast<mindspore::MSTensor *>(pointer);
  auto *local_data = static_cast<jbyte *>(ms_tensor_ptr->MutableData());
  if (local_data == nullptr) {
    MS_LOG(DEBUG) << "Tensor has no data";
    return env->NewByteArray(0);
  }

  auto local_size = ms_tensor_ptr->DataSize();
  if (local_size <= 0) {
    MS_LOG(ERROR) << "Size of tensor is negative: " << local_size;
    return env->NewByteArray(0);
  }
  auto ret = env->NewByteArray(local_size);
  if (ret == nullptr) {
    MS_LOG(ERROR) << "malloc failed.";
    return env->NewByteArray(0);
  }
  env->SetByteArrayRegion(ret, 0, local_size, local_data);
  return ret;
}

extern "C" JNIEXPORT jlongArray JNICALL Java_com_mindspore_MSTensor_getLongData(JNIEnv *env, jobject thiz,
                                                                                jlong tensor_ptr) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Tensor pointer from java is nullptr";
    return env->NewLongArray(0);
  }

  auto *ms_tensor_ptr = static_cast<mindspore::MSTensor *>(pointer);

  auto *local_data = static_cast<jlong *>(ms_tensor_ptr->MutableData());
  if (local_data == nullptr) {
    MS_LOG(DEBUG) << "Tensor has no data";
    return env->NewLongArray(0);
  }

  if (ms_tensor_ptr->DataType() != mindspore::DataType::kNumberTypeInt64) {
    MS_LOG(ERROR) << "data type is error : " << static_cast<int>(ms_tensor_ptr->DataType());
    return env->NewLongArray(0);
  }
  auto local_element_num = ms_tensor_ptr->ElementNum();
  if (local_element_num <= 0) {
    MS_LOG(ERROR) << "ElementsNum of tensor is negative: " << static_cast<int>(local_element_num);
    return env->NewLongArray(0);
  }
  auto ret = env->NewLongArray(local_element_num);
  if (ret == nullptr) {
    MS_LOG(ERROR) << "malloc failed.";
    return env->NewLongArray(0);
  }
  env->SetLongArrayRegion(ret, 0, local_element_num, local_data);
  return ret;
}

extern "C" JNIEXPORT jintArray JNICALL Java_com_mindspore_MSTensor_getIntData(JNIEnv *env, jobject thiz,
                                                                              jlong tensor_ptr) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Tensor pointer from java is nullptr";
    return env->NewIntArray(0);
  }

  auto *ms_tensor_ptr = static_cast<mindspore::MSTensor *>(pointer);

  auto *local_data = static_cast<jint *>(ms_tensor_ptr->MutableData());
  if (local_data == nullptr) {
    MS_LOG(DEBUG) << "Tensor has no data";
    return env->NewIntArray(0);
  }

  if (ms_tensor_ptr->DataType() != mindspore::DataType::kNumberTypeInt32) {
    MS_LOG(ERROR) << "data type is error : " << static_cast<int>(ms_tensor_ptr->DataType());
    return env->NewIntArray(0);
  }
  auto local_element_num = ms_tensor_ptr->ElementNum();
  if (local_element_num <= 0) {
    MS_LOG(ERROR) << "ElementsNum of tensor is negative: " << static_cast<int>(local_element_num);
    return env->NewIntArray(0);
  }
  auto ret = env->NewIntArray(local_element_num);
  if (ret == nullptr) {
    MS_LOG(ERROR) << "malloc failed.";
    return env->NewIntArray(0);
  }
  env->SetIntArrayRegion(ret, 0, local_element_num, local_data);
  return ret;
}

extern "C" JNIEXPORT jfloatArray JNICALL Java_com_mindspore_MSTensor_getFloatData(JNIEnv *env, jobject thiz,
                                                                                  jlong tensor_ptr) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Tensor pointer from java is nullptr";
    return env->NewFloatArray(0);
  }

  auto *ms_tensor_ptr = static_cast<mindspore::MSTensor *>(pointer);

  auto *local_data = static_cast<jfloat *>(ms_tensor_ptr->MutableData());
  if (local_data == nullptr) {
    MS_LOG(DEBUG) << "Tensor has no data";
    return env->NewFloatArray(0);
  }

  if (ms_tensor_ptr->DataType() != mindspore::DataType::kNumberTypeFloat32) {
    MS_LOG(ERROR) << "data type is error : " << static_cast<int>(ms_tensor_ptr->DataType());
    return env->NewFloatArray(0);
  }
  auto local_element_num = ms_tensor_ptr->ElementNum();
  if (local_element_num <= 0) {
    MS_LOG(ERROR) << "ElementsNum of tensor is negative: " << static_cast<int>(local_element_num);
    return env->NewFloatArray(0);
  }
  auto ret = env->NewFloatArray(local_element_num);
  if (ret == nullptr) {
    MS_LOG(ERROR) << "malloc failed.";
    return env->NewFloatArray(0);
  }
  env->SetFloatArrayRegion(ret, 0, local_element_num, local_data);
  return ret;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_MSTensor_setByteData(JNIEnv *env, jobject thiz,
                                                                              jlong tensor_ptr, jbyteArray data,
                                                                              jlong data_len) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Tensor pointer from java is nullptr";
    return static_cast<jboolean>(false);
  }
  auto *ms_tensor_ptr = static_cast<mindspore::MSTensor *>(pointer);
  if (static_cast<size_t>(data_len) != ms_tensor_ptr->DataSize()) {
    MS_LOG(ERROR) << "data_len(" << data_len << ") not equal to Size of ms_tensor(" << ms_tensor_ptr->DataSize() << ")";
    return static_cast<jboolean>(false);
  }
  jboolean is_copy = false;

  if (data == nullptr) {
    MS_LOG(ERROR) << "data from java is nullptr.";
    return static_cast<jboolean>(false);
  }
  auto *data_arr = env->GetByteArrayElements(data, &is_copy);
  auto *local_data = ms_tensor_ptr->MutableData();
  if (data_arr == nullptr || local_data == nullptr) {
    MS_LOG(ERROR) << "data_arr or local_data is nullptr.";
    env->ReleaseByteArrayElements(data, data_arr, JNI_ABORT);
    return static_cast<jboolean>(false);
  }
  memcpy(local_data, data_arr, data_len);
  env->ReleaseByteArrayElements(data, data_arr, JNI_ABORT);
  return static_cast<jboolean>(true);
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_MSTensor_setFloatData(JNIEnv *env, jobject thiz,
                                                                               jlong tensor_ptr, jfloatArray data,
                                                                               jlong data_len) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Tensor pointer from java is nullptr";
    return static_cast<jboolean>(false);
  }
  auto *ms_tensor_ptr = static_cast<mindspore::MSTensor *>(pointer);
  const int float_flag = 41;
  if (ms_tensor_ptr->DataType() != mindspore::DataType::kNumberTypeFloat32 &&
      static_cast<int>(ms_tensor_ptr->DataType()) != float_flag) {
    MS_LOG(ERROR) << "data_type must be Float32(43), but got (" << static_cast<int>(ms_tensor_ptr->DataType()) << ").";
    return static_cast<jboolean>(false);
  }
  if (data_len != ms_tensor_ptr->ElementNum()) {
    MS_LOG(ERROR) << "data_len(" << data_len << ") not equal to Size of ms_tensor(" << ms_tensor_ptr->DataSize() << ")";
    return static_cast<jboolean>(false);
  }
  auto *local_data = reinterpret_cast<jfloat *>(ms_tensor_ptr->MutableData());
  if (local_data == nullptr) {
    MS_LOG(ERROR) << "malloc memory failed.";
    return static_cast<jboolean>(false);
  }
  if (data == nullptr) {
    MS_LOG(ERROR) << "data from java is nullptr";
    return static_cast<jboolean>(false);
  }
  env->GetFloatArrayRegion(data, 0, static_cast<jsize>(data_len), local_data);
  return static_cast<jboolean>(true);
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_MSTensor_setIntData(JNIEnv *env, jobject thiz,
                                                                             jlong tensor_ptr, jintArray data,
                                                                             jlong data_len) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Tensor pointer from java is nullptr";
    return static_cast<jboolean>(false);
  }
  auto *ms_tensor_ptr = static_cast<mindspore::MSTensor *>(pointer);
  const int int_flag = 31;
  if (ms_tensor_ptr->DataType() != mindspore::DataType::kNumberTypeInt32 &&
      static_cast<int>(ms_tensor_ptr->DataType()) != int_flag) {
    MS_LOG(ERROR) << "data_type must be Int32(34), but got (" << static_cast<int>(ms_tensor_ptr->DataType()) << ").";
    return static_cast<jboolean>(false);
  }
  if (data_len != ms_tensor_ptr->ElementNum()) {
    MS_LOG(ERROR) << "data_len(" << data_len << ") not equal to Size of ms_tensor(" << ms_tensor_ptr->DataSize() << ")";
    return static_cast<jboolean>(false);
  }
  auto *local_data = reinterpret_cast<jint *>(ms_tensor_ptr->MutableData());
  if (local_data == nullptr) {
    MS_LOG(ERROR) << "malloc memory failed.";
    return static_cast<jboolean>(false);
  }
  if (data == nullptr) {
    MS_LOG(ERROR) << "data from java is nullptr";
    return static_cast<jboolean>(false);
  }
  env->GetIntArrayRegion(data, 0, static_cast<jsize>(data_len), local_data);
  return static_cast<jboolean>(true);
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_MSTensor_setLongData(JNIEnv *env, jobject thiz,
                                                                              jlong tensor_ptr, jlongArray data,
                                                                              jlong data_len) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Tensor pointer from java is nullptr";
    return static_cast<jboolean>(false);
  }
  auto *ms_tensor_ptr = static_cast<mindspore::MSTensor *>(pointer);
  if (ms_tensor_ptr->DataType() != mindspore::DataType::kNumberTypeInt64) {
    MS_LOG(ERROR) << "data_type must be Int64(35), but got (" << static_cast<int>(ms_tensor_ptr->DataType()) << ").";
    return static_cast<jboolean>(false);
  }
  if (data_len != ms_tensor_ptr->ElementNum()) {
    MS_LOG(ERROR) << "data_len(" << data_len << ") not equal to Size of ms_tensor(" << ms_tensor_ptr->DataSize() << ")";
    return static_cast<jboolean>(false);
  }
  auto *local_data = reinterpret_cast<jlong *>(ms_tensor_ptr->MutableData());
  if (local_data == nullptr) {
    MS_LOG(ERROR) << "malloc memory failed.";
    return static_cast<jboolean>(false);
  }
  if (data == nullptr) {
    MS_LOG(ERROR) << "data from java is nullptr";
    return static_cast<jboolean>(false);
  }
  env->GetLongArrayRegion(data, 0, static_cast<jsize>(data_len), local_data);
  return static_cast<jboolean>(true);
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_MSTensor_setByteBufferData(JNIEnv *env, jobject thiz,
                                                                                    jlong tensor_ptr, jobject buffer) {
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

  auto *ms_tensor_ptr = static_cast<mindspore::MSTensor *>(pointer);
  if (static_cast<size_t>(data_len) != ms_tensor_ptr->DataSize()) {
    MS_LOG(ERROR) << "data_len(" << data_len << ") not equal to Size of ms_tensor(" << ms_tensor_ptr->DataSize() << ")";
    return static_cast<jboolean>(false);
  }
  auto *local_data = ms_tensor_ptr->MutableData();
  if (local_data == nullptr) {
    MS_LOG(ERROR) << "get MutableData nullptr.";
    return static_cast<jboolean>(false);
  }
  memcpy(local_data, p_data, data_len);
  return static_cast<jboolean>(true);
}

extern "C" JNIEXPORT jlong JNICALL Java_com_mindspore_MSTensor_size(JNIEnv *env, jobject thiz, jlong tensor_ptr) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Tensor pointer from java is nullptr";
    return 0;
  }
  auto *ms_tensor_ptr = static_cast<mindspore::MSTensor *>(pointer);
  return ms_tensor_ptr->DataSize();
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_MSTensor_setShape(JNIEnv *env, jobject thiz, jlong tensor_ptr,
                                                                           jintArray tensor_shape) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr || tensor_shape == nullptr) {
    MS_LOG(ERROR) << "input params from java is nullptr";
    return static_cast<jboolean>(false);
  }

  auto *ms_tensor_ptr = static_cast<mindspore::MSTensor *>(pointer);
  auto size = static_cast<int>(env->GetArrayLength(tensor_shape));
  std::vector<int64_t> c_shape(size);
  jint *shape_pointer = env->GetIntArrayElements(tensor_shape, nullptr);
  for (int i = 0; i < size; i++) {
    c_shape[i] = static_cast<int64_t>(shape_pointer[i]);
  }
  env->ReleaseIntArrayElements(tensor_shape, shape_pointer, JNI_ABORT);
  ms_tensor_ptr->SetShape(c_shape);
  return static_cast<jboolean>(true);
}

extern "C" JNIEXPORT jint JNICALL Java_com_mindspore_MSTensor_elementsNum(JNIEnv *env, jobject thiz, jlong tensor_ptr) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Tensor pointer from java is nullptr";
    return 0;
  }
  auto *ms_tensor_ptr = static_cast<mindspore::MSTensor *>(pointer);
  return ms_tensor_ptr->ElementNum();
}

extern "C" JNIEXPORT void JNICALL Java_com_mindspore_MSTensor_free(JNIEnv *env, jobject thiz, jlong tensor_ptr) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Tensor pointer from java is nullptr";
    return;
  }
  auto *ms_tensor_ptr = static_cast<mindspore::MSTensor *>(pointer);
  delete (ms_tensor_ptr);
}

extern "C" JNIEXPORT jstring JNICALL Java_com_mindspore_MSTensor_tensorName(JNIEnv *env, jobject thiz,
                                                                            jlong tensor_ptr) {
  auto *pointer = reinterpret_cast<void *>(tensor_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Tensor pointer from java is nullptr";
    return nullptr;
  }
  auto *ms_tensor_ptr = static_cast<mindspore::MSTensor *>(pointer);

  return env->NewStringUTF(ms_tensor_ptr->Name().c_str());
}

extern "C" JNIEXPORT jlong JNICALL Java_com_mindspore_MSTensor_createTensorByNative(JNIEnv *env, jobject thiz,
                                                                                    jstring tensor_name, jint data_type,
                                                                                    jintArray tensor_shape,
                                                                                    jobject buffer) {
  // check inputs
  if (buffer == nullptr || tensor_name == nullptr || tensor_shape == nullptr) {
    MS_LOG(ERROR) << "input param from java is nullptr";
    return 0;
  }
  auto *p_data = reinterpret_cast<jbyte *>(env->GetDirectBufferAddress(buffer));
  jlong data_len = env->GetDirectBufferCapacity(buffer);
  if (p_data == nullptr) {
    MS_LOG(ERROR) << "GetDirectBufferAddress return null";
    return false;
  }

  auto size = static_cast<int>(env->GetArrayLength(tensor_shape));
  std::vector<int64_t> c_shape(size);
  jint *shape_pointer = env->GetIntArrayElements(tensor_shape, nullptr);
  for (int i = 0; i < size; i++) {
    c_shape[i] = static_cast<int64_t>(shape_pointer[i]);
  }
  const char *c_tensor_name = env->GetStringUTFChars(tensor_name, nullptr);
  std::string str_tensor_name(c_tensor_name, env->GetStringLength(tensor_name));
  auto tensor = mindspore::MSTensor::CreateTensor(str_tensor_name, static_cast<mindspore::DataType>(data_type), c_shape,
                                                  p_data, data_len);
  env->ReleaseIntArrayElements(tensor_shape, shape_pointer, JNI_ABORT);
  env->ReleaseStringUTFChars(tensor_name, c_tensor_name);
  return jlong(tensor);
}

bool WriteScalar(JNIEnv *env, jobject src, jint data_type, void *dst) {
#define CASE(type, jtype, dtype, name, sig)            \
  case mindspore::DataType::dtype: {                   \
    jclass clazz = env->FindClass("java/lang/Number"); \
    jmethodID id = env->GetMethodID(clazz, name, sig); \
    jtype src_data = env->Call##type##Method(src, id); \
    *static_cast<jtype *>(dst) = src_data;             \
    break;                                             \
  }
  switch (static_cast<mindspore::DataType>(data_type)) {
    CASE(Float, jfloat, kNumberTypeFloat32, "floatValue", "()F");
    CASE(Int, jint, kNumberTypeInt32, "intValue", "()I");
    CASE(Long, jlong, kNumberTypeInt64, "longValue", "()J");
    CASE(Boolean, jboolean, kNumberTypeBool, "booleanValue", "()Z");
#undef CASE
    default:
      MS_LOG(ERROR) << "The dataType only support float32, int32, int64 and bool, but got " << data_type;
      return false;
  }
  return true;
}

bool Write1DArray(JNIEnv *env, jarray src, jint data_type, void *dst, size_t *offset, int require_len) {
#define CASE(type, jtype, dtype)                                                                                      \
  case mindspore::DataType::dtype:                                                                                    \
    env->Get##type##ArrayRegion(static_cast<jtype##Array>(src), 0, element_num, static_cast<jtype *>(dst) + *offset); \
    break;

  int element_num = env->GetArrayLength(src);
  if (element_num != require_len) {
    MS_LOG(ERROR) << "The length of 1DArray " << element_num << "is not equal to " << require_len;
    return false;
  }
  switch (static_cast<mindspore::DataType>(data_type)) {
    CASE(Float, jfloat, kNumberTypeFloat32);
    CASE(Int, jint, kNumberTypeInt32);
    CASE(Long, jlong, kNumberTypeInt64);
    CASE(Boolean, jboolean, kNumberTypeBool);
#undef CASE
    default:
      MS_LOG(ERROR) << "The dataType only support float32, int32, int64 and bool, but got " << data_type;
      return false;
  }
  *offset += element_num;
  return true;
}

bool WriteNDArray(JNIEnv *env, jarray src, jint data_type, void *dst, size_t *offset, const std::vector<int> &shape,
                  size_t dim_index) {
  if (dim_index + 1 == shape.size()) {
    return Write1DArray(env, src, data_type, dst, offset, shape.back());
  } else {
    auto nd_src = static_cast<jobjectArray>(src);
    int len = env->GetArrayLength(nd_src);
    if (len != shape[dim_index]) {
      MS_LOG(ERROR) << "The dim-value " << len << "is not equal to " << shape[dim_index];
      return false;
    }
    for (int i = 0; i < len; ++i) {
      auto row = static_cast<jarray>(env->GetObjectArrayElement(nd_src, i));
      auto ret = WriteNDArray(env, row, data_type, dst, offset, shape, dim_index + 1);
      if (!ret) {
        MS_LOG(ERROR) << "Write data failed.";
        return false;
      }
    }
  }
  return true;
}

extern "C" JNIEXPORT jlong JNICALL Java_com_mindspore_MSTensor_createTensorByObject(JNIEnv *env, jobject thiz,
                                                                                    jstring tensor_name, jint data_type,
                                                                                    jintArray tensor_shape,
                                                                                    jobject value) {
  auto size = static_cast<jsize>(env->GetArrayLength(tensor_shape));
  std::vector<int> shape(size);
  env->GetIntArrayRegion(tensor_shape, 0, size, shape.data());
  std::vector<int64_t> c_shape(shape.begin(), shape.end());

  const char *c_tensor_name = env->GetStringUTFChars(tensor_name, nullptr);
  std::string str_tensor_name(c_tensor_name, env->GetStringLength(tensor_name));
  env->ReleaseStringUTFChars(tensor_name, c_tensor_name);
  auto tensor = mindspore::MSTensor::CreateTensor(str_tensor_name, static_cast<mindspore::DataType>(data_type), c_shape,
                                                  nullptr, 0);
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "Create a tensor failed.";
    return 0;
  }
  auto *local_data = tensor->MutableData();
  if (local_data == nullptr) {
    delete tensor;
    MS_LOG(ERROR) << "Create a tensor failed, due to fail to alloc.";
    return 0;
  }
  bool ret = true;
  if (size == 0) {
    ret = WriteScalar(env, value, data_type, local_data);
  } else {
    size_t offset = 0;
    ret = WriteNDArray(env, static_cast<jarray>(value), data_type, local_data, &offset, shape, 0);
  }
  if (!ret) {
    delete tensor;
    tensor = nullptr;
    MS_LOG(ERROR) << "Create a tensor failed, due to fail to write.";
  }
  return jlong(tensor);
}

