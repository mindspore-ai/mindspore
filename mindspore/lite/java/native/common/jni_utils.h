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

#ifndef MINDSPORE_LITE_JAVA_NATIVE_COMMON_JNI_UTILS_H_
#define MINDSPORE_LITE_JAVA_NATIVE_COMMON_JNI_UTILS_H_
#include <jni.h>
#include <string>
#include <vector>

std::string RealPath(const char *path);

/**
 * An util function to new a java Arraylist with the type clzname.
 * @tparam T the data type in C++
 * @param env jni env
 * @param values the java constructor params
 * @param clzname the java class name
 * @param sig the java constructor signature of the java class
 * @return
 */
template <typename T>
jobject newObjectArrayList(JNIEnv_ *env, const std::vector<T> &values, const char *clzname, const char *sig) {
  jclass array_list = env->FindClass("java/util/ArrayList");
  jmethodID array_list_construct = env->GetMethodID(array_list, "<init>", "()V");
  jobject ret = env->NewObject(array_list, array_list_construct);
  jmethodID array_list_add = env->GetMethodID(array_list, "add", "(Ljava/lang/Object;)Z");
  jclass class_object = env->FindClass(clzname);
  for (auto &value : values) {
    jmethodID object_construct = env->GetMethodID(class_object, "<init>", sig);
    jobject obj = env->NewObject(class_object, object_construct, value);
    env->CallBooleanMethod(ret, array_list_add, obj);
    env->DeleteLocalRef(obj);
  }
  env->DeleteLocalRef(array_list);
  env->DeleteLocalRef(class_object);
  return ret;
}

#endif  // MINDSPORE_LITE_JAVA_NATIVE_COMMON_JNI_UTILS_H_
