/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "include/api/model_parallel_runner.h"

extern "C" JNIEXPORT jlong JNICALL Java_com_mindspore_config_RunnerConfig_createRunnerConfig(JNIEnv *env,
                                                                                             jobject thiz) {
  auto runner_config = new (std::nothrow) mindspore::RunnerConfig();
  if (runner_config == nullptr) {
    MS_LOG(ERROR) << "new RunnerConfig fail!";
    return (jlong) nullptr;
  }
  return (jlong)runner_config;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_mindspore_config_RunnerConfig_createRunnerConfigWithContext(JNIEnv *env, jobject thiz, jlong context_ptr) {
  auto runner_config = new (std::nothrow) mindspore::RunnerConfig();
  if (runner_config == nullptr) {
    MS_LOG(ERROR) << "new RunnerConfig fail!";
    return (jlong) nullptr;
  }
  auto *c_context_ptr = reinterpret_cast<mindspore::Context *>(context_ptr);
  if (c_context_ptr == nullptr) {
    delete runner_config;
    MS_LOG(ERROR) << "Context pointer from java is nullptr";
    return (jlong) nullptr;
  }
  auto context = std::make_shared<mindspore::Context>();
  if (context == nullptr) {
    delete runner_config;
    MS_LOG(ERROR) << "Make context failed";
    return (jlong) nullptr;
  }
  context.reset(c_context_ptr);
  runner_config->SetContext(context);
  return (jlong)runner_config;
}

extern "C" JNIEXPORT void JNICALL Java_com_mindspore_config_RunnerConfig_setWorkersNum(JNIEnv *env, jobject thiz,
                                                                                       jstring runner_config_ptr,
                                                                                       jint workers_num) {
  auto *pointer = reinterpret_cast<mindspore::RunnerConfig *>(runner_config_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "runner config pointer from java is nullptr";
    return;
  }
  pointer->SetWorkersNum(workers_num);
}

extern "C" JNIEXPORT void JNICALL Java_com_mindspore_config_RunnerConfig_setConfigPath(JNIEnv *env, jobject thiz,
                                                                                       jstring runner_config_ptr,
                                                                                       jstring config_path) {
  auto *runner_config = reinterpret_cast<mindspore::RunnerConfig *>(runner_config_ptr);
  if (runner_config == nullptr) {
    MS_LOG(ERROR) << "runner config from java is nullptr";
    return;
  }
  const char *c_config_path = env->GetStringUTFChars(config_path, nullptr);
  std::string str_config_path(c_config_path, env->GetStringLength(config_path));
  runner_config->SetConfigPath(str_config_path);
  env->ReleaseStringUTFChars(config_path, c_config_path);
}

extern "C" JNIEXPORT jstring JNICALL Java_com_mindspore_config_RunnerConfig_getConfigPath(JNIEnv *env, jobject thiz,
                                                                                          jlong runner_config_ptr) {
  auto *runner_config = reinterpret_cast<mindspore::RunnerConfig *>(runner_config_ptr);
  if (runner_config == nullptr) {
    MS_LOG(ERROR) << "runner config pointer from java is nullptr";
    return nullptr;
  }
  return env->NewStringUTF(runner_config->GetConfigPath().c_str());
}

extern "C" JNIEXPORT void JNICALL Java_com_mindspore_config_RunnerConfig_setConfigInfo(JNIEnv *env, jobject thiz,
                                                                                       jstring runner_config_ptr,
                                                                                       jstring section,
                                                                                       jobject hashMapConfig) {
  auto *pointer = reinterpret_cast<mindspore::RunnerConfig *>(runner_config_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "runner config pointer from java is nullptr";
    return;
  }
  const char *c_section = env->GetStringUTFChars(section, nullptr);
  std::string str_section(c_section, env->GetStringLength(section));

  jclass classHashMap = env->FindClass("java/util/HashMap");
  jclass ClassSet = env->FindClass("java/util/Set");
  jmethodID methodIDSet = env->GetMethodID(classHashMap, "entrySet", "()Ljava/util/Set;");
  jmethodID methodIDIterator = env->GetMethodID(ClassSet, "iterator", "()Ljava/util/Iterator;");
  jobject objectMethodSet = env->CallObjectMethod(hashMapConfig, methodIDSet);
  jobject iteratorObject = env->CallObjectMethod(objectMethodSet, methodIDIterator);
  jclass classIterator = env->FindClass("java/util/Iterator");
  jmethodID nextMethodID = env->GetMethodID(classIterator, "next", "()Ljava/lang/Object;");
  jmethodID hasNextMethodID = env->GetMethodID(classIterator, "hasNext", "()Z");
  jclass classMapEntry = env->FindClass("java/util/Map$Entry");
  jmethodID keyMethodID = env->GetMethodID(classMapEntry, "getKey", "()Ljava/lang/Object;");
  jmethodID valueMethodID = env->GetMethodID(classMapEntry, "getValue", "()Ljava/lang/Object;");
  std::map<std::string, std::string> configInfo;
  while (env->CallBooleanMethod(iteratorObject, hasNextMethodID)) {
    jobject objectEntry = env->CallObjectMethod(iteratorObject, nextMethodID);
    jstring keyObjectMethod = (jstring)env->CallObjectMethod(objectEntry, keyMethodID);
    if (keyObjectMethod == nullptr) {
      continue;
    }
    const char *c_keyConfigInfo = env->GetStringUTFChars(keyObjectMethod, nullptr);
    std::string str_keyConfigInfo(c_keyConfigInfo, env->GetStringLength(keyObjectMethod));
    jstring valueObjectMethod = (jstring)env->CallObjectMethod(objectEntry, valueMethodID);
    if (valueObjectMethod == nullptr) {
      continue;
    }
    const char *c_valueConfigInfo = env->GetStringUTFChars(valueObjectMethod, nullptr);
    std::string str_valueConfigInfo(c_valueConfigInfo, env->GetStringLength(valueObjectMethod));
    configInfo.insert(std::make_pair(str_keyConfigInfo, str_valueConfigInfo));
    env->ReleaseStringUTFChars(keyObjectMethod, c_keyConfigInfo);
    env->ReleaseStringUTFChars(valueObjectMethod, c_valueConfigInfo);
    env->DeleteLocalRef(objectEntry);
    env->DeleteLocalRef(keyObjectMethod);
    env->DeleteLocalRef(valueObjectMethod);
  }
  env->DeleteLocalRef(classHashMap);
  env->DeleteLocalRef(objectMethodSet);
  env->DeleteLocalRef(ClassSet);
  env->DeleteLocalRef(iteratorObject);
  env->DeleteLocalRef(classIterator);
  env->DeleteLocalRef(classMapEntry);
  pointer->SetConfigInfo(str_section, configInfo);
  env->ReleaseStringUTFChars(section, c_section);
}

extern "C" JNIEXPORT void JNICALL Java_com_mindspore_config_RunnerConfig_free(JNIEnv *env, jobject thiz,
                                                                              jlong runner_config_ptr) {
  auto *pointer = reinterpret_cast<void *>(runner_config_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Model pointer from java is nullptr";
    return;
  }
  auto *runner_config = static_cast<mindspore::RunnerConfig *>(pointer);
  delete (runner_config);
}
