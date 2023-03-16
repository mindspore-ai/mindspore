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

#include "include/api/model.h"
#include <jni.h>
#include "common/log_adapter.h"
#include "include/api/serialization.h"

extern "C" JNIEXPORT jlong JNICALL Java_com_mindspore_Model_createModel(JNIEnv *env, jobject thiz) {
  auto model = new (std::nothrow) mindspore::Model();
  if (model == nullptr) {
    MS_LOG(ERROR) << "createModel failed";
    return jlong(nullptr);
  }
  return jlong(model);
}

extern "C" JNIEXPORT bool JNICALL Java_com_mindspore_Model_buildByGraph(JNIEnv *env, jobject thiz, jlong model_ptr,
                                                                        jlong graph_ptr, jlong context_ptr,
                                                                        jlong cfg_ptr) {
  auto *pointer = reinterpret_cast<void *>(model_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Session pointer from java is nullptr";
    return false;
  }
  auto *lite_model_ptr = static_cast<mindspore::Model *>(pointer);

  auto *c_graph_ptr = reinterpret_cast<mindspore::Graph *>(graph_ptr);
  if (c_graph_ptr == nullptr) {
    MS_LOG(ERROR) << "Graph pointer from java is nullptr";
    return false;
  }

  auto *c_context_ptr = reinterpret_cast<mindspore::Context *>(context_ptr);
  if (c_context_ptr == nullptr) {
    MS_LOG(ERROR) << "Context pointer from java is nullptr";
    return false;
  }
  auto context = std::make_shared<mindspore::Context>(*c_context_ptr);
  auto *c_cfg_ptr = reinterpret_cast<mindspore::TrainCfg *>(cfg_ptr);
  auto cfg = std::make_shared<mindspore::TrainCfg>();
  if (cfg == nullptr) {
    MS_LOG(ERROR) << "Make train config failed";
    return false;
  }
  if (c_cfg_ptr != nullptr) {
    cfg.reset(c_cfg_ptr);
  } else {
    cfg.reset();
  }
  auto status = lite_model_ptr->Build(mindspore::GraphCell(*c_graph_ptr), context, cfg);
  if (status != mindspore::kSuccess) {
    MS_LOG(ERROR) << "Error status " << static_cast<int>(status) << " during build of model";
    return false;
  }
  return true;
}

extern "C" JNIEXPORT bool JNICALL Java_com_mindspore_Model_buildByBuffer(JNIEnv *env, jobject thiz, jlong model_ptr,
                                                                         jobject model_buffer, jint model_type,
                                                                         jlong context_ptr, jcharArray key_str,
                                                                         jstring dec_mod, jstring cropto_lib_path) {
  auto *pointer = reinterpret_cast<void *>(model_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Session pointer from java is nullptr";
    return false;
  }
  auto *lite_model_ptr = static_cast<mindspore::Model *>(pointer);

  if (model_buffer == nullptr) {
    MS_LOG(ERROR) << "Buffer from java is nullptr";
    return false;
  }
  mindspore::ModelType c_model_type;
  if (model_type >= static_cast<int>(mindspore::kMindIR) && model_type <= static_cast<int>(mindspore::kMindIR_Lite)) {
    c_model_type = static_cast<mindspore::ModelType>(model_type);
  } else {
    MS_LOG(ERROR) << "Invalid model type : " << model_type;
    return false;
  }
  jlong buffer_len = env->GetDirectBufferCapacity(model_buffer);
  auto *model_buf = static_cast<char *>(env->GetDirectBufferAddress(model_buffer));

  auto *c_context_ptr = reinterpret_cast<mindspore::Context *>(context_ptr);
  if (c_context_ptr == nullptr) {
    MS_LOG(ERROR) << "Context pointer from java is nullptr";
    return false;
  }
  auto context = std::make_shared<mindspore::Context>(*c_context_ptr);
  mindspore::Status status;
  if (key_str != NULL) {
    auto key_len = static_cast<size_t>(env->GetArrayLength(key_str));
    char *dec_key_data = new (std::nothrow) char[key_len];
    if (dec_key_data == nullptr) {
      MS_LOG(ERROR) << "Dec key new failed";
      return false;
    }
    jchar *key_array = env->GetCharArrayElements(key_str, NULL);
    if (key_array == nullptr) {
      MS_LOG(ERROR) << "key_array is nullptr.";
      return false;
    }
    for (size_t i = 0; i < key_len; i++) {
      dec_key_data[i] = key_array[i];
    }
    env->ReleaseCharArrayElements(key_str, key_array, JNI_ABORT);
    mindspore::Key dec_key{dec_key_data, key_len};
    if (cropto_lib_path == nullptr || dec_mod == nullptr) {
      MS_LOG(ERROR) << "cropto_lib_path or dec_mod from java is nullptr.";
      return jlong(nullptr);
    }
    auto c_dec_mod = env->GetStringUTFChars(dec_mod, JNI_FALSE);
    auto c_cropto_lib_path = env->GetStringUTFChars(cropto_lib_path, JNI_FALSE);
    status = lite_model_ptr->Build(model_buf, buffer_len, c_model_type, context, dec_key, c_dec_mod, c_cropto_lib_path);
    env->ReleaseStringUTFChars(cropto_lib_path, c_cropto_lib_path);
    env->ReleaseStringUTFChars(dec_mod, c_dec_mod);
    delete[] dec_key_data;
  } else {
    status = lite_model_ptr->Build(model_buf, buffer_len, c_model_type, context);
  }
  if (status != mindspore::kSuccess) {
    MS_LOG(ERROR) << "Error status " << static_cast<int>(status) << " during build of model";
    return false;
  }
  return true;
}

extern "C" JNIEXPORT bool JNICALL Java_com_mindspore_Model_buildByPath(JNIEnv *env, jobject thiz, jlong model_ptr,
                                                                       jstring model_path, jint model_type,
                                                                       jlong context_ptr, jcharArray key_str,
                                                                       jstring dec_mod, jstring cropto_lib_path) {
  auto *pointer = reinterpret_cast<void *>(model_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Session pointer from java is nullptr";
    return false;
  }
  auto *lite_model_ptr = static_cast<mindspore::Model *>(pointer);
  auto c_model_path = env->GetStringUTFChars(model_path, JNI_FALSE);
  mindspore::ModelType c_model_type;
  if (model_type >= static_cast<int>(mindspore::kMindIR) && model_type <= static_cast<int>(mindspore::kMindIR_Lite)) {
    c_model_type = static_cast<mindspore::ModelType>(model_type);
  } else {
    MS_LOG(ERROR) << "Invalid model type : " << model_type;
    return false;
  }
  auto *c_context_ptr = reinterpret_cast<mindspore::Context *>(context_ptr);
  if (c_context_ptr == nullptr) {
    MS_LOG(ERROR) << "Context pointer from java is nullptr";
    return false;
  }
  auto context = std::make_shared<mindspore::Context>(*c_context_ptr);
  mindspore::Status status;
  if (key_str != NULL) {
    auto key_len = static_cast<size_t>(env->GetArrayLength(key_str));
    char *dec_key_data = new (std::nothrow) char[key_len];
    if (dec_key_data == nullptr) {
      MS_LOG(ERROR) << "Dec key new failed";
      env->ReleaseStringUTFChars(model_path, c_model_path);
      return false;
    }

    jchar *key_array = env->GetCharArrayElements(key_str, NULL);
    if (key_array == nullptr) {
      MS_LOG(ERROR) << "GetCharArrayElements failed.";
      env->ReleaseStringUTFChars(model_path, c_model_path);
      return jlong(nullptr);
    }
    for (size_t i = 0; i < key_len; i++) {
      dec_key_data[i] = key_array[i];
    }
    env->ReleaseCharArrayElements(key_str, key_array, JNI_ABORT);
    mindspore::Key dec_key{dec_key_data, key_len};

    if (dec_mod == nullptr || cropto_lib_path == nullptr) {
      MS_LOG(ERROR) << "dec_mod, cropto_lib_path from java is nullptr.";
      env->ReleaseStringUTFChars(model_path, c_model_path);
      return jlong(nullptr);
    }
    auto c_dec_mod = env->GetStringUTFChars(dec_mod, JNI_FALSE);
    auto c_cropto_lib_path = env->GetStringUTFChars(cropto_lib_path, JNI_FALSE);
    status = lite_model_ptr->Build(c_model_path, c_model_type, context, dec_key, c_dec_mod, c_cropto_lib_path);
    env->ReleaseStringUTFChars(dec_mod, c_dec_mod);
    env->ReleaseStringUTFChars(cropto_lib_path, c_cropto_lib_path);
    delete[] dec_key_data;
  } else {
    status = lite_model_ptr->Build(c_model_path, c_model_type, context);
  }
  env->ReleaseStringUTFChars(model_path, c_model_path);
  if (status != mindspore::kSuccess) {
    MS_LOG(ERROR) << "Error status " << static_cast<int>(status) << " during build of model";
    return false;
  }
  return true;
}

jobject GetInOrOutTensors(JNIEnv *env, jobject thiz, jlong model_ptr, bool is_input) {
  jclass array_list = env->FindClass("java/util/ArrayList");
  jmethodID array_list_construct = env->GetMethodID(array_list, "<init>", "()V");
  jobject ret = env->NewObject(array_list, array_list_construct);
  jmethodID array_list_add = env->GetMethodID(array_list, "add", "(Ljava/lang/Object;)Z");

  jclass long_object = env->FindClass("java/lang/Long");
  jmethodID long_object_construct = env->GetMethodID(long_object, "<init>", "(J)V");
  auto *pointer = reinterpret_cast<mindspore::Model *>(model_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Model pointer from java is nullptr";
    env->DeleteLocalRef(array_list);
    env->DeleteLocalRef(long_object);
    return ret;
  }
  std::vector<mindspore::MSTensor> tensors;
  if (is_input) {
    tensors = pointer->GetInputs();
  } else {
    tensors = pointer->GetOutputs();
  }
  for (auto &tensor : tensors) {
    auto tensor_ptr = std::make_unique<mindspore::MSTensor>(tensor);
    if (tensor_ptr == nullptr) {
      MS_LOG(ERROR) << "Make ms tensor failed";
      return ret;
    }
    jobject tensor_addr = env->NewObject(long_object, long_object_construct, jlong(tensor_ptr.release()));
    env->CallBooleanMethod(ret, array_list_add, tensor_addr);
    env->DeleteLocalRef(tensor_addr);
  }
  env->DeleteLocalRef(array_list);
  env->DeleteLocalRef(long_object);
  return ret;
}

jlong GetTensorByInOutName(JNIEnv *env, jlong model_ptr, jstring tensor_name, bool is_input) {
  auto *pointer = reinterpret_cast<void *>(model_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Model pointer from java is nullptr";
    return jlong(nullptr);
  }
  auto *lite_model_ptr = static_cast<mindspore::Model *>(pointer);
  mindspore::MSTensor tensor;
  if (tensor_name == nullptr) {
    MS_LOG(ERROR) << "tensor_name from java is nullptr.";
    return jlong(nullptr);
  }
  auto c_tensor_name = env->GetStringUTFChars(tensor_name, JNI_FALSE);
  if (is_input) {
    tensor = lite_model_ptr->GetInputByTensorName(c_tensor_name);
  } else {
    tensor = lite_model_ptr->GetOutputByTensorName(c_tensor_name);
  }
  env->ReleaseStringUTFChars(tensor_name, c_tensor_name);
  if (tensor.impl() == nullptr) {
    return jlong(nullptr);
  }
  auto tensor_ptr = std::make_unique<mindspore::MSTensor>(tensor);
  if (tensor_ptr == nullptr) {
    MS_LOG(ERROR) << "Make ms tensor failed";
    return jlong(nullptr);
  }
  return jlong(tensor_ptr.release());
}

extern "C" JNIEXPORT jobject JNICALL Java_com_mindspore_Model_getInputs(JNIEnv *env, jobject thiz, jlong model_ptr) {
  return GetInOrOutTensors(env, thiz, model_ptr, true);
}

extern "C" JNIEXPORT jlong JNICALL Java_com_mindspore_Model_getInputByTensorName(JNIEnv *env, jobject thiz,
                                                                                 jlong model_ptr, jstring tensor_name) {
  return GetTensorByInOutName(env, model_ptr, tensor_name, true);
}

extern "C" JNIEXPORT jobject JNICALL Java_com_mindspore_Model_getOutputs(JNIEnv *env, jobject thiz, jlong model_ptr) {
  return GetInOrOutTensors(env, thiz, model_ptr, false);
}

extern "C" JNIEXPORT jlong JNICALL Java_com_mindspore_Model_getOutputByTensorName(JNIEnv *env, jobject thiz,
                                                                                  jlong model_ptr,
                                                                                  jstring tensor_name) {
  return GetTensorByInOutName(env, model_ptr, tensor_name, false);
}

extern "C" JNIEXPORT jobject JNICALL Java_com_mindspore_Model_getOutputTensorNames(JNIEnv *env, jobject thiz,
                                                                                   jlong model_ptr) {
  jclass array_list = env->FindClass("java/util/ArrayList");
  jmethodID array_list_construct = env->GetMethodID(array_list, "<init>", "()V");
  jobject ret = env->NewObject(array_list, array_list_construct);
  jmethodID array_list_add = env->GetMethodID(array_list, "add", "(Ljava/lang/Object;)Z");

  auto *pointer = reinterpret_cast<void *>(model_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Session pointer from java is nullptr";
    return ret;
  }
  auto *lite_model_ptr = static_cast<mindspore::Model *>(pointer);
  auto output_names = lite_model_ptr->GetOutputTensorNames();
  for (const auto &output_name : output_names) {
    auto output_name_jstring = env->NewStringUTF(output_name.c_str());
    env->CallBooleanMethod(ret, array_list_add, output_name_jstring);
    env->DeleteLocalRef(output_name_jstring);
  }
  env->DeleteLocalRef(array_list);
  return ret;
}

extern "C" JNIEXPORT jobject JNICALL Java_com_mindspore_Model_getOutputsByNodeName(JNIEnv *env, jobject thiz,
                                                                                   jlong model_ptr, jstring node_name) {
  jclass array_list = env->FindClass("java/util/ArrayList");
  jmethodID array_list_construct = env->GetMethodID(array_list, "<init>", "()V");
  jobject ret = env->NewObject(array_list, array_list_construct);
  jmethodID array_list_add = env->GetMethodID(array_list, "add", "(Ljava/lang/Object;)Z");

  jclass long_object = env->FindClass("java/lang/Long");
  jmethodID long_object_construct = env->GetMethodID(long_object, "<init>", "(J)V");
  auto *pointer = reinterpret_cast<void *>(model_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Session pointer from java is nullptr";
    return ret;
  }
  auto *lite_model_ptr = static_cast<mindspore::Model *>(pointer);
  if (node_name == nullptr) {
    MS_LOG(ERROR) << "node_name from java is nullptr";
    return ret;
  }
  auto c_node_name = env->GetStringUTFChars(node_name, JNI_FALSE);
  auto tensors = lite_model_ptr->GetOutputsByNodeName(c_node_name);
  env->ReleaseStringUTFChars(node_name, c_node_name);
  for (auto &tensor : tensors) {
    auto tensor_ptr = std::make_unique<mindspore::MSTensor>(tensor);
    if (tensor_ptr == nullptr) {
      MS_LOG(ERROR) << "Make ms tensor failed";
      return ret;
    }
    jobject tensor_addr = env->NewObject(long_object, long_object_construct, jlong(tensor_ptr.release()));
    env->CallBooleanMethod(ret, array_list_add, tensor_addr);
    env->DeleteLocalRef(tensor_addr);
  }
  env->DeleteLocalRef(array_list);
  env->DeleteLocalRef(long_object);
  return ret;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_Model_getTrainMode(JNIEnv *env, jobject thiz,
                                                                            jlong model_ptr) {
  auto *pointer = reinterpret_cast<void *>(model_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Model pointer from java is nullptr";
    return (jboolean) false;
  }
  auto *lite_model_ptr = static_cast<mindspore::Model *>(pointer);
  return static_cast<jboolean>(lite_model_ptr->GetTrainMode());
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_Model_setTrainMode(JNIEnv *env, jobject thiz, jlong model_ptr,
                                                                            jboolean train_mode) {
  auto *pointer = reinterpret_cast<void *>(model_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Model pointer from java is nullptr";
    return jlong(false);
  }
  auto *lite_model_ptr = static_cast<mindspore::Model *>(pointer);
  auto status = lite_model_ptr->SetTrainMode(train_mode);
  return static_cast<jboolean>(status.IsOk());
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_Model_runStep(JNIEnv *env, jobject thiz, jlong model_ptr) {
  auto *pointer = reinterpret_cast<void *>(model_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Model pointer from java is nullptr";
    return (jboolean) false;
  }
  auto *lite_model_ptr = static_cast<mindspore::Model *>(pointer);
  auto status = lite_model_ptr->RunStep(nullptr, nullptr);
  return static_cast<jboolean>(status.IsOk());
}

std::vector<mindspore::MSTensor> convertArrayToVector(JNIEnv *env, jlongArray inputs) {
  std::vector<mindspore::MSTensor> c_inputs;
  if (inputs == nullptr) {
    MS_LOG(ERROR) << "inputs from java is nullptr";
    return c_inputs;
  }
  auto input_size = static_cast<int>(env->GetArrayLength(inputs));
  jlong *input_data = env->GetLongArrayElements(inputs, nullptr);
  for (int i = 0; i < input_size; i++) {
    auto *tensor_pointer = reinterpret_cast<void *>(input_data[i]);
    if (tensor_pointer == nullptr) {
      MS_LOG(ERROR) << "Tensor pointer from java is nullptr";
      env->ReleaseLongArrayElements(inputs, input_data, JNI_ABORT);
      return c_inputs;
    }
    auto *ms_tensor_ptr = static_cast<mindspore::MSTensor *>(tensor_pointer);
    c_inputs.push_back(*ms_tensor_ptr);
  }
  env->ReleaseLongArrayElements(inputs, input_data, JNI_ABORT);
  return c_inputs;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_Model_predict(JNIEnv *env, jobject thiz, jlong model_ptr,
                                                                       jlongArray inputs, jlongArray outputs) {
  auto *pointer = reinterpret_cast<void *>(model_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Model pointer from java is nullptr";
    return (jboolean) false;
  }
  auto *lite_model_ptr = static_cast<mindspore::Model *>(pointer);
  auto c_inputs = convertArrayToVector(env, inputs);
  auto c_outputs = convertArrayToVector(env, outputs);
  auto status = lite_model_ptr->Predict(c_inputs, &c_outputs);
  return static_cast<jboolean>(status.IsOk());
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_Model_resize(JNIEnv *env, jobject thiz, jlong model_ptr,
                                                                      jlongArray inputs, jobjectArray dims) {
  std::vector<std::vector<int64_t>> c_dims;
  auto *pointer = reinterpret_cast<void *>(model_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Model pointer from java is nullptr";
    return (jboolean) false;
  }
  auto *lite_model_ptr = static_cast<mindspore::Model *>(pointer);
  if (inputs == nullptr || dims == nullptr) {
    MS_LOG(ERROR) << "inputs or dims from java is nullptr";
    return (jboolean) false;
  }
  auto input_size = static_cast<int>(env->GetArrayLength(inputs));
  jlong *input_data = env->GetLongArrayElements(inputs, nullptr);
  if (input_data == nullptr) {
    MS_LOG(ERROR) << "input_data is nullptr";
    return (jboolean) false;
  }
  std::vector<mindspore::MSTensor> c_inputs;
  for (int i = 0; i < input_size; i++) {
    auto *tensor_pointer = reinterpret_cast<void *>(input_data[i]);
    if (tensor_pointer == nullptr) {
      MS_LOG(ERROR) << "Tensor pointer from java is nullptr";
      env->ReleaseLongArrayElements(inputs, input_data, JNI_ABORT);
      return (jboolean) false;
    }
    auto &ms_tensor = *static_cast<mindspore::MSTensor *>(tensor_pointer);
    c_inputs.push_back(ms_tensor);
  }
  auto tensor_size = static_cast<int>(env->GetArrayLength(dims));
  for (int i = 0; i < tensor_size; i++) {
    auto array = static_cast<jintArray>(env->GetObjectArrayElement(dims, i));
    if (array == nullptr) {
      MS_LOG(ERROR) << "Tensor pointer from java is nullptr";
      env->ReleaseLongArrayElements(inputs, input_data, JNI_ABORT);
      return (jboolean) false;
    }
    auto dim_size = static_cast<int>(env->GetArrayLength(array));
    jint *dim_data = env->GetIntArrayElements(array, nullptr);
    if (dim_data == nullptr) {
      MS_LOG(ERROR) << "dim_data is nullptr";
      env->ReleaseLongArrayElements(inputs, input_data, JNI_ABORT);
      env->DeleteLocalRef(array);
      return (jboolean) false;
    }
    std::vector<int64_t> tensor_dims(dim_size);
    for (int j = 0; j < dim_size; j++) {
      tensor_dims[j] = dim_data[j];
    }
    c_dims.push_back(tensor_dims);
    env->ReleaseIntArrayElements(array, dim_data, JNI_ABORT);
    env->DeleteLocalRef(array);
  }
  auto ret = lite_model_ptr->Resize(c_inputs, c_dims);
  env->ReleaseLongArrayElements(inputs, input_data, JNI_ABORT);
  return (jboolean)(ret.IsOk());
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_Model_loadConfig(JNIEnv *env, jobject thiz, jstring model_ptr,
                                                                          jstring config_path) {
  if (model_ptr == nullptr || config_path == nullptr) {
    MS_LOG(ERROR) << "input params from java is nullptr";
    return (jboolean) false;
  }
  auto *model_pointer = reinterpret_cast<void *>(model_ptr);
  auto *lite_model_ptr = static_cast<mindspore::Model *>(model_pointer);
  const char *c_config_path = env->GetStringUTFChars(config_path, nullptr);
  std::string str_config_path(c_config_path, env->GetStringLength(config_path));
  env->ReleaseStringUTFChars(config_path, c_config_path);
  auto ret = lite_model_ptr->LoadConfig(str_config_path);
  return (jboolean)(ret.IsOk());
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_Model_updateConfig(JNIEnv *env, jobject thiz,
                                                                            jstring model_ptr, jstring section,
                                                                            jobject hashMapConfig) {
  auto *model_pointer = reinterpret_cast<void *>(model_ptr);
  if (model_pointer == nullptr) {
    MS_LOG(ERROR) << "Model pointer from java is nullptr";
    return (jboolean) false;
  }
  auto *lite_model_ptr = static_cast<mindspore::Model *>(model_pointer);

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
  env->ReleaseStringUTFChars(section, c_section);
  for (auto &item : configInfo) {
    auto ret = lite_model_ptr->UpdateConfig(str_section, item);
    if (ret.IsError()) {
      return (jboolean) false;
    }
  }
  return (jboolean) true;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_Model_export(JNIEnv *env, jobject thiz, jlong model_ptr,
                                                                      jstring model_name, jint quantization_type,
                                                                      jboolean export_inference_only,
                                                                      jobjectArray tensorNames) {
  auto *model_pointer = reinterpret_cast<void *>(model_ptr);
  if (model_pointer == nullptr) {
    MS_LOG(ERROR) << "Model pointer from java is nullptr";
    return (jboolean) false;
  }
  auto *lite_model_ptr = static_cast<mindspore::Model *>(model_pointer);
  auto model_path = env->GetStringUTFChars(model_name, JNI_FALSE);
  std::vector<std::string> output_tensor_names;
  if (tensorNames != NULL) {
    auto tensor_size = static_cast<int>(env->GetArrayLength(tensorNames));
    for (int i = 0; i < tensor_size; i++) {
      auto tensor_name = static_cast<jstring>(env->GetObjectArrayElement(tensorNames, i));
      output_tensor_names.emplace_back(env->GetStringUTFChars(tensor_name, JNI_FALSE));
      env->DeleteLocalRef(tensor_name);
    }
  }
  mindspore::QuantizationType quant_type;
  if (quantization_type >= static_cast<int>(mindspore::kNoQuant) &&
      quantization_type <= static_cast<int>(mindspore::kFullQuant)) {
    quant_type = static_cast<mindspore::QuantizationType>(quantization_type);
  } else {
    MS_LOG(ERROR) << "Invalid quantization_type : " << quantization_type;
    return (jlong) nullptr;
  }
  auto ret = mindspore::Serialization::ExportModel(*lite_model_ptr, mindspore::kMindIR, model_path, quant_type,
                                                   export_inference_only, output_tensor_names);
  return (jboolean)(ret.IsOk());
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_Model_updateFeatureMaps(JNIEnv *env, jclass, jlong model_ptr,
                                                                                 jlongArray features) {
  auto size = static_cast<int>(env->GetArrayLength(features));
  jlong *input_data = env->GetLongArrayElements(features, nullptr);
  std::vector<mindspore::MSTensor> newFeatures;
  for (int i = 0; i < size; ++i) {
    auto *tensor_pointer = reinterpret_cast<void *>(input_data[i]);
    if (tensor_pointer == nullptr) {
      MS_LOG(ERROR) << "Tensor pointer from java is nullptr";
      return (jboolean) false;
    }
    auto *ms_tensor_ptr = static_cast<mindspore::MSTensor *>(tensor_pointer);
    newFeatures.emplace_back(*ms_tensor_ptr);
  }
  auto lite_model_ptr = reinterpret_cast<mindspore::Model *>(model_ptr);
  auto ret = lite_model_ptr->UpdateFeatureMaps(newFeatures);
  return (jboolean)(ret.IsOk());
}

extern "C" JNIEXPORT jobject JNICALL Java_com_mindspore_Model_getFeatureMaps(JNIEnv *env, jobject thiz,
                                                                             jlong model_ptr) {
  jclass array_list = env->FindClass("java/util/ArrayList");
  jmethodID array_list_construct = env->GetMethodID(array_list, "<init>", "()V");
  jobject ret = env->NewObject(array_list, array_list_construct);
  jmethodID array_list_add = env->GetMethodID(array_list, "add", "(Ljava/lang/Object;)Z");

  jclass long_object = env->FindClass("java/lang/Long");
  jmethodID long_object_construct = env->GetMethodID(long_object, "<init>", "(J)V");
  auto *pointer = reinterpret_cast<void *>(model_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Model pointer from java is nullptr";
    return ret;
  }
  auto *lite_model_ptr = static_cast<mindspore::Model *>(pointer);
  auto features = lite_model_ptr->GetFeatureMaps();
  for (auto &feature : features) {
    auto tensor_ptr = std::make_unique<mindspore::MSTensor>(feature);
    if (tensor_ptr == nullptr) {
      MS_LOG(ERROR) << "Make ms tensor failed";
      return ret;
    }
    jobject tensor_addr = env->NewObject(long_object, long_object_construct, jlong(tensor_ptr.release()));
    env->CallBooleanMethod(ret, array_list_add, tensor_addr);
  }
  return ret;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_Model_setLearningRate(JNIEnv *env, jclass, jlong model_ptr,
                                                                               jfloat learning_rate) {
  auto *pointer = reinterpret_cast<void *>(model_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Model pointer from java is nullptr";
    return (jboolean) false;
  }
  auto *lite_model_ptr = static_cast<mindspore::Model *>(pointer);
  auto ret = lite_model_ptr->SetLearningRate(learning_rate);
  return (jboolean)(ret.IsOk());
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_mindspore_Model_setupVirtualBatch(
  JNIEnv *env, jobject thiz, jlong model_ptr, jint virtual_batch_factor, jfloat learning_rate, jfloat momentum) {
  auto *pointer = reinterpret_cast<void *>(model_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Model pointer from java is nullptr";
    return (jboolean) false;
  }
  auto *lite_model_ptr = static_cast<mindspore::Model *>(pointer);
  auto ret = lite_model_ptr->SetupVirtualBatch(virtual_batch_factor, learning_rate, momentum);
  return (jboolean)(ret.IsOk());
}

extern "C" JNIEXPORT void JNICALL Java_com_mindspore_Model_free(JNIEnv *env, jobject thiz, jlong model_ptr) {
  auto *pointer = reinterpret_cast<void *>(model_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Model pointer from java is nullptr";
    return;
  }
  auto *lite_model_ptr = static_cast<mindspore::Model *>(pointer);
  delete (lite_model_ptr);
}
