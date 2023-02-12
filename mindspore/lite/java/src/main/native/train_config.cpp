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
#include "include/api/types.h"
#include "include/api/cfg.h"

constexpr const int kOLKeepBatchNorm = 2;
constexpr const int kOLNotKeepBatchNorm = 3;
constexpr const int kOLAuto = 4;
extern "C" JNIEXPORT jlong JNICALL Java_com_mindspore_config_TrainCfg_createTrainCfg(JNIEnv *env, jobject thiz,
                                                                                     jstring loss_name,
                                                                                     jint optimizationLevel,
                                                                                     jboolean accmulateGrads) {
  mindspore::OptimizationLevel ol;
  switch (optimizationLevel) {
    case 0:
      ol = mindspore::OptimizationLevel::kO0;
      break;
    case kOLKeepBatchNorm:
      ol = mindspore::OptimizationLevel::kO2;
      break;
    case kOLNotKeepBatchNorm:
      ol = mindspore::OptimizationLevel::kO3;
      break;
    case kOLAuto:
      ol = mindspore::OptimizationLevel::kAuto;
      break;
    default:
      MS_LOG(ERROR) << "Invalid optimization_type : " << optimizationLevel;
      return (jlong) nullptr;
  }
  auto *traincfg_ptr = new (std::nothrow) mindspore::TrainCfg();
  if (traincfg_ptr == nullptr) {
    MS_LOG(ERROR) << "new train config fail!";
    return (jlong) nullptr;
  }
  if (loss_name != nullptr) {
    std::vector<std::string> traincfg_loss_name = traincfg_ptr->GetLossName();
    traincfg_loss_name.emplace_back(env->GetStringUTFChars(loss_name, JNI_FALSE));
    traincfg_ptr->SetLossName(traincfg_loss_name);
  }
  traincfg_ptr->optimization_level_ = ol;
  traincfg_ptr->accumulate_gradients_ = accmulateGrads;
  return (jlong)traincfg_ptr;
}

extern "C" JNIEXPORT jlong JNICALL Java_com_mindspore_config_TrainCfg_addMixPrecisionCfg(JNIEnv *env, jobject thiz,
                                                                                         jlong train_cfg_ptr,
                                                                                         jboolean dynamic_loss_scale,
                                                                                         jfloat loss_scale,
                                                                                         jint thresholdIterNum) {
  mindspore::MixPrecisionCfg mix_precision_cfg;
  mix_precision_cfg.dynamic_loss_scale_ = dynamic_loss_scale;
  mix_precision_cfg.loss_scale_ = loss_scale;
  mix_precision_cfg.num_of_not_nan_iter_th_ = thresholdIterNum;
  auto *pointer = reinterpret_cast<void *>(train_cfg_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Context pointer from java is nullptr";
    return jboolean(false);
  }
  auto *c_train_cfg_ptr = static_cast<mindspore::TrainCfg *>(pointer);
  c_train_cfg_ptr->mix_precision_cfg_ = mix_precision_cfg;
  return jboolean(true);
}

extern "C" JNIEXPORT void JNICALL Java_com_mindspore_config_TrainCfg_free(JNIEnv *env, jobject thiz,
                                                                          jlong train_cfg_ptr) {
  auto *pointer = reinterpret_cast<void *>(train_cfg_ptr);
  if (pointer == nullptr) {
    MS_LOG(ERROR) << "Context pointer from java is nullptr";
    return;
  }
  auto *c_cfg_ptr = static_cast<mindspore::TrainCfg *>(pointer);
  delete (c_cfg_ptr);
}
