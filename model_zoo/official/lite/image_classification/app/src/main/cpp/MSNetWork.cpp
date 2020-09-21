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

#include "MSNetWork.h"
#include <android/log.h>
#include <iostream>
#include <string>
#include "include/errorcode.h"

#define MS_PRINT(format, ...) __android_log_print(ANDROID_LOG_INFO, "MSJNI", format, ##__VA_ARGS__)

MSNetWork::MSNetWork(void) : session_(nullptr) {}

MSNetWork::~MSNetWork(void) {}

void MSNetWork::CreateSessionMS(char *modelBuffer, size_t bufferLen, mindspore::lite::Context *ctx) {
  session_ = mindspore::session::LiteSession::CreateSession(ctx);
  if (session_ == nullptr) {
    MS_PRINT("Create Session failed.");
    return;
  }

  // Compile model.
  auto model = mindspore::lite::Model::Import(modelBuffer, bufferLen);
  if (model == nullptr) {
    MS_PRINT("Import model failed.");
    return;
  }

  int ret = session_->CompileGraph(model);
  if (ret != mindspore::lite::RET_OK) {
    MS_PRINT("CompileGraph failed.");
    return;
  }
}

int MSNetWork::ReleaseNets(void) {
  delete session_;
  return 0;
}
