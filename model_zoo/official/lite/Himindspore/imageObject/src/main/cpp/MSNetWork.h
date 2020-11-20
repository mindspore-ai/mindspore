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

#ifndef MSNETWORK_H
#define MSNETWORK_H

#include <context.h>
#include <lite_session.h>
#include <model.h>
#include <errorcode.h>
#include <cstdio>
#include <algorithm>
#include <fstream>
#include <functional>
#include <sstream>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <utility>

struct ImgDims {
  int channel = 0;
  int width = 0;
  int height = 0;
};

/*struct SessIterm {
    std::shared_ptr<mindspore::session::LiteSession> sess = nullptr;
};*/

class MSNetWork {
 public:
  MSNetWork();

  ~MSNetWork();

  void CreateSessionMS(char *modelBuffer, size_t bufferLen, mindspore::lite::Context *ctx);

  void ReleaseNets(void);

  mindspore::session::LiteSession *session() const { return session_; }

 private:
  mindspore::session::LiteSession *session_;
  mindspore::lite::Model *model_;
};

#endif
