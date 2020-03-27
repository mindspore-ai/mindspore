/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "include/session.h"
#include <map>
#include <atomic>
#include "include/errorcode.h"
#include "common/mslog.h"
#include "src/graph.h"
#include "src/graph_execution.h"

namespace mindspore {
namespace predict {
Context m_ctx;
bool m_isConfig = false;

// In 32bits, this evaluates to 2GB - 1
static constexpr auto MAX_BUFFER_SIZE = ((1ULL << (sizeof(int32_t) * 8 - 1)) - 1);

std::shared_ptr<Session> CreateSession(const char *graphBuf, size_t size, const Context &ctx) {
  if (graphBuf == nullptr) {
    MS_LOGE("the graphBuf is nullptr");
    return nullptr;
  }
  if (size > MAX_BUFFER_SIZE) {
    MS_LOGE("the size is invalid");
    return nullptr;
  }
  auto session = std::make_shared<Session>(ctx);
  MS_ASSERT(session != nullptr);
  auto ret = session->Init(graphBuf, size);
  if (ret != RET_OK) {
    MS_LOGE("Init session failed.");
    return nullptr;
  }
  return session;
}
Session::Session(const Context &ctx) : _ctx(ctx) {
  Context cfgCtx;
  cfgCtx = ctx;
  if (cfgCtx.threadNum > m_ctx.threadNum) {
    cfgCtx.threadNum = m_ctx.threadNum;
  }
}

int Session::Init(const char *graphBuf, size_t size) {
  _graph = Graph::CreateFromBuf(graphBuf, size, _ctx);
  if (_graph == nullptr) {
    MS_LOGE("Graph create from buf failed.");
    return RET_NULL_PTR;
  }

  auto ret = this->InitExecutor();
  if (ret != RET_OK) {
    MS_LOGE("Init Executor failed");
    return ret;
  }
  return ret;
}

int Session::InitExecutor() {
  if (_executor != nullptr) {
    delete _executor;
    _executor = nullptr;
  }
  if (_graph != nullptr) {
    _executor = new (std::nothrow) GraphExecution(_ctx, _graph);
    if (_executor == nullptr) {
      MS_LOGE("new GraphExecution fail");
      return RET_ERROR;
    }
    return RET_OK;
  } else {
    MS_LOGE("the graph is nullptr");
    return RET_ERROR;
  }
}

Session::~Session() {
  if (_executor != nullptr) {
    delete _executor;
  }
  if (_graph != nullptr) {
    delete _graph;
  }
}

int Session::Run(const std::vector<Tensor *> &inputs) {
  auto ret = RET_OK;
  if (reinitExecutor) {
    ret = this->InitExecutor();
    if (ret != RET_OK) {
      MS_LOGE("Init Executor failed");
      return ret;
    }
  }
  if (_executor == nullptr) {
    MS_LOGE("_executor is nullptr");
    return ret;
  }
  ret = _executor->Run(inputs);
  return ret;
}

std::vector<Tensor *> Session::GetInput() {
  if (_executor == nullptr) {
    MS_LOGE("_executor is nullptr");
    return std::vector<Tensor *>{};
  }
  auto inputs = _executor->GetInput();
  if (inputs.empty()) {
    MS_LOGI("output is empty.");
  }
  return inputs;
}

std::vector<Tensor *> Session::GetOutput(const std::string &nodeName) {
  if (_executor == nullptr) {
    MS_LOGE("graph's executor is nullptr.");
    return std::vector<Tensor *>{};
  }
  auto outputs = _executor->GetOutput(nodeName);
  if (outputs.empty()) {
    MS_LOGI("output is empty.");
  }
  return outputs;
}

std::map<std::string, std::vector<Tensor *>> Session::GetAllOutput() {
  if (_executor == nullptr) {
    MS_LOGE("graph's executor is nullptr.");
    return std::map<std::string, std::vector<Tensor *>>{};
  }
  auto outputs = _executor->GetAllOutput();
  if (outputs.empty()) {
    MS_LOGI("outputs is empty.");
  }
  return outputs;
}
}  // namespace predict
}  // namespace mindspore
