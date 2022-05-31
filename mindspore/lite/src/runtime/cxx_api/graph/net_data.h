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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_CXX_API_GRAPH_NET_DATA_H_
#define MINDSPORE_LITE_SRC_RUNTIME_CXX_API_GRAPH_NET_DATA_H_

#include <memory>
#include "include/api/net.h"

namespace mindspore {
class NetData {
 public:
  explicit NetData(const std::shared_ptr<Net> &net) : net_(net) {}
  virtual ~NetData();
  void set_net(std::shared_ptr<Net> net) { net_ = net; }
  std::shared_ptr<Net> &net() { return net_; }

 private:
  std::shared_ptr<Net> net_;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_RUNTIME_CXX_API_GRAPH_NET_DATA_H_
