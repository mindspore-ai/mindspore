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

#ifndef MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_HTTP_CLIENT_H_
#define MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_HTTP_CLIENT_H_

#include <event2/event.h>
#include <event2/bufferevent.h>
#include <event2/thread.h>
#include <event2/bufferevent_ssl.h>
#include <event2/buffer.h>
#include <event2/http.h>
#include <event2/http_struct.h>
#include <event2/keyvalq_struct.h>

#include <functional>
#include <string>
#include <memory>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>

#include "ps/core/cluster_config.h"
#include "utils/convert_utils_base.h"
#include "ps/core/comm_util.h"
#include "ps/core/communicator/ssl_client.h"
#include "ps/constants.h"
#include "ps/ps_context.h"
#include "ps/core/file_configuration.h"

namespace mindspore {
namespace ps {
namespace core {
#define HTTP_CONTENT_TYPE_URL_ENCODED "application/x-www-form-urlencoded"
#define HTTP_CONTENT_TYPE_FORM_DATA "multipart/form-data"
#define HTTP_CONTENT_TYPE_TEXT_PLAIN "text/plain"

class HttpClient {
 public:
  using OnConnected = std::function<void()>;
  using OnDisconnected = std::function<void()>;
  using OnTimeout = std::function<void()>;
  using OnMessage = std::function<void(const size_t request_id, const std::string &kernel_path,
                                       const std::shared_ptr<std::vector<unsigned char>> &response_data)>;
  using OnTimer = std::function<void()>;

  explicit HttpClient(const std::string &server_domain);
  virtual ~HttpClient();

  void set_disconnected_callback(const OnDisconnected &disconnected);
  void set_connected_callback(const OnConnected &connected);
  bool WaitConnected(
    const uint32_t &connected_timeout = PSContext::instance()->cluster_config().cluster_available_timeout);
  void Init();
  bool Stop();
  void SetMessageCallback(const OnMessage &cb);
  bool SendMessage(const std::string &kernel_path, const std::string &content_type, const void *data, size_t data_size,
                   const size_t request_id);
  event_base *get_event_base() const;
  bool BreakLoopEvent();
  void set_request_id(size_t request_id);
  size_t request_id() const;
  void set_kernel_path(const std::string kernel_path);
  std::string kernel_path() const;

 protected:
  static void ReadCallback(struct evhttp_request *http_req, void *message_callback);
  bool EstablishSSL();
  void OnReadHandler(const size_t request_id, const std::string kernel_name,
                     const std::shared_ptr<std::vector<unsigned char>> &response_msg);
  std::string PeerRoleName() const;

 private:
  OnMessage message_callback_;

  OnConnected connected_callback_;
  OnDisconnected disconnected_callback_;
  OnTimeout timeout_callback_;
  OnTimer on_timer_callback_;

  std::string server_domain_;
  size_t request_id_;
  std::string kernel_path_;
  event_base *event_base_;
  bufferevent *buffer_event_;

  std::mutex connection_mutex_;
  std::condition_variable connection_cond_;

  std::uint16_t server_port_;
  evhttp_request *http_req_;
  evhttp_connection *evhttp_conn_;
  evhttp_uri *uri;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_HTTP_CLIENT_H_
