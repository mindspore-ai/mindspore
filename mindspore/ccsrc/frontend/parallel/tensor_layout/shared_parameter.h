/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_SHARED_PARAMETER_TENSOR_LAYOUT_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_SHARED_PARAMETER_TENSOR_LAYOUT_H_

#include <string>

namespace mindspore {
namespace parallel {
class SharedParameter {
 public:
  SharedParameter(bool pipeline_shared, bool is_send, int64_t peer_rank, int64_t sr_tag)
      : pipeline_shared_(pipeline_shared), is_send_(is_send), peer_rank_(peer_rank), sr_tag_(sr_tag) {}
  ~SharedParameter() = default;

  void set_pipeline_shared(bool pipeline_shared) { pipeline_shared_ = pipeline_shared; }
  bool pipeline_shared() const { return pipeline_shared_; }

  void set_is_send(bool is_send) { is_send_ = is_send; }
  bool is_send() const { return is_send_; }

  void set_peer_rank(int64_t peer_rank) { peer_rank_ = peer_rank; }
  int64_t peer_rank() const { return peer_rank_; }

  void set_sr_tag(int64_t sr_tag) { sr_tag_ = sr_tag; }
  int64_t sr_tag() const { return sr_tag_; }

  // Key for user data.
  constexpr static char key[] = "SharedParameter";

 private:
  bool pipeline_shared_ = false;
  bool is_send_ = false;
  int64_t peer_rank_{0};
  int64_t sr_tag_{0};
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_SHARED_PARAMETER_TENSOR_LAYOUT_H_
