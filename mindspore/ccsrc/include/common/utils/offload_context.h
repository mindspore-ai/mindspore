/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_OFFLOAD_CONTEXT_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_OFFLOAD_CONTEXT_H_

#include <string>
#include <memory>
#include "include/common/visible.h"

namespace mindspore {
class COMMON_EXPORT OffloadContext {
 public:
  static std::shared_ptr<OffloadContext> GetInstance();
  ~OffloadContext() = default;
  OffloadContext(const OffloadContext &) = delete;
  OffloadContext &operator=(const OffloadContext &) = delete;

  void set_enable_offload(bool enable_offload);
  bool enable_offload() const { return enable_offload_; }

  void set_offload_param(const std::string &offload_param);
  std::string offload_param() const { return offload_param_; }

  void set_offload_path(const std::string &offload_path);
  std::string offload_path() { return offload_path_; }

  void set_offload_checkpoint(const std::string &offload_checkpoint);
  std::string offload_checkpoint() const { return offload_checkpoint_; }

  void set_offload_ddr_size(size_t offload_ddr_size);
  size_t offload_ddr_size();

  void set_offload_disk_size(size_t offload_disk_size);
  size_t offload_disk_size() const { return offload_disk_size_; }

  void set_enable_aio(bool enable_aio);
  bool enable_aio() const { return enable_aio_; }

  void set_aio_block_size(size_t aio_block_size);
  size_t aio_block_size() const { return aio_block_size_; }

  void set_aio_queue_depth(size_t aio_queue_depth);
  size_t aio_queue_depth() const { return aio_queue_depth_; }

  void set_enable_pinned_mem(bool enable_pinned_mem);
  bool enable_pinned_mem() const { return enable_pinned_mem_; }

 private:
  OffloadContext()
      : enable_offload_(false),
        offload_param_(""),
        offload_path_("./offload"),
        offload_checkpoint_(""),
        offload_ddr_size_(0),
        offload_disk_size_(0),
        enable_aio_(true),
        aio_block_size_(0),
        aio_queue_depth_(0),
        enable_pinned_mem_(false) {}
  bool enable_offload_;
  std::string offload_param_;
  std::string offload_path_;
  std::string offload_checkpoint_;
  size_t offload_ddr_size_;
  size_t offload_disk_size_;
  bool enable_aio_;
  size_t aio_block_size_;
  size_t aio_queue_depth_;
  bool enable_pinned_mem_;
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_OFFLOAD_CONTEXT_H_
