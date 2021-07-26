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

#ifndef MINDSPORE_CCSRC_UTILS_TENSORPRINT_UTILS_H_
#define MINDSPORE_CCSRC_UTILS_TENSORPRINT_UTILS_H_

#include <map>
#include <string>
#include "ir/dtype/type.h"
#ifndef NO_DLIB
#include "acl/acl_tdt.h"
#include "tdt/tsd_client.h"
#include "tdt/data_common.h"
#include "tdt/tdt_host_interface.h"
#include "proto/print.pb.h"

namespace mindspore {
class TensorPrint {
 public:
  explicit TensorPrint(const std::string &path, const acltdtChannelHandle *acl_handle)
      : print_file_path_(path), acl_handle_(acl_handle) {}
  ~TensorPrint() = default;
  void operator()();

 private:
  std::string print_file_path_;
  const acltdtChannelHandle *acl_handle_;
};
}  // namespace mindspore
#endif
#endif  // MINDSPORE_CCSRC_UTILS_TENSOR_PRINT_H_
