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
#include "ir/dtype/type.h"
#ifndef NO_DLIB
#include "acl/acl_tdt.h"
#include "tdt/tsd_client.h"
#include "tdt/data_common.h"
#include "tdt/tdt_host_interface.h"
#include "proto/print.pb.h"
#include "utils/ms_context.h"
#endif
namespace mindspore {
class TensorPrint {
 public:
  TensorPrint() {}
  ~TensorPrint() = default;
#ifndef NO_DLIB
  explicit TensorPrint(acltdtChannelHandle *acl_handle) { acl_handle_ = acl_handle; }
  void operator()();

 private:
  acltdtChannelHandle *acl_handle_ = nullptr;
#endif
};
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_UTILS_TENSOR_PRINT_H_
