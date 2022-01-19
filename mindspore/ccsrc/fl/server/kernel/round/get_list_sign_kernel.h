/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FL_SERVER_KERNEL_GET_LIST_SIGN_KERNEL_H
#define MINDSPORE_CCSRC_FL_SERVER_KERNEL_GET_LIST_SIGN_KERNEL_H

#include <string>
#include <vector>
#include <memory>
#include <map>
#include "fl/server/common.h"
#include "fl/server/kernel/round/round_kernel.h"
#include "fl/server/kernel/round/round_kernel_factory.h"
#include "fl/armour/cipher/cipher_init.h"
#include "fl/server/executor.h"
namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
// results of signature verification
enum sigVerifyResult { FAILED, TIMEOUT, PASSED };

class GetListSignKernel : public RoundKernel {
 public:
  GetListSignKernel() = default;
  ~GetListSignKernel() override = default;
  void InitKernel(size_t required_cnt) override;
  bool Launch(const uint8_t *req_data, size_t len, const std::shared_ptr<ps::core::MessageHandler> &message) override;
  bool Reset() override;
  void BuildGetListSignKernelRsp(const std::shared_ptr<server::FBBuilder> &fbb, const schema::ResponseCode retcode,
                                 const string &reason, const string &next_req_time, const size_t iteration,
                                 const std::map<std::string, std::vector<unsigned char>> &list_signs);

 private:
  armour::CipherInit *cipher_init_;
  Executor *executor_;
  size_t iteration_time_window_;
  sigVerifyResult VerifySignature(const schema::RequestAllClientListSign *client_list_sign_req);
  bool GetListSign(const size_t cur_iterator, const std::string &next_req_time,
                   const schema::RequestAllClientListSign *client_list_sign_req,
                   const std::shared_ptr<fl::server::FBBuilder> &fbb);
};
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_KERNEL_GET_LIST_SIGN_KERNEL_H
