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

#ifndef MINDSPORE_CCSRC_FL_WORKER_FL_CLOUD_WORKER_H_
#define MINDSPORE_CCSRC_FL_WORKER_FL_CLOUD_WORKER_H_

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include "proto/comm.pb.h"
#include "schema/fl_job_generated.h"
#include "schema/cipher_generated.h"
#include "fl/armour/secure_protocol/key_agreement.h"
#include "ps/ps_context.h"
#include "ps/core/node.h"
#include "ps/core/cluster_metadata.h"
#include "ps/core/communicator/http_client.h"
#include "include/backend/visible.h"

struct EncryptPublicKeys {
  std::string flID;
  std::vector<uint8_t> publicKey;
  std::vector<uint8_t> pwIV;
  std::vector<uint8_t> pwSalt;
};

namespace mindspore {
namespace fl {
using FBBuilder = flatbuffers::FlatBufferBuilder;
namespace worker {
// This class is used for hybrid training mode for now. In later version, parameter server mode will also use this class
// as worker.
class BACKEND_EXPORT FLCloudWorker : public ps::core::Node {
 public:
  using MessageReceive = std::function<void(const std::shared_ptr<std::vector<unsigned char>> &response_msg)>;
  static FLCloudWorker &GetInstance();
  bool Run();
  bool Start(const uint32_t &timeout) override { return true; }
  bool Stop() override { return true; }
  bool Finish(const uint32_t &timeout) override;
  bool SendToServerSync(const std::string path, const std::string content_type, const void *data, size_t data_size);

  void set_fl_iteration_num(uint64_t iteration_num);
  uint64_t fl_iteration_num() const;

  void set_data_size(int data_size);
  int data_size() const;

  void set_secret_pk(armour::PrivateKey *secret_pk);
  armour::PrivateKey *secret_pk() const;

  void set_pw_salt(const std::vector<uint8_t> pw_salt);
  std::vector<uint8_t> pw_salt() const;

  void set_pw_iv(const std::vector<uint8_t> pw_iv);
  std::vector<uint8_t> pw_iv() const;

  void set_public_keys_list(const std::vector<EncryptPublicKeys> public_keys_list);
  std::vector<EncryptPublicKeys> public_keys_list() const;

  std::string fl_name() const;
  std::string fl_id() const;

  void RegisterMessageCallback(const std::string kernel_path, const MessageReceive &cb);

 private:
  FLCloudWorker()
      : running_(false),
        fl_id_("worker0"),
        iteration_num_(0),
        data_size_(0),
        secret_pk_(nullptr),
        pw_salt_({}),
        pw_iv_({}),
        public_keys_list_({}),
        handlers_({}) {}

  ~FLCloudWorker() = default;
  FLCloudWorker(const FLCloudWorker &) = delete;
  FLCloudWorker &operator=(const FLCloudWorker &) = delete;
  std::atomic_bool running_;
  std::string fl_id_;

  // The federated learning iteration number.
  std::atomic<uint64_t> iteration_num_;

  // Data size for this federated learning job.
  int data_size_;

  // The worker standalone training step number before communicating with server. This used in hybrid training mode.
  uint64_t worker_step_num_per_iteration_;

  // The private key used for computing the pairwise encryption's secret.
  armour::PrivateKey *secret_pk_;

  // The salt value used for generate pairwise noise.
  std::vector<uint8_t> pw_salt_;

  // The initialization vector value used for generate pairwise noise.
  std::vector<uint8_t> pw_iv_;

  // The public keys used for computing the pairwise encryption's secret.
  std::vector<EncryptPublicKeys> public_keys_list_;

  std::string server_domain_;

  std::shared_ptr<ps::core::HttpClient> http_client_;

  std::unordered_map<std::string, MessageReceive> handlers_;
};
}  // namespace worker
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_WORKER_FL_CLOUD_WORKER_H_
