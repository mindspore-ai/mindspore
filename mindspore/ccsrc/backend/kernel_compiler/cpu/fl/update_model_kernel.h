/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FL_UPDATE_MODEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FL_UPDATE_MODEL_H_

#include <vector>
#include <string>
#include <memory>
#include <functional>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"
#include "fl/worker/fl_worker.h"
#include "fl/armour/secure_protocol/masking.h"

namespace mindspore {
namespace kernel {
constexpr int SECRET_MAX_LEN = 32;
class UpdateModelKernelMod : public NativeCpuKernelMod {
 public:
  UpdateModelKernelMod() = default;
  ~UpdateModelKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &, const std::vector<AddressPtr> &) {
    MS_LOG(INFO) << "Launching client UpdateModelKernelMod";
    if (inputs.size() != weight_full_names_.size()) {
      MS_LOG(EXCEPTION) << "Input number of UpdateModelKernelMod should be " << weight_full_names_.size()
                        << ", but got " << inputs.size();
      return false;
    }

    if (!WeightingData(inputs)) {
      MS_LOG(EXCEPTION) << "Weighting data with data_size failed.";
      return false;
    }

    if (encrypt_mode.compare("STABLE_PW_ENCRYPT") == 0) {
      EncryptData(inputs);
    }

    if (!BuildUpdateModelReq(fbb_, inputs)) {
      MS_LOG(EXCEPTION) << "Building request for FusedPushWeight failed.";
      return false;
    }

    std::shared_ptr<std::vector<unsigned char>> update_model_rsp_msg = nullptr;
    if (!fl::worker::FLWorker::GetInstance().SendToServer(target_server_rank_, fbb_->GetBufferPointer(),
                                                          fbb_->GetSize(), ps::core::TcpUserCommand::kUpdateModel,
                                                          &update_model_rsp_msg)) {
      MS_LOG(EXCEPTION) << "Sending request for UpdateModel to server " << target_server_rank_ << " failed.";
      return false;
    }
    flatbuffers::Verifier verifier(update_model_rsp_msg->data(), update_model_rsp_msg->size());
    if (!verifier.VerifyBuffer<schema::ResponseUpdateModel>()) {
      MS_LOG(EXCEPTION) << "The schema of ResponseUpdateModel is invalid.";
      return false;
    }

    const schema::ResponseFLJob *update_model_rsp =
      flatbuffers::GetRoot<schema::ResponseFLJob>(update_model_rsp_msg->data());
    MS_EXCEPTION_IF_NULL(update_model_rsp);
    auto response_code = update_model_rsp->retcode();
    switch (response_code) {
      case schema::ResponseCode_SUCCEED:
      case schema::ResponseCode_OutOfTime:
        break;
      default:
        MS_LOG(EXCEPTION) << "Launching start fl job for worker failed. Reason: " << update_model_rsp->reason();
    }
    return true;
  }

  void Init(const CNodePtr &kernel_node) {
    MS_LOG(INFO) << "Initializing UpdateModel kernel";
    fbb_ = std::make_shared<fl::FBBuilder>();
    MS_EXCEPTION_IF_NULL(fbb_);

    MS_EXCEPTION_IF_NULL(kernel_node);
    server_num_ = fl::worker::FLWorker::GetInstance().server_num();
    rank_id_ = fl::worker::FLWorker::GetInstance().rank_id();
    if (rank_id_ == UINT32_MAX) {
      MS_LOG(EXCEPTION) << "Federated worker is not initialized yet.";
      return;
    }
    target_server_rank_ = rank_id_ % server_num_;
    fl_name_ = fl::worker::FLWorker::GetInstance().fl_name();
    fl_id_ = fl::worker::FLWorker::GetInstance().fl_id();
    encrypt_mode = AnfAlgo::GetNodeAttr<string>(kernel_node, "encrypt_mode");
    if (encrypt_mode.compare("") != 0 && encrypt_mode.compare("STABLE_PW_ENCRYPT") != 0) {
      MS_LOG(EXCEPTION) << "Value Error: the parameter 'encrypt_mode' of updateModel kernel can only be '' or "
                           "'STABLE_PW_ENCRYPT' until now, but got: "
                        << encrypt_mode;
    }
    MS_LOG(INFO) << "Initializing StartFLJob kernel. fl_name: " << fl_name_ << ", fl_id: " << fl_id_
                 << ". Request will be sent to server " << target_server_rank_;
    if (encrypt_mode.compare("STABLE_PW_ENCRYPT") == 0) {
      MS_LOG(INFO) << "STABLE_PW_ENCRYPT mode is open, model weights will be encrypted before send to server.";
      client_keys = fl::worker::FLWorker::GetInstance().public_keys_list();
      if (client_keys.size() == 0) {
        MS_LOG(EXCEPTION) << "The size of local-stored client_keys_list is 0, please check whether P.ExchangeKeys() "
                             "and P.GetKeys() have been executed before updateModel.";
      }
    }

    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    for (size_t i = 0; i < input_num; i++) {
      auto input_node = AnfAlgo::VisitKernelWithReturnType(AnfAlgo::GetInputNode(kernel_node, i), 0).first;
      MS_EXCEPTION_IF_NULL(input_node);
      auto weight_node = input_node->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(weight_node);
      std::string weight_name = weight_node->fullname_with_scope();
      MS_LOG(INFO) << "Parameter name is " << weight_name;
      weight_full_names_.push_back(weight_name);

      auto weight_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, i);
      size_t weight_size_ =
        std::accumulate(weight_shape.begin(), weight_shape.end(), sizeof(float), std::multiplies<float>());
      input_size_list_.push_back(weight_size_);
    }
    output_size_list_.push_back(sizeof(float));
  }

  void InitKernel(const CNodePtr &kernel_node) { return; }

 protected:
  void InitSizeLists() { return; }

 private:
  bool BuildUpdateModelReq(const std::shared_ptr<fl::FBBuilder> &fbb, const std::vector<AddressPtr> &weights) {
    MS_EXCEPTION_IF_NULL(fbb_);
    auto fbs_fl_name = fbb->CreateString(fl_name_);
    auto fbs_fl_id = fbb->CreateString(fl_id_);
    std::vector<flatbuffers::Offset<schema::FeatureMap>> fbs_feature_maps;
    for (size_t i = 0; i < weight_full_names_.size(); i++) {
      const std::string &weight_name = weight_full_names_[i];
      auto fbs_weight_fullname = fbb->CreateString(weight_name);
      auto fbs_weight_data =
        fbb->CreateVector(reinterpret_cast<const float *>(weights[i]->addr), weights[i]->size / sizeof(float));
      auto fbs_feature_map = schema::CreateFeatureMap(*(fbb.get()), fbs_weight_fullname, fbs_weight_data);
      fbs_feature_maps.push_back(fbs_feature_map);
    }
    auto fbs_feature_maps_vector = fbb->CreateVector(fbs_feature_maps);

    schema::RequestUpdateModelBuilder req_update_model_builder(*(fbb.get()));
    req_update_model_builder.add_fl_name(fbs_fl_name);
    req_update_model_builder.add_fl_id(fbs_fl_id);
    iteration_ = fl::worker::FLWorker::GetInstance().fl_iteration_num();
    req_update_model_builder.add_iteration(SizeToInt(iteration_));
    req_update_model_builder.add_feature_map(fbs_feature_maps_vector);
    auto req_update_model = req_update_model_builder.Finish();
    fbb->Finish(req_update_model);
    return true;
  }

  bool WeightingData(const std::vector<AddressPtr> &inputs) {
    data_size_ = fl::worker::FLWorker::GetInstance().data_size();
    for (auto &input : inputs) {
      float *data = reinterpret_cast<float *>(input->addr);
      for (size_t i = 0; i < input->size / sizeof(float); i++) {
        data[i] *= data_size_;
      }
    }
    return true;
  }

  void EncryptData(const std::vector<AddressPtr> &inputs) {
    // calculate the sum of all layer's weight size
    size_t total_size = 0;
    for (size_t i = 0; i < weight_full_names_.size(); i++) {
      total_size += (inputs[i]->size / sizeof(float));
    }
    // get pairwise encryption noise vector
    std::vector<float> noise_vector = GetEncryptNoise(total_size);

    // encrypt original data
    size_t encrypt_num = 0;
    for (size_t i = 0; i < weight_full_names_.size(); i++) {
      const std::string &weight_name = weight_full_names_[i];
      MS_LOG(INFO) << "Encrypt weights of layer: " << weight_name;
      size_t weights_size = inputs[i]->size / sizeof(float);
      float *original_data = reinterpret_cast<float *>(inputs[i]->addr);
      for (size_t j = 0; j < weights_size; j++) {
        original_data[j] += noise_vector[j + encrypt_num];
      }
      encrypt_num += weights_size;
    }
    MS_LOG(INFO) << "Encrypt data finished.";
  }

  // compute the pairwise noise based on local worker's private key and remote workers' public key
  std::vector<float> GetEncryptNoise(size_t noise_len) {
    std::vector<float> total_noise(noise_len, 0);
    int client_num = client_keys.size();
    for (int i = 0; i < client_num; i++) {
      EncryptPublicKeys public_key_set_i = client_keys[i];
      std::string remote_fl_id = public_key_set_i.flID;
      // do not need to compute pairwise noise with itself
      if (remote_fl_id == fl_id_) {
        continue;
      }
      // get local worker's private key
      armour::PrivateKey *local_private_key = fl::worker::FLWorker::GetInstance().secret_pk();
      if (local_private_key == nullptr) {
        MS_LOG(EXCEPTION) << "Local secret private key is nullptr, get encryption noise failed!";
      }

      // choose pw_iv and pw_salt for encryption, we choose that of smaller fl_id worker's
      std::vector<uint8_t> encrypt_pw_iv;
      std::vector<uint8_t> encrypt_pw_salt;
      if (fl_id_ < remote_fl_id) {
        encrypt_pw_iv = fl::worker::FLWorker::GetInstance().pw_iv();
        encrypt_pw_salt = fl::worker::FLWorker::GetInstance().pw_salt();
      } else {
        encrypt_pw_iv = public_key_set_i.pwIV;
        encrypt_pw_salt = public_key_set_i.pwSalt;
      }

      // get keyAgreement seed
      std::vector<uint8_t> remote_public_key = public_key_set_i.publicKey;
      armour::PublicKey *pubKey =
        armour::KeyAgreement::FromPublicBytes(remote_public_key.data(), remote_public_key.size());
      uint8_t secret1[SECRET_MAX_LEN] = {0};
      int ret = armour::KeyAgreement::ComputeSharedKey(
        local_private_key, pubKey, SECRET_MAX_LEN, encrypt_pw_salt.data(), SizeToInt(encrypt_pw_salt.size()), secret1);
      delete pubKey;
      if (ret < 0) {
        MS_LOG(EXCEPTION) << "Get secret seed failed!";
      }

      // generate pairwise encryption noise
      std::vector<float> noise_i;
      if (armour::Masking::GetMasking(&noise_i, noise_len, (const uint8_t *)secret1, SECRET_MAX_LEN,
                                      encrypt_pw_iv.data(), encrypt_pw_iv.size()) < 0) {
        MS_LOG(EXCEPTION) << "Get masking noise failed.";
      }
      int noise_sign = (fl_id_ < remote_fl_id) ? -1 : 1;
      for (size_t k = 0; k < noise_len; k++) {
        total_noise[k] += noise_sign * noise_i[k];
      }
      MS_LOG(INFO) << "Generate noise between fl_id: " << fl_id_ << " and fl_id: " << remote_fl_id << " finished.";
    }
    return total_noise;
  }

  std::shared_ptr<fl::FBBuilder> fbb_;
  uint32_t rank_id_;
  uint32_t server_num_;
  uint32_t target_server_rank_;
  std::string fl_name_;
  std::string fl_id_;
  int data_size_;
  uint64_t iteration_;
  std::vector<std::string> weight_full_names_;
  std::string encrypt_mode;
  std::vector<EncryptPublicKeys> client_keys;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FL_UPDATE_MODEL_H_
