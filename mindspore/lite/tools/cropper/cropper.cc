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

#include <string>
#include "tools/cropper/cropper.h"
#include "tools/cropper/cropper_utils.h"
#define BUF_SIZE 1024

namespace mindspore {
namespace lite {
namespace cropper {
static const char *DELIM_COMMA = ",";

int Cropper::ReadPackage() {
  std::ifstream in_file(this->flags_->package_file_);
  if (ValidFile(in_file, this->flags_->package_file_.c_str()) == RET_OK) {
    in_file.close();

    char buf[BUF_SIZE];
    std::string cmd = "ar -t " + this->flags_->package_file_;
    MS_LOG(DEBUG) << cmd;

    FILE *p_file = popen(cmd.c_str(), "r");
    if (p_file == nullptr) {
      MS_LOG(ERROR) << "Error to popen" << this->flags_->package_file_;
      return RET_ERROR;
    }
    while (fgets(buf, BUF_SIZE, p_file) != nullptr) {
      this->all_files_.push_back(std::string(buf).substr(0, std::string(buf).length() - 1));
      this->discard_files_.push_back(std::string(buf).substr(0, std::string(buf).length() - 1));
    }
    pclose(p_file);
    MS_LOG(DEBUG) << "file nums: " << this->all_files_.size();
  } else {
    return RET_ERROR;
  }
  return RET_OK;
}

int Cropper::RunCropper() {
  int status;
  status = ReadPackage();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "read package failed.";
    return status;
  }
  status = GetModelFiles();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "get model files failed.";
    return status;
  }
  status = GetModelOps();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "get model ops failed.";
    return status;
  }
  status = GetOpMatchFiles();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "get op match files failed.";
    return status;
  }
  status = GetDiscardFileList();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "get discard file list failed.";
    return status;
  }
  status = CutPackage();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "cut package failed.";
    return status;
  }
  return RET_OK;
}

int Cropper::GetModelOps() {
  for (const auto &path : this->model_files_) {
    size_t buffer_lens;
    char *graph_buf = ReadFile(path.c_str(), &buffer_lens);
    if (graph_buf == nullptr) {
      MS_LOG(ERROR) << "Read model file failed while running " << path.c_str();
      std::cerr << "Read model file failed while running " << path.c_str() << std::endl;
      return RET_ERROR;
    }
    auto meta_graph = schema::GetMetaGraph(graph_buf);
    if (meta_graph == nullptr) {
      delete[] graph_buf;
      MS_LOG(ERROR) << "meta_graph is nullptr.";
      std::cerr << "meta_graph is nullptr!" << std::endl;
      return RET_ERROR;
    }
    auto nodes = meta_graph->nodes();
    for (auto node : *nodes) {
      if (node->primitive() == nullptr) {
        delete[] graph_buf;
        MS_LOG(ERROR) << "node primitive is nullptr.";
        std::cerr << "node primitive is nullptr." << std::endl;
        return RET_ERROR;
      }
      this->all_operators_.insert(node->primitive()->value_type());
      MS_LOG(DEBUG) << "PrimitiveType:" << schema::EnumNamePrimitiveType(node->primitive()->value_type())
                    << " QuantType:" << schema::EnumNameQuantType(node->quantType());
      // QuantType_AwareTraining may change
      if (node->quantType() == schema::QuantType_AwareTraining || node->quantType() == schema::QuantType_PostTraining) {
        this->int8_operators_.insert(node->primitive()->value_type());
      } else {
        this->fp32_operators_.insert(node->primitive()->value_type());
      }
    }
    delete[] graph_buf;
  }
  return RET_OK;
}

int Cropper::GetModelFiles() {
  if (!this->flags_->model_file_.empty()) {
    auto files = StringSplit(this->flags_->model_file_, std::string(DELIM_COMMA));
    for (const auto &file : files) {
      if (ValidFileSuffix(file, "ms") != RET_OK) {
        return RET_INPUT_PARAM_INVALID;
      }
      MS_LOG(DEBUG) << file;
      std::string real_path = RealPath(file.c_str());
      if (real_path.empty()) {
        return RET_INPUT_PARAM_INVALID;
      }
      this->model_files_.push_back(real_path);
    }
  }
  // get models from folder
  if (!this->flags_->model_folder_path_.empty()) {
    std::string cmd = "find " + this->flags_->model_folder_path_ + " -name '*.ms'";
    MS_LOG(DEBUG) << cmd;

    char buf[BUF_SIZE];
    FILE *p_file = popen(cmd.c_str(), "r");
    if (p_file == nullptr) {
      MS_LOG(ERROR) << "Error to popen";
      return RET_ERROR;
    }
    while (fgets(buf, BUF_SIZE, p_file) != nullptr) {
      std::string real_path = RealPath(std::string(buf).substr(0, std::string(buf).length() - 1).c_str());
      if (real_path.empty()) {
        pclose(p_file);
        return RET_INPUT_PARAM_INVALID;
      }
      this->model_files_.emplace_back(real_path);
    }
    pclose(p_file);
  }
  if (this->model_files_.empty()) {
    MS_LOG(ERROR) << "model file does not exist.";
    return RET_ERROR;
  }
  return RET_OK;
}

int Cropper::GetOpMatchFiles() {
  std::ifstream in_file(this->flags_->config_file_);
  if (ValidFile(in_file, this->flags_->config_file_.c_str()) == RET_OK) {
    MS_LOG(DEBUG) << this->flags_->config_file_.c_str();
    char buf[BUF_SIZE];
    while (!in_file.eof()) {
      in_file.getline(buf, BUF_SIZE);
      std::string buf_str = buf;
      auto mapping = StringSplit(buf_str, DELIM_COMMA);
      if (!mapping.empty()) {
        std::string primitive = mapping.at(0);
        std::string type = mapping.at(1);
        std::string file = mapping.at(2);
        if (type == "kNumberTypeFloat32" || type == "kNumberTypeFloat16" || type == "kNumberTypeInt32") {
          for (auto op : this->fp32_operators_) {
            if (schema::EnumNamePrimitiveType(op) == primitive) {
              MS_LOG(DEBUG) << "kNumberTypeFloat32:" << mapping[2];
              this->archive_files_.insert(mapping[2]);
              break;
            }
          }
        } else if (type == "kNumberTypeInt8") {
          for (auto op : this->int8_operators_) {
            if (schema::EnumNamePrimitiveType(op) == primitive) {
              MS_LOG(DEBUG) << "int8_operators_:" << mapping[2];
              this->archive_files_.insert(mapping[2]);
              break;
            }
          }
        } else if (type == "prototype") {
          for (auto op : this->all_operators_) {
            if (schema::EnumNamePrimitiveType(op) == primitive) {
              MS_LOG(DEBUG) << "prototype:" << mapping[2];
              this->archive_files_.insert(mapping[2]);
              break;
            }
          }
        } else if (type == "common") {
          MS_LOG(DEBUG) << "common:" << mapping[2];
          this->archive_files_.insert(mapping[2]);
        } else {
          MS_LOG(ERROR) << "invalid type symbol:" << type;
          return RET_ERROR;
        }
      }
    }
    in_file.close();
  } else {
    return RET_ERROR;
  }
  return RET_OK;
}

int Cropper::GetDiscardFileList() {
  // discard_files_=all_files_-archive_files_
  for (const auto &archive : this->archive_files_) {
    for (auto it = this->discard_files_.begin(); it != this->discard_files_.end();) {
      if (*it == archive) {
        it = this->discard_files_.erase(it);
      } else {
        it++;
      }
    }
  }
  return RET_OK;
}
int Cropper::CutPackage() {
  std::string copy_bak_cmd = "cp " + this->flags_->package_file_ + " " + this->flags_->package_file_ + ".bak";
  std::string ar_cmd = "ar -d " + this->flags_->package_file_ + ".bak ";
  for (const auto &file : this->discard_files_) {
    ar_cmd.append(file).append(" ");
  }
  std::string copy_to_output_cmd = "cp " + this->flags_->package_file_ + ".bak " + this->flags_->output_file_;
  std::string rm_bak_cmd = "rm " + this->flags_->package_file_ + ".bak";
  int status;
  status = system(copy_bak_cmd.c_str());
  if (status != 0) {
    MS_LOG(ERROR) << copy_bak_cmd << " executor failed.";
    return RET_ERROR;
  }
  status = system(ar_cmd.c_str());
  if (status != 0) {
    MS_LOG(ERROR) << ar_cmd << " executor failed.";
    status = system(rm_bak_cmd.c_str());
    // delete bak file.
    if (status != 0) {
      MS_LOG(ERROR) << rm_bak_cmd << " executor failed.";
    }
    return RET_ERROR;
  }
  status = system(copy_to_output_cmd.c_str());
  if (status != 0) {
    MS_LOG(ERROR) << copy_to_output_cmd << " executor failed.";
    // delete bak file.
    status = system(rm_bak_cmd.c_str());
    if (status != 0) {
      MS_LOG(ERROR) << rm_bak_cmd << " executor failed.";
    }
    return RET_ERROR;
  }
  // delete bak file.
  status = system(rm_bak_cmd.c_str());
  if (status != 0) {
    MS_LOG(ERROR) << rm_bak_cmd << " executor failed.";
    return RET_ERROR;
  }
  MS_LOG(INFO) << "Save package file " << this->flags_->output_file_ << " success.";
  std::cout << "Save package file" << this->flags_->output_file_ << " success." << std::endl;
  return RET_OK;
}

int RunCropper(int argc, const char **argv) {
  CropperFlags flags;
  int status;
  status = flags.Init(argc, argv);
  if (status == RET_SUCCESS_EXIT) {
    return status;
  }
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Flags init Error:" << status << " " << GetErrorInfo(status);
    std::cerr << "Flags init Error:" << status << " " << GetErrorInfo(status) << std::endl;
    return status;
  }
  Cropper cropper(&flags);

  status = cropper.RunCropper();
  if (status == RET_OK) {
    MS_LOG(INFO) << "CROPPER RESULT SUCCESS:" << status;
    std::cout << "CROPPER RESULT SUCCESS:" << status << std::endl;
  } else {
    MS_LOG(ERROR) << "CROPPER RESULT FAILED:" << status << " " << GetErrorInfo(status);
    std::cerr << "CROPPER RESULT FAILED:" << status << " " << GetErrorInfo(status) << std::endl;
  }
  return status;
}
}  // namespace cropper
}  // namespace lite
}  // namespace mindspore
