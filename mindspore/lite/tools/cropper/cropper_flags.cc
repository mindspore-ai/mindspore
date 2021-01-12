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

#include "tools/cropper/cropper_flags.h"
#include <string>
#include "tools/cropper/cropper_utils.h"
#include "src/common/file_utils.h"

namespace mindspore {
namespace lite {
namespace cropper {
CropperFlags::CropperFlags() {
  AddFlag(&CropperFlags::package_file_, "packageFile", "The libmindspore-lite.a file that needs to be cropped", "");
  AddFlag(&CropperFlags::model_file_, "modelFile", "List of model files, separated by commas", "");
  AddFlag(&CropperFlags::model_folder_path_, "modelFolderPath", "Load all ms models in the folder", "");
  AddFlag(&CropperFlags::config_file_, "configFile", "The mapping configuration file path", "");
  AddFlag(&CropperFlags::output_file_, "outputFile", "Output library file path", "");
}

int CropperFlags::Init(int argc, const char **argv) {
  if (argc == 1) {
    std::cout << this->Usage() << std::endl;
    return RET_SUCCESS_EXIT;
  }
  Option<std::string> err = this->ParseFlags(argc, argv);

  if (err.IsSome()) {
    std::cerr << err.Get();
    std::cerr << this->Usage() << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }

  if (this->help) {
    std::cout << this->Usage() << std::endl;
    return RET_SUCCESS_EXIT;
  }

  MS_LOG(INFO) << "packageFile = " << this->package_file_;
  MS_LOG(INFO) << "modelFile = " << this->model_file_;
  MS_LOG(INFO) << "modelFolderPath = " << this->model_folder_path_;
  MS_LOG(INFO) << "configFile = " << this->config_file_;
  MS_LOG(INFO) << "outputFile = " << this->output_file_;

  if (this->package_file_.empty()) {
    std::cerr << "INPUT MISSING: packageFile is necessary" << std::endl;
    return RET_INPUT_PARAM_INVALID;
  } else {
    // Verify whether it is a static library file(.a)
    if (ValidFileSuffix(this->package_file_, "a") != RET_OK) {
      return RET_INPUT_PARAM_INVALID;
    }
    this->package_file_ = RealPath(this->package_file_.c_str());
    if (this->package_file_.empty()) {
      return RET_INPUT_PARAM_INVALID;
    }
  }

  if (this->model_file_.empty() && this->model_folder_path_.empty()) {
    std::cerr << "INPUT MISSING: modelFile or modelFolderPath is necessary" << std::endl;
    return RET_INPUT_PARAM_INVALID;
  } else if (!this->model_file_.empty() && !this->model_folder_path_.empty()) {
    std::cerr << "INPUT ILLEGAL: modelFile and modelFolderPath must choose one" << std::endl;
    return RET_INPUT_PARAM_INVALID;
  } else if (!this->model_folder_path_.empty()) {
    this->model_folder_path_ = RealPath(this->model_folder_path_.c_str());
    if (this->model_folder_path_.empty()) {
      return RET_INPUT_PARAM_INVALID;
    }
  }

  if (this->config_file_.empty()) {
    std::cerr << "INPUT MISSING: configFile is necessary" << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }
  this->config_file_ = RealPath(this->config_file_.c_str());
  if (this->config_file_.empty()) {
    return RET_INPUT_PARAM_INVALID;
  }

  if (this->output_file_.empty()) {
    this->output_file_ = this->package_file_;
  } else {
    if (ValidFileSuffix(this->output_file_, "a") != RET_OK) {
      MS_LOG(ERROR) << "INPUT ILLEGAL: packageFile need to pass package name, such as libmindspore-lite.a";
      return RET_INPUT_PARAM_INVALID;
    }
    std::string folder_name = this->output_file_.substr(0, this->output_file_.rfind('/'));
    folder_name = RealPath(folder_name.c_str());
    // folder does not exist.
    if (folder_name.empty()) {
      return RET_INPUT_PARAM_INVALID;
    }
  }
  if (this->output_file_.empty()) {
    return RET_INPUT_PARAM_INVALID;
  }
  return RET_OK;
}
}  // namespace cropper
}  // namespace lite
}  // namespace mindspore
