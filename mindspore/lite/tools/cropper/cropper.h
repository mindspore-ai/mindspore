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

#ifndef MINDSPORE_LITE_TOOLS_CROPPER_CROPPER_H_
#define MINDSPORE_LITE_TOOLS_CROPPER_CROPPER_H_

#include <set>
#include <string>
#include <vector>
#include "tools/common/flag_parser.h"
#include "src/common/file_utils.h"
#include "src/common/utils.h"
#include "schema/model_generated.h"
#include "tools/cropper/cropper_flags.h"

namespace mindspore::lite::cropper {

class Cropper {
 public:
  explicit Cropper(CropperFlags *flags) : flags_(flags) {}

  ~Cropper() = default;

  int RunCropper();

  int ReadPackage();

  int GetModelFiles();

  int GetModelOps();

  int GetOpMatchFiles();

  int GetDiscardFileList();

  int CutPackage();

  std::vector<std::string> model_files_;
  std::vector<std::string> all_files_;
  std::set<std::string> archive_files_;
  std::vector<std::string> discard_files_;

  std::set<schema::PrimitiveType> all_operators_;
  std::set<schema::PrimitiveType> int8_operators_;
  std::set<schema::PrimitiveType> fp32_operators_;

 private:
  CropperFlags *flags_;
};

int RunCropper(int argc, const char **argv);
}  // namespace mindspore::lite::cropper
#endif  // MINDSPORE_LITE_TOOLS_CROPPER_CROPPER_H_
