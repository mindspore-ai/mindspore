/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/util/json_helper.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <nlohmann/json.hpp>

#include "minddata/dataset/util/log_adapter.h"
#include "minddata/dataset/util/path.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
// Create a numbered json file from image folder
Status JsonHelper::CreateAlbum(const std::string &in_dir, const std::string &out_dir) {
  // in check
  Path base_dir = Path(in_dir);
  if (!base_dir.IsDirectory() || !base_dir.Exists()) {
    RETURN_STATUS_UNEXPECTED("Input dir is not a directory or doesn't exist");
  }
  // check if output_dir exists and create it if it does not exist
  Path target_dir = Path(out_dir);
  RETURN_IF_NOT_OK(target_dir.CreateDirectory());

  // iterate over in dir and create json for all images
  uint64_t index = 0;
  auto dir_it = Path::DirIterator::OpenDirectory(&base_dir);
  while (dir_it->hasNext()) {
    Path v = dir_it->next();
    // check if found file fits image extension

    // create json file in output dir with the path
    std::string out_file = out_dir + "/" + std::to_string(index) + ".json";
    UpdateValue(out_file, "image", v.toString(), out_file);
    index++;
  }
  return Status::OK();
}

// A print method typically used for debugging
void JsonHelper::Print(std::ostream &out) const {
  out << "  Data Helper"
      << "\n";
}

Status JsonHelper::UpdateArray(const std::string &in_file, const std::string &key,
                               const std::vector<std::string> &value, const std::string &out_file) {
  try {
    Path in = Path(in_file);
    nlohmann::json js;
    if (in.Exists()) {
      std::ifstream in_stream(in_file);
      MS_LOG(INFO) << "Filename: " << in_file << ".";
      in_stream >> js;
      in_stream.close();
    }
    js[key] = value;
    MS_LOG(INFO) << "Write outfile is: " << js << ".";
    if (out_file == "") {
      std::ofstream o(in_file, std::ofstream::trunc);
      o << js;
      o.close();
    } else {
      std::ofstream o(out_file, std::ofstream::trunc);
      o << js;
      o.close();
    }
  }
  // Catch any exception and convert to Status return code
  catch (const std::exception &err) {
    RETURN_STATUS_UNEXPECTED("Update json failed ");
  }
  return Status::OK();
}

Status JsonHelper::RemoveKey(const std::string &in_file, const std::string &key, const std::string &out_file) {
  try {
    Path in = Path(in_file);
    nlohmann::json js;
    if (in.Exists()) {
      std::ifstream in_stream(in_file);
      MS_LOG(INFO) << "Filename: " << in_file << ".";
      in_stream >> js;
      in_stream.close();
    }
    js.erase(key);
    MS_LOG(INFO) << "Write outfile is: " << js << ".";
    if (out_file == "") {
      std::ofstream o(in_file, std::ofstream::trunc);
      o << js;
      o.close();
    } else {
      std::ofstream o(out_file, std::ofstream::trunc);
      o << js;
      o.close();
    }
  }
  // Catch any exception and convert to Status return code
  catch (const std::exception &err) {
    RETURN_STATUS_UNEXPECTED("Update json failed ");
  }
  return Status::OK();
}

size_t JsonHelper::DumpData(const unsigned char *tensor_addr, const size_t &tensor_size, void *addr,
                            const size_t &buffer_size) {
  // write to address, input order is: destination, source
  errno_t ret = memcpy_s(addr, buffer_size, tensor_addr, tensor_size);
  if (ret != 0) {
    // memcpy failed
    MS_LOG(ERROR) << "memcpy tensor memory failed"
                  << ".";
    return 0;  // amount of data copied is 0, error
  }
  return tensor_size;
}
}  // namespace dataset
}  // namespace mindspore
