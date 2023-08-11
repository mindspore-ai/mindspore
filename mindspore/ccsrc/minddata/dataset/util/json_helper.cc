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
  RETURN_IF_NOT_OK(RealPath(in_dir));
  if (!base_dir.IsDirectory() || !base_dir.Exists()) {
    RETURN_STATUS_UNEXPECTED("Input dir is not a directory or doesn't exist");
  }
  // check if output_dir exists and create it if it does not exist
  Path target_dir = Path(out_dir);
  RETURN_IF_NOT_OK(target_dir.CreateDirectory());

  // iterate over in dir and create json for all images
  uint64_t index = 0;
  auto dir_it = Path::DirIterator::OpenDirectory(&base_dir);
  RETURN_UNEXPECTED_IF_NULL(dir_it);
  while (dir_it->HasNext()) {
    Path v = dir_it->Next();
    // check if found file fits image extension

    // create json file in output dir with the path
    std::string out_file = out_dir + "/" + std::to_string(index) + ".json";
    RETURN_IF_NOT_OK(UpdateValue(out_file, "image", v.ToString(), out_file));
    index++;
  }
  return Status::OK();
}

Status JsonHelper::RealPath(const std::string &path) {
  std::string real_path;
  RETURN_IF_NOT_OK(Path::RealPath(path, real_path));
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
      RETURN_IF_NOT_OK(RealPath(in_file));
      std::ifstream in_stream(in_file, std::ios::in);
      try {
        MS_LOG(INFO) << "Filename: " << in_file << ".";
        in_stream >> js;
      } catch (const std::exception &err) {
        in_stream.close();
        RETURN_STATUS_UNEXPECTED("Invalid file, failed to open json file: " + in_file +
                                 ", please delete it and try again!");
      }
      in_stream.close();
    }
    js[key] = value;
    if (out_file == "") {
      std::ofstream o(in_file, std::ofstream::out | std::ofstream::trunc);
      o << js;
      o.close();
      platform::ChangeFileMode(in_file, S_IRUSR | S_IWUSR);
    } else {
      std::ofstream o(out_file, std::ofstream::out | std::ofstream::trunc);
      o << js;
      o.close();
      platform::ChangeFileMode(out_file, S_IRUSR | S_IWUSR);
    }
  }
  // Catch any exception and convert to Status return code
  catch (nlohmann::json::exception &e) {
    std::string err_msg = "Parse json failed. Error info: ";
    err_msg += e.what();
    RETURN_STATUS_UNEXPECTED(err_msg);
  } catch (const std::exception &e) {
    std::string err_msg = "Update json failed. Error info: ";
    err_msg += e.what();
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}

Status JsonHelper::RemoveKey(const std::string &in_file, const std::string &key, const std::string &out_file) {
  try {
    Path in = Path(in_file);
    nlohmann::json js;
    if (in.Exists()) {
      RETURN_IF_NOT_OK(RealPath(in_file));
      std::ifstream in_stream(in_file, std::ios::in);
      try {
        MS_LOG(INFO) << "Filename: " << in_file << ".";
        in_stream >> js;
      } catch (const std::exception &err) {
        in_stream.close();
        RETURN_STATUS_UNEXPECTED("Invalid file, failed to open json file: " + in_file +
                                 ", please delete it and try again!");
      }
      in_stream.close();
    }
    (void)js.erase(key);
    MS_LOG(INFO) << "Write outfile is: " << js << ".";
    if (out_file == "") {
      std::ofstream o(in_file, std::ios::out | std::ofstream::trunc);
      o << js;
      o.close();
      platform::ChangeFileMode(in_file, S_IRUSR | S_IWUSR);
    } else {
      std::ofstream o(out_file, std::ios::out | std::ofstream::trunc);
      o << js;
      o.close();
      platform::ChangeFileMode(out_file, S_IRUSR | S_IWUSR);
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
