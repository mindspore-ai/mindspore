/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef SUMMARY_EVENT_WRITER_H_
#define SUMMARY_EVENT_WRITER_H_

#include <memory>
#include <string>

#include "pybind11/pybind11.h"
#include "securec/include/securec.h"
#include "utils/system/base.h"
#include "utils/system/file_system.h"
#include "utils/system/crc32c.h"
#include "utils/system/env.h"

namespace mindspore {
namespace summary {
namespace py = pybind11;
using string = std::string;
using Env = system::Env;
using WriteFile = system::WriteFile;
using WriteFilePtr = std::shared_ptr<WriteFile>;
using FileSystem = system::FileSystem;

class EventWriter {
 public:
  // The file name = path + file_name
  explicit EventWriter(const std::string &file_full_name);

  ~EventWriter();

  // return the file name
  std::string GetFileName() const { return filename_; }

  // return the count of write event
  int32_t GetWriteEventCount() const;

  // Open the file
  bool Open();

  // write the Serialized "event_str" to file
  void Write(const std::string &event_str);

  // Flush the cache to disk
  bool Flush();

  // close the file
  bool Close() noexcept;

  // Final close: flush and close the event writer and clean
  bool Shut() noexcept;

  // Summary Record Format:
  //  1 uint64 : data length
  //  2 uint32 : mask crc value of data length
  //  3 bytes  : data
  //  4 uint32 : mask crc value of data
  bool WriteRecord(const std::string &data);

 private:
  // True: valid / False: closed
  bool status_ = false;
  std::shared_ptr<FileSystem> fs_;
  std::string filename_;
  WriteFilePtr event_file_;
  int32_t events_write_count_ = 0;
};

}  // namespace summary
}  // namespace mindspore

#endif  // SUMMARY_EVENT_WRITER_H_
