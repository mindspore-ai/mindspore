/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_SUMMARY_EVENT_WRITER_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_SUMMARY_EVENT_WRITER_H_

#include <memory>
#include <string>

#include "pybind11/pybind11.h"
#include "include/securec.h"
#include "tools/visible.h"

namespace mindspore {
namespace system {
class FileSystem;
class WriteFile;
}  // namespace system
namespace summary {
class TOOLS_EXPORT EventWriter {
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
  void CloseFile() noexcept;
  // True: valid / False: closed
  bool status_ = false;
  std::shared_ptr<system::FileSystem> fs_;
  std::string filename_;
  std::shared_ptr<system::WriteFile> event_file_;
  int32_t events_write_count_ = 0;
};

}  // namespace summary
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_SUMMARY_EVENT_WRITER_H_
