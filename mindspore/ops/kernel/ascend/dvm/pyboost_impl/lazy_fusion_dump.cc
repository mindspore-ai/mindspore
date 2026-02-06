/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/dvm/pyboost_impl/lazy_fusion_dump.h"
#include "kernel/ascend/dvm/pyboost_impl/lazy_fusion_flags.h"
#include "utils/file_utils.h"

namespace mindspore {
namespace kernel {
namespace {
static constexpr size_t kTensorSizePrintLimit = 2;
}

LazyFusionDump &LazyFusionDump::Instance() {
  static LazyFusionDump instance;
  return instance;
}

LazyFusionDump::LazyFusionDump() { CreateDumpDir(); }

std::string LazyFusionDump::ToString(const TensorPtr &t) {
  if (t == nullptr) {
    return "nullptr";
  }
  auto storage_info = t->storage_info();
  if (storage_info == nullptr) {
    return t->ToStringInternal(kTensorSizePrintLimit);
  }
  return t->ToStringInternal(kTensorSizePrintLimit) + " " + storage_info->ToString();
}

void LazyFusionDump::DumpGraphInfo(std::stringstream *buf) {
  std::string file_name = "lazy_fusion_" + std::to_string(getpid()) + "_graph.txt";
  std::string file_path = dump_dir_ + "/" + file_name;
  DumpToFile(file_path, buf);
}

void LazyFusionDump::DumpKernelInfo(std::stringstream *buf) {
  std::string file_name = "lazy_fusion_" + std::to_string(getpid()) + "_kernel.txt";
  std::string file_path = dump_dir_ + "/" + file_name;
  DumpToFile(file_path, buf);
}

void LazyFusionDump::DumpToFile(const std::string &file_path, std::stringstream *buf) {
  if (!enable_dump_) {
    return;
  }
  ChangeFileMode(file_path, S_IWUSR);
  std::ofstream of(file_path, std::ios::app);
  if (!of.is_open()) {
    MS_LOG(WARNING) << "Open dump file '" << file_path << "' failed!";
    ChangeFileMode(file_path, S_IRUSR);
    return;
  }
  MS_EXCEPTION_IF_NULL(buf);
  of << buf->str() << "\n";
  of.close();
  ChangeFileMode(file_path, S_IRUSR);
  buf->str("");
}

void LazyFusionDump::CreateDumpDir() {
  auto dir_path = FileUtils::CreateNotExistDirs(LazyFusionFlags::GetInstance().dump_dir);
  enable_dump_ = dir_path.has_value();
  if (!enable_dump_) {
    MS_LOG(WARNING) << "Failed to create dump directory: " << LazyFusionFlags::GetInstance().dump_dir;
    return;
  }
  dump_dir_ = dir_path.value();
  MS_LOG(INFO) << "dump directory: " << dump_dir_;
}
}  // namespace kernel
}  // namespace mindspore
