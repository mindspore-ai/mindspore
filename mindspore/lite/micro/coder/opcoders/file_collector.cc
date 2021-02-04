/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include <set>
#include <vector>
#include <string>
#include "micro/coder/opcoders/file_collector.h"

namespace mindspore::lite::micro {
class Collector {
 public:
  explicit Collector(CoderContext *const ctx) : ctx_(ctx) {}

  virtual ~Collector() = default;
  virtual void operator+=(std::string file) = 0;

 protected:
  CoderContext *const ctx_{nullptr};
};

class HFileCollector : public Collector {
 public:
  HFileCollector() = delete;

  explicit HFileCollector(CoderContext *const ctx) : Collector(ctx) {}

  void operator+=(std::string file) override { this->files_.insert(file); }

  ~HFileCollector() override { this->ctx_->set_h_files(files_); }

 private:
  std::set<std::string> files_;
};

class CFileCollector : public Collector {
 public:
  CFileCollector() = delete;

  explicit CFileCollector(CoderContext *const ctx) : Collector(ctx) {}

  void operator+=(std::string file) override { this->files_.insert(file); }

  ~CFileCollector() override { this->ctx_->set_c_files(this->files_); }

 private:
  std::set<std::string> files_;
};

class ASMFileCollector : public Collector {
 public:
  ASMFileCollector() = delete;

  explicit ASMFileCollector(CoderContext *const ctx) : Collector(ctx) {}

  void operator+=(std::string file) override { this->files_.insert(file); }

  ~ASMFileCollector() override { this->ctx_->set_asm_files(this->files_); }

 private:
  std::set<std::string> files_;
};

void Collect(CoderContext *const ctx, const std::vector<std::string> &headers, const std::vector<std::string> &cFiles,
             const std::vector<std::string> &asmFiles) {
  auto collect = [](Collector &cc, const std::vector<std::string> &content) {
    std::for_each(content.begin(), content.end(), [&cc](const std::string &s) { cc += s; });
  };
  HFileCollector hc(ctx);
  collect(hc, headers);

  CFileCollector cc(ctx);
  collect(cc, cFiles);

  ASMFileCollector ac(ctx);
  collect(ac, asmFiles);
}
}  // namespace mindspore::lite::micro
