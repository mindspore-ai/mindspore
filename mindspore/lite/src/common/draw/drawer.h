/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_COMMON_DRAW_DRAWER_H_
#define MINDSPORE_LITE_SRC_COMMON_DRAW_DRAWER_H_

#include <string>

#include "src/executor/sub_graph_kernel.h"

#ifdef ENABLE_CLOUD_INFERENCE
#include "src/extendrt/graph_compiler/compile_result.h"
#endif

namespace mindspore {
namespace lite {
class Drawer {
 public:
  static Drawer &Instance() {
    static Drawer instance;
    return instance;
  }

  void Init();

  void Reset();

  std::string GetNextFileName(const std::string &name);

  void Draw(const kernel::SubGraphKernel *graph, const std::string &name = "");
#ifdef ENABLE_CLOUD_INFERENCE
  void Draw(const CompileResult *graph, const std::string &name = "");
#endif

 private:
  Drawer() = default;
  bool SaveDotFile(const std::string &dot_name, const std::string &dot_content);

  bool enabled_{false};
  std::string base_dir_;
  size_t count_{0};
};
}  // namespace lite

#if (defined Debug) && (defined ENABLE_DRAW)
#define InitDotDrawer() mindspore::lite::Drawer::Instance().Init()
#define DrawDot(graph, name) mindspore::lite::Drawer::Instance().Draw(graph, name)
#else
#define InitDotDrawer()
#define DrawDot(graph, name)
#endif
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_COMMON_DRAW_DRAWER_H_
