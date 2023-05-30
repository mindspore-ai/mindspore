/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "utils/file_utils.h"

#include <climits>
#include <cstring>
#include <string>
#include <optional>
#include <memory>
#include "utils/system/file_system.h"
#include "utils/system/env.h"

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#include <wchar.h>

#undef ERROR  // which is in wingdi.h and conflict with log_adaptor.h
#endif

namespace mindspore {
#if defined(_WIN32) || defined(_WIN64)
int IncludeChinese(const char *str) {
  if (str == nullptr) {
    MS_LOG(ERROR) << "Input str is nullptr";
    return 0;
  }

  char *tmp_str = const_cast<char *>(str);
  while (1) {
    char c = *tmp_str++;

    // end the input str
    if (c == 0) {
      break;
    }

    // The highest bit of chinese character is 1
    if (c & 0x80) {
      if (*tmp_str & 0x80) {
        return 1;
      }
    }
  }
  return 0;
}

bool IsStrUTF_8(const char *str) {
  MS_EXCEPTION_IF_NULL(str);
  uint32_t n_bytes = 0;
  bool b_all_ascii = true;
  for (uint32_t i = 0; str[i] != '\0'; ++i) {
    unsigned char chr = *(str + i);
    if (n_bytes == 0 && (chr & 0x80) != 0) {
      b_all_ascii = false;
    }
    if (n_bytes == 0) {
      if (chr >= 0x80) {
        if (chr >= 0xFC && chr <= 0xFD) {
          n_bytes = 6;
        } else if (chr >= 0xF8) {
          n_bytes = 5;
        } else if (chr >= 0xF0) {
          n_bytes = 4;
        } else if (chr >= 0xE0) {
          n_bytes = 3;
        } else if (chr >= 0xC0) {
          n_bytes = 2;
        } else {
          return false;
        }
        n_bytes--;
      }
    } else {
      if ((chr & 0xC0) != 0x80) {
        return false;
      }
      n_bytes--;
    }
  }

  if (n_bytes != 0) {
    return false;
  }
  if (b_all_ascii) {
    return true;
  }
  return true;
}

bool IsStrGBK(const char *str) {
  MS_EXCEPTION_IF_NULL(str);
  uint32_t n_bytes = 0;
  bool b_all_ascii = true;
  for (uint32_t i = 0; str[i] != '\0'; ++i) {
    unsigned char chr = *(str + i);
    if ((chr & 0x80) != 0 && n_bytes == 0) {
      b_all_ascii = false;
    }
    if (n_bytes == 0) {
      if (chr >= 0x80) {
        if (chr >= 0x81 && chr <= 0xFE) {
          n_bytes = +2;
        } else {
          return false;
        }
        n_bytes--;
      }
    } else {
      if (chr < 0x40 || chr > 0xFE) {
        return false;
      }
      n_bytes--;
    }
  }
  if (n_bytes != 0) {
    return false;
  }
  if (b_all_ascii) {
    return true;
  }
  return true;
}

void UTF_8ToUnicode(WCHAR *p_out, char *p_text) {
  MS_EXCEPTION_IF_NULL(p_out);
  MS_EXCEPTION_IF_NULL(p_text);
  char *uchar = reinterpret_cast<char *>(p_out);
  uchar[1] = ((p_text[0] & 0x0F) << 4) + ((p_text[1] >> 2) & 0x0F);
  uchar[0] = ((p_text[1] & 0x03) << 6) + (p_text[2] & 0x3F);
  return;
}

void UnicodeToGB2312(char *p_out, WCHAR u_data) {
  MS_EXCEPTION_IF_NULL(p_out);
  WideCharToMultiByte(CP_ACP, 0, &u_data, 1, p_out, sizeof(WCHAR), nullptr, nullptr);
  return;
}

std::string FileUtils::UTF_8ToGB2312(const char *text) {
  if (text == nullptr) {
    MS_LOG(ERROR) << "Input text is nullptr";
    return "";
  }

  std::string out;
  if (!IncludeChinese(text) && IsStrUTF_8(text)) {
    out = text;
    return out;
  }

  if (IsStrGBK(text) && !IsStrUTF_8(text)) {
    out = text;
    return out;
  }
  char buf[4] = {0};
  int len = strlen(text);
  char *new_text = const_cast<char *>(text);
  auto rst = std::make_unique<char[]>(len + (len >> 2) + 2);
  errno_t ret = memset_s(rst.get(), len + (len >> 2) + 2, 0, len + (len >> 2) + 2);
  if (ret != EOK) {
    MS_LOG(ERROR) << "memset_s error, error code: " << ret;
    return "";
  }

  int i = 0;
  int j = 0;

  while (i < len) {
    if (*(new_text + i) >= 0) {
      rst[j++] = new_text[i++];
    } else {
      WCHAR w_temp;
      UTF_8ToUnicode(&w_temp, new_text + i);
      UnicodeToGB2312(buf, w_temp);

      rst[j] = buf[0];
      rst[j + 1] = buf[1];
      rst[j + 2] = buf[2];

      i += 3;
      j += 2;
    }
  }

  rst[j] = '\0';
  out = rst.get();
  return out;
}

// gb2312 to utf8
std::string FileUtils::GB2312ToUTF_8(const char *gb2312) {
  if (gb2312 == nullptr) {
    MS_LOG(ERROR) << "Input string gb2312 is nullptr";
    return "";
  }

  if (IsStrUTF_8(gb2312)) {
    return std::string(gb2312);
  }

  int len = MultiByteToWideChar(CP_ACP, 0, gb2312, -1, nullptr, 0);
  auto wstr = std::make_unique<wchar_t[]>(len + 1);
  errno_t ret = memset_s(wstr.get(), len + 1, 0, len + 1);
  if (ret != EOK) {
    MS_LOG(ERROR) << "memset_s error, error code: " << ret;
    return "";
  }
  MultiByteToWideChar(CP_ACP, 0, gb2312, -1, wstr.get(), len);
  len = WideCharToMultiByte(CP_UTF8, 0, wstr.get(), -1, nullptr, 0, nullptr, nullptr);

  auto str = std::make_unique<char[]>(len + 1);
  errno_t ret2 = memset_s(str.get(), len + 1, 0, len + 1);
  if (ret2 != EOK) {
    MS_LOG(ERROR) << "memset_s error, error code: " << ret2;
    return "";
  }
  WideCharToMultiByte(CP_UTF8, 0, wstr.get(), -1, str.get(), len, nullptr, nullptr);
  std::string str_temp(str.get());

  return str_temp;
}
#endif

std::optional<std::string> FileUtils::GetRealPath(const char *path) {
  if (path == nullptr) {
    MS_LOG(ERROR) << "Input path is nullptr";
    return std::nullopt;
  }

  char real_path[PATH_MAX] = {0};
#if defined(_WIN32) || defined(_WIN64)
  std::string new_path = FileUtils::UTF_8ToGB2312(path);
  if (new_path.length() >= PATH_MAX || _fullpath(real_path, new_path.data(), PATH_MAX) == nullptr) {
    MS_LOG(ERROR) << "Get realpath failed, path[" << path << "]";
    return std::nullopt;
  }
#else
  if (strlen(path) >= PATH_MAX || realpath(path, real_path) == nullptr) {
    MS_LOG(ERROR) << "Get realpath failed, path[" << path << "]";
    return std::nullopt;
  }
#endif
  return std::string(real_path);
}

// do not call RealPath function in OpenFile, because OpenFile may open a non-exist file
std::fstream *FileUtils::OpenFile(const std::string &file_path, std::ios_base::openmode open_mode) {
  auto fs = new (std::nothrow) std::fstream();
  if (fs == nullptr) {
    MS_LOG(DEBUG) << "Create file stream failed";
    return nullptr;
  }
  fs->open(file_path, open_mode);
  if (!fs->good()) {
    MS_LOG(DEBUG) << "File is not exist: " << file_path;
    delete fs;
    return nullptr;
  }
  if (!fs->is_open()) {
    MS_LOG(DEBUG) << "Can not open file: " << file_path;
    delete fs;
    return nullptr;
  }
  return fs;
}

bool FileUtils::ParserPathAndModelName(const std::string &output_path, std::string *save_path,
                                       std::string *model_name) {
  auto pos = output_path.find_last_of('/');
  if (pos == std::string::npos) {
    pos = output_path.find_last_of('\\');
  }
  std::string tmp_model_name;
  if (pos == std::string::npos) {
#ifdef _WIN32
    *save_path = ".\\";
#else
    *save_path = "./";
#endif
    tmp_model_name = output_path;
  } else {
    *save_path = output_path.substr(0, pos + 1);
    tmp_model_name = output_path.substr(pos + 1);
  }
  *save_path = FileUtils::GetRealPath(save_path->c_str()).value();
  if (save_path->empty()) {
    MS_LOG(DEBUG) << "File path not regular: " << save_path;
    return false;
  }
  auto suffix_pos = tmp_model_name.find_last_of('.');
  if (suffix_pos == std::string::npos) {
    *model_name = tmp_model_name;
  } else {
    if (tmp_model_name.substr(suffix_pos + 1) == "ms") {
      *model_name = tmp_model_name.substr(0, suffix_pos);
    } else {
      *model_name = tmp_model_name;
    }
  }
  return true;
}

void FileUtils::SplitDirAndFileName(const std::string &path, std::optional<std::string> *prefix_path,
                                    std::optional<std::string> *file_name) {
  auto path_split_pos = path.find_last_of('/');
  auto path_split_pos_backslash = path.find_last_of('\\');
  if (path_split_pos != std::string::npos) {
    if (path_split_pos_backslash != std::string::npos && path_split_pos < path_split_pos_backslash) {
      path_split_pos = path_split_pos_backslash;
    }
  } else {
    path_split_pos = path_split_pos_backslash;
  }

  MS_EXCEPTION_IF_NULL(prefix_path);
  MS_EXCEPTION_IF_NULL(file_name);

  if (path_split_pos != std::string::npos) {
    *prefix_path = path.substr(0, path_split_pos);
    *file_name = path.substr(path_split_pos + 1);
  } else {
    *prefix_path = std::nullopt;
    *file_name = path;
  }
}

void FileUtils::ConcatDirAndFileName(const std::optional<std::string> *dir, const std::optional<std::string> *file_name,
                                     std::optional<std::string> *path) {
  MS_EXCEPTION_IF_NULL(dir);
  MS_EXCEPTION_IF_NULL(file_name);
  MS_EXCEPTION_IF_NULL(path);
#if defined(_WIN32) || defined(_WIN64)
  *path = dir->value() + "\\" + file_name->value();
#else
  *path = dir->value() + "/" + file_name->value();
#endif
}

std::optional<std::string> FileUtils::CreateNotExistDirs(const std::string &path, const bool support_relative_path) {
  if (path.size() >= PATH_MAX) {
    MS_LOG(ERROR) << "The length of the path is greater than or equal to:" << PATH_MAX;
    return std::nullopt;
  }
  if (!support_relative_path) {
    auto dot_pos = path.find("..");
    if (dot_pos != std::string::npos) {
      MS_LOG(ERROR) << "Do not support relative path";
      return std::nullopt;
    }
  }

  std::shared_ptr<system::FileSystem> fs = system::Env::GetFileSystem();
  MS_EXCEPTION_IF_NULL(fs);
  char temp_path[PATH_MAX] = {0};
  for (uint32_t i = 0; i < path.length(); i++) {
    temp_path[i] = path[i];
    if (temp_path[i] == '\\' || temp_path[i] == '/') {
      if (i != 0) {
        char tmp_char = temp_path[i];
        temp_path[i] = '\0';
        std::string path_handle(temp_path);
        if (!fs->FileExist(path_handle)) {
          if (!fs->CreateDir(path_handle)) {
            MS_LOG(ERROR) << "Create " << path_handle << " dir error";
            return std::nullopt;
          }
        }
        temp_path[i] = tmp_char;
      }
    }
  }

  if (!fs->FileExist(path)) {
    if (!fs->CreateDir(path)) {
      MS_LOG(ERROR) << "Create " << path << " dir error";
      return std::nullopt;
    }
  }
  return GetRealPath(path.c_str());
}
}  // namespace mindspore
