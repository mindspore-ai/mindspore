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

#include "utils/log_adapter.h"

#include <unistd.h>
#include <map>
#include "pybind11/pybind11.h"
#include "debug/trace.h"

// namespace to support utils module definition
namespace mindspore {
#ifdef USE_GLOG
static std::string GetTime() {
#define BUFLEN 80
  static char buf[BUFLEN];
#if defined(_WIN32) || defined(_WIN64)
  time_t time_seconds = time(0);
  struct tm now_time;
  localtime_s(&now_time, &time_seconds);
  sprintf_s(buf, BUFLEN, "%d-%d-%d %d:%d:%d", now_time.tm_year + 1900, now_time.tm_mon + 1, now_time.tm_mday,
            now_time.tm_hour, now_time.tm_min, now_time.tm_sec);
#else
  struct timeval cur_time;
  (void)gettimeofday(&cur_time, nullptr);

  struct tm now;
  (void)localtime_r(&cur_time.tv_sec, &now);
  (void)strftime(buf, BUFLEN, "%Y-%m-%d-%H:%M:%S", &now);  // format date and time
  // set micro-second
  buf[27] = '\0';
  int idx = 26;
  auto num = cur_time.tv_usec;
  for (int i = 5; i >= 0; i--) {
    buf[idx--] = static_cast<char>(num % 10 + '0');
    num /= 10;
    if (i % 3 == 0) {
      buf[idx--] = '.';
    }
  }
#endif
  return std::string(buf);
}

static std::string GetProcName() {
#if defined(__APPLE__) || defined(__FreeBSD__)
  const char *appname = getprogname();
#elif defined(_GNU_SOURCE)
  const char *appname = program_invocation_name;
#else
  const char *appname = "?";
#endif
  // some times, the appname is an absolute path, its too long
  std::string app_name(appname);
  std::size_t pos = app_name.rfind("/");
  if (pos == std::string::npos) {
    return app_name;
  }
  if (pos + 1 >= app_name.size()) {
    return app_name;
  }
  return app_name.substr(pos + 1);
}

static std::string GetLogLevel(MsLogLevel level) {
#define _TO_STRING(x) #x
  static const char *const level_names[] = {
    _TO_STRING(DEBUG),
    _TO_STRING(INFO),
    _TO_STRING(WARNING),
    _TO_STRING(ERROR),
  };
#undef _TO_STRING
  if (level > ERROR) {
    level = ERROR;
  }
  return std::string(level_names[level]);
}

// convert MsLogLevel to corresponding glog level
static int GetGlogLevel(MsLogLevel level) {
  switch (level) {
    case DEBUG:
    case INFO:
      return google::GLOG_INFO;
    case WARNING:
      return google::GLOG_WARNING;
    case ERROR:
    default:
      return google::GLOG_ERROR;
  }
}
#else

#undef Dlog
#define Dlog(module_id, level, format, ...)                   \
  do {                                                        \
    DlogInner((module_id), (level), (format), ##__VA_ARGS__); \
  } while (0)

// convert MsLogLevel to corresponding slog level
static int GetSlogLevel(MsLogLevel level) {
  switch (level) {
    case DEBUG:
      return DLOG_DEBUG;
    case INFO:
      return DLOG_INFO;
    case WARNING:
      return DLOG_WARN;
    case ERROR:
    default:
      return DLOG_ERROR;
  }
}
#endif

static std::string ExceptionTypeToString(ExceptionType type) {
#define _TO_STRING(x) #x
  // clang-format off
  static const char *const type_names[] = {
      _TO_STRING(NoExceptionType),
      _TO_STRING(UnknownError),
      _TO_STRING(ArgumentError),
      _TO_STRING(NotSupportError),
      _TO_STRING(NotExistsError),
      _TO_STRING(AlreadyExistsError),
      _TO_STRING(UnavailableError),
      _TO_STRING(DeviceProcessError),
      _TO_STRING(AbortedError),
      _TO_STRING(TimeOutError),
      _TO_STRING(ResourceUnavailable),
      _TO_STRING(NoPermissionError),
      _TO_STRING(IndexError),
      _TO_STRING(ValueError),
      _TO_STRING(TypeError),
  };
  // clang-format on
#undef _TO_STRING
  if (type < UnknownError || type > TypeError) {
    type = UnknownError;
  }
  return std::string(type_names[type]);
}

static const char *GetSubModuleName(SubModuleId module_id) {
  static const char *sub_module_names[NUM_SUBMODUES] = {
    "UNKNOWN",    // SM_UNKNOWN
    "ANALYZER",   // SM_ANALYZER
    "COMMON",     // SM_COMMON
    "DEBUG",      // SM_DEBUG
    "DEVICE",     // SM_DEVICE
    "GE_ADPT",    // SM_GE_ADPT
    "IR",         // SM_IR
    "KERNEL",     // SM_KERNEL
    "MD",         // SM_MD
    "ME",         // SM_ME
    "ONNX",       // SM_ONNX
    "OPTIMIZER",  // SM_OPTIMIZER
    "PARALLEL",   // SM_PARALLEL
    "PARSER",     // SM_PARSER
    "PIPELINE",   // SM_PIPELINE
    "PRE_ACT",    // SM_PRE_ACT
    "PYNATIVE",   // SM_PYNATIVE
    "SESSION",    // SM_SESSION
    "UTILS",      // SM_UTILS
    "VM"          // SM_VM
  };

  return sub_module_names[module_id % NUM_SUBMODUES];
}

void LogWriter::OutputLog(const std::ostringstream &msg) const {
#ifdef USE_GLOG
  auto submodule_name = GetSubModuleName(submodule_);
  google::LogMessage("", 0, GetGlogLevel(log_level_)).stream()
    << "[" << GetLogLevel(log_level_) << "] " << submodule_name << "(" << getpid() << "," << GetProcName()
    << "):" << GetTime() << " "
    << "[" << location_.file_ << ":" << location_.line_ << "] " << location_.func_ << "] " << msg.str() << std::endl;
#else
  auto str_msg = msg.str();
  auto slog_module_id = (submodule_ == SM_MD ? MD : ME);
  Dlog(static_cast<int>(slog_module_id), GetSlogLevel(log_level_), "[%s:%d] %s] %s", location_.file_, location_.line_,
       location_.func_, str_msg.c_str());
#endif
}

void LogWriter::operator<(const LogStream &stream) const noexcept {
  std::ostringstream msg;
  msg << stream.sstream_->rdbuf();
  OutputLog(msg);
}

void LogWriter::operator^(const LogStream &stream) const {
  std::ostringstream msg;
  msg << stream.sstream_->rdbuf();
  OutputLog(msg);

  std::ostringstream oss;
  oss << location_.file_ << ":" << location_.line_ << " " << location_.func_ << "] ";
  if (exception_type_ != NoExceptionType && exception_type_ != IndexError && exception_type_ != TypeError &&
      exception_type_ != ValueError) {
    oss << ExceptionTypeToString(exception_type_) << " ";
  }
  oss << msg.str();

  trace::TraceGraphEval();
  trace::GetEvalStackInfo(oss);

  if (exception_type_ == IndexError) {
    throw pybind11::index_error(oss.str());
  }
  if (exception_type_ == ValueError) {
    throw pybind11::value_error(oss.str());
  }
  if (exception_type_ == TypeError) {
    throw pybind11::type_error(oss.str());
  }
  pybind11::pybind11_fail(oss.str());
}

static std::string GetEnv(const std::string &envvar) {
  const char *value = ::getenv(envvar.c_str());

  if (value == nullptr) {
    return std::string();
  }

  return std::string(value);
}

enum LogConfigToken {
  INVALID,      // indicate invalid token
  LEFT_BRACE,   // '{'
  RIGHT_BRACE,  // '}'
  VARIABLE,     // '[A-Za-z][A-Za-z0-9_]*'
  NUMBER,       // [0-9]+
  COMMA,        // ','
  COLON,        // ':'
  EOS,          // End Of String, '\0'
  NUM_LOG_CFG_TOKENS
};

static const char *g_tok_names[NUM_LOG_CFG_TOKENS] = {
  "invalid",        // indicate invalid token
  "{",              // '{'
  "}",              // '}'
  "variable",       // '[A-Za-z][A-Za-z0-9_]*'
  "number",         // [0-9]+
  ",",              // ','
  ":",              // ':'
  "end-of-string",  // End Of String, '\0'
};

static inline bool IsAlpha(char ch) { return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z'); }

static inline bool IsDigit(char ch) { return ch >= '0' && ch <= '9'; }

class LogConfigLexer {
 public:
  explicit LogConfigLexer(const std::string &text) : buffer_(text) {
    cur_idx_ = 0;
    cur_token_ = LogConfigToken::INVALID;
  }
  ~LogConfigLexer() = default;

  // skip white space, and return the first char after white space
  char SkipWhiteSpace() {
    while (cur_idx_ < buffer_.size()) {
      char ch = buffer_[cur_idx_];
      if (ch == ' ' || ch == '\t') {
        ++cur_idx_;
        continue;
      }
      return ch;
    }
    return '\0';
  }

  LogConfigToken GetNext(std::string *const ptr) {
#ifdef DEBUG
    std::string text;
    auto tok = GetNextInner(&text);
    MS_LOG(DEBUG) << "Got token " << tok << " with value [" << text << "]";
    if (ptr != nullptr) {
      *ptr = text;
    }
    return tok;
  }

  LogConfigToken GetNextInner(std::string *ptr) {
#endif
    char ch = SkipWhiteSpace();
    // clang-format off
    static const std::map<char, LogConfigToken> single_char_map = {
      {'{', LogConfigToken::LEFT_BRACE},
      {'}', LogConfigToken::RIGHT_BRACE},
      {',', LogConfigToken::COMMA},
      {':', LogConfigToken::COLON},
      {'\0', LogConfigToken::EOS},
    };
    // clang-format on

    auto iter = single_char_map.find(ch);
    if (iter != single_char_map.end()) {
      if (ptr != nullptr) {
        *ptr = std::string() + ch;
      }
      ++cur_idx_;
      return iter->second;
    } else if (IsAlpha(ch)) {
      std::ostringstream oss;
      do {
        oss << ch;
        ch = buffer_[++cur_idx_];
      } while (cur_idx_ < buffer_.size() && (IsAlpha(ch) || IsDigit(ch) || ch == '_'));
      if (ptr != nullptr) {
        *ptr = std::string(oss.str());
      }
      return LogConfigToken::VARIABLE;
    } else if (IsDigit(ch)) {
      std::ostringstream oss;
      do {
        oss << ch;
        ch = buffer_[++cur_idx_];
      } while (cur_idx_ < buffer_.size() && IsDigit(ch));
      if (ptr != nullptr) {
        *ptr = std::string(oss.str());
      }
      return LogConfigToken::NUMBER;
    }
    return LogConfigToken::INVALID;
  }

 private:
  std::string buffer_;
  size_t cur_idx_;

  LogConfigToken cur_token_;
  std::string cur_text_;
};

class LogConfigParser {
 public:
  explicit LogConfigParser(const std::string &cfg) : lexer(cfg) {}
  ~LogConfigParser() = default;

  bool Expect(LogConfigToken expected, LogConfigToken tok) {
    if (expected != tok) {
      MS_LOG(WARNING) << "Parse submodule log configuration text error, expect `" << g_tok_names[expected]
                      << "`, but got `" << g_tok_names[tok] << "`. The whole configuration will be ignored.";
      return false;
    }
    return true;
  }

  // The text of config MS_SUBMODULE_LOG_v is in the form {submodule1:log_level1,submodule2:log_level2,...}.
  // Valid values of log levels are: 0 - debug, 1 - info, 2 - warning, 3 - error
  // e.g. MS_SUBMODULE_LOG_v={PARSER:0, ANALYZER:2, PIPELINE:1}
  std::map<std::string, std::string> Parse() {
    std::map<std::string, std::string> log_levels;

    bool flag_error = false;
    std::string text;
    auto tok = lexer.GetNext(&text);

    // empty string
    if (tok == LogConfigToken::EOS) {
      return log_levels;
    }

    if (!Expect(LogConfigToken::LEFT_BRACE, tok)) {
      return log_levels;
    }

    do {
      std::string key, val;
      tok = lexer.GetNext(&key);
      if (!Expect(LogConfigToken::VARIABLE, tok)) {
        flag_error = true;
        break;
      }

      tok = lexer.GetNext(&text);
      if (!Expect(LogConfigToken::COLON, tok)) {
        flag_error = true;
        break;
      }

      tok = lexer.GetNext(&val);
      if (!Expect(LogConfigToken::NUMBER, tok)) {
        flag_error = true;
        break;
      }

      log_levels[key] = val;
      tok = lexer.GetNext(&text);
    } while (tok == LogConfigToken::COMMA);

    if (!flag_error && !Expect(LogConfigToken::RIGHT_BRACE, tok)) {
      flag_error = true;
    }

    if (flag_error) {
      log_levels.clear();
    }
    return log_levels;
  }

 private:
  LogConfigLexer lexer;
};

bool ParseLogLevel(const std::string &str_level, MsLogLevel *ptr_level) {
  if (str_level.size() == 1) {
    int ch = str_level.c_str()[0];
    ch = ch - '0';  // substract ASCII code of '0', which is 48
    if (ch >= DEBUG && ch <= ERROR) {
      if (ptr_level != nullptr) {
        *ptr_level = static_cast<MsLogLevel>(ch);
      }
      return true;
    }
  }
  return false;
}

static MsLogLevel GetGlobalLogLevel() {
#ifdef USE_GLOG
  return static_cast<MsLogLevel>(FLAGS_v);
#else
  int log_level = WARNING;  // set default log level to WARNING
  auto str_level = GetEnv("GLOG_v");
  if (str_level.size() == 1) {
    int ch = str_level.c_str()[0];
    ch = ch - '0';  // substract ASCII code of '0', which is 48
    if (ch >= DEBUG && ch <= ERROR) {
      log_level = ch;
    }
  }
  return static_cast<MsLogLevel>(log_level);
#endif
}

void InitSubModulesLogLevel() {
  // initialize submodule's log level using global
  auto global_log_level = GetGlobalLogLevel();
  for (int i = 0; i < NUM_SUBMODUES; ++i) {
    g_ms_submodule_log_levels[i] = global_log_level;
  }

  // set submodule's log level
  auto submodule = GetEnv("MS_SUBMODULE_LOG_v");
  MS_LOG(DEBUG) << "MS_SUBMODULE_LOG_v=`" << submodule << "`";
  LogConfigParser parser(submodule);
  auto configs = parser.Parse();
  for (const auto &cfg : configs) {
    int mod_idx = -1;
    for (int i = 0; i < NUM_SUBMODUES; ++i) {
      if (cfg.first == GetSubModuleName(static_cast<SubModuleId>(i))) {
        mod_idx = i;
        break;
      }
    }
    if (mod_idx < 0) {
      MS_LOG(WARNING) << "Undefined module name " << cfg.first << ", ignore it";
      continue;
    }
    MsLogLevel submodule_log_level;
    if (!ParseLogLevel(cfg.second, &submodule_log_level)) {
      MS_LOG(WARNING) << "Illegal log level value " << cfg.second << " for " << cfg.first << ", ignore it.";
      continue;
    }
    g_ms_submodule_log_levels[mod_idx] = submodule_log_level;
  }
}
}  // namespace mindspore

extern "C" {
#if defined(_WIN32) || defined(_WIN64)
__attribute__((constructor)) void common_log_init(void) {
#else
void common_log_init(void) {
#endif
#ifdef USE_GLOG
  // do not use glog predefined log prefix
  FLAGS_log_prefix = false;
  // set default log level to WARNING
  if (mindspore::GetEnv("GLOG_v").empty()) {
    FLAGS_v = mindspore::WARNING;
  }

  // set default log file mode to 0640
  if (mindspore::GetEnv("GLOG_logfile_mode").empty()) {
    FLAGS_logfile_mode = 0640;
  }
  std::string logtostderr = mindspore::GetEnv("GLOG_logtostderr");
  // default print log to screen
  if (logtostderr.empty()) {
    FLAGS_logtostderr = true;
  } else if (logtostderr == "0" && mindspore::GetEnv("GLOG_log_dir").empty()) {
    FLAGS_logtostderr = true;
    MS_LOG(WARNING) << "`GLOG_log_dir` is not set, output log to screen.";
  }
#endif
  mindspore::InitSubModulesLogLevel();
}

// shared lib init hook
#if defined(_WIN32) || defined(_WIN64)
__attribute__((constructor)) void mindspore_log_init(void) {
#else
void mindspore_log_init(void) {
#endif
#ifdef USE_GLOG
  static bool is_glog_initialzed = false;
  if (!is_glog_initialzed) {
#if !defined(_WIN32) && !defined(_WIN64)
    google::InitGoogleLogging("mindspore");
#endif
    is_glog_initialzed = true;
  }
#endif
  common_log_init();
}
}
