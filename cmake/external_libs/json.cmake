set(nlohmann_json_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(nlohmann_json_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
mindspore_add_pkg(nlohmann_json
        VER 3.6.1
        HEAD_ONLY ./
        URL https://github.com/nlohmann/json/releases/download/v3.6.1/include.zip
        MD5 0dc903888211db3a0f170304cd9f3a89)
include_directories(${nlohmann_json_INC})
add_library(mindspore::json ALIAS nlohmann_json)