#pragma once
#include <fstream>

#include "nlohmann_json.hpp"

namespace uni {
using Json = nlohmann::json;
static inline Json LoadJsonFromFile(std::string filename) {
    std::ifstream ifs(filename);
    return Json::parse(ifs, nullptr, false, true);
}
static inline Json LoadJsonFromString(std::string json_string) {
    return Json::parse(json_string, nullptr, false, true);
}
static inline Json ReplaceKey(Json json, std::string original_key,
                              std::string new_key) {
    Json result;
    result[new_key] = json[original_key];
    return result;
}
}  // namespace uni
