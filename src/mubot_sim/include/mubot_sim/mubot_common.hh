/* Helpers shared by the MUBot Hydro and Actuator plugins
 * (Gazebo Harmonic / gz-sim8).
 *
 * Original (Gazebo Classic) by Donghao Li.
 */

#pragma once

#include <gz/sim/Entity.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/components/Inertial.hh>

#include <sdf/Element.hh>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ROS 2 package lookup is optional: the Database directory can also be given
// through <database_dir> in the plugin SDF or the MUBOT_DATABASE_DIR env var.
#if __has_include(<ament_index_cpp/get_package_share_directory.hpp>)
#include <ament_index_cpp/get_package_share_directory.hpp>
#define MUBOT_HAVE_AMENT 1
#else
#define MUBOT_HAVE_AMENT 0
#endif

namespace mubot
{
    constexpr double Pi = 3.1415926;

    //------------------------------------------------------------------
    // CSV reading and processing
    //------------------------------------------------------------------

    // erase empty line and space
    inline std::string Trim(std::string s) {
        auto is_ws = [](unsigned char ch){ return std::isspace(ch) != 0; };
        while (!s.empty() && is_ws(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
        while (!s.empty() && is_ws(static_cast<unsigned char>(s.back())))  s.pop_back();
        return s;
    }

    // split comma seperate csv into vector of strings
    inline std::vector<std::string> Split_CSV(const std::string& line) {
        std::istringstream iss(line);
        std::vector<std::string> cols;
        std::string tok;
        while (std::getline(iss, tok, ',')) cols.push_back(Trim(tok));
        return cols;
    }

    // `tag` is a message prefix for "[Hydro]" or "[Actuator]".
    // Safely convert string to double
    inline double Parse_Double(const std::string& s, int line_no, int col_no,
                               const std::string& tag = "") {
        const std::string pre = tag.empty() ? "" : tag + " ";
        const std::string t = Trim(s);
        if (t.empty()) {
            throw std::runtime_error(pre + "_CSV parse error: empty numeric field at line " +
                                     std::to_string(line_no) + ", col " + std::to_string(col_no));
        }
        size_t idx = 0;
        double v = 0.0;
        try {
            v = std::stod(t, &idx);
        }
        catch (const std::exception& e) {
            throw std::runtime_error(pre + "_CSV parse error: invalid number '" + t +
                                     "' at line " + std::to_string(line_no) +
                                     ", col " + std::to_string(col_no) + " (" + e.what() + ")");
        }
        if (idx != t.size()) {
            throw std::runtime_error(pre + "_CSV parse error: trailing chars in number '" + t +
                                     "' at line " + std::to_string(line_no) +
                                     ", col " + std::to_string(col_no));
        }
        if (!std::isfinite(v)) {
            throw std::runtime_error(pre + "_CSV parse error: non-finite number '" + t +
                                     "' at line " + std::to_string(line_no) +
                                     ", col " + std::to_string(col_no));
        }
        return v;
    }

    struct CsvRow {
        int line_no = 0;
        std::vector<std::string> cols;
    };

    // Reads all non-empty lines; every line must have exactly `expected_cols` columns. 
    // 15 for hydro_parameter and 4 for joint parameter
    inline std::vector<CsvRow> Read_CSV_Rows(const std::string& path,
                                             size_t expected_cols,
                                             const std::string& tag) 
    {
        std::ifstream in(path);
        if (!in.is_open()) {
            throw std::runtime_error(tag + " Cannot open CSV: " + path);
        }
        std::vector<CsvRow> rows;
        std::string line;
        int line_no = 0;
        while (std::getline(in, line)) {
            ++line_no;
            const std::string stripped = Trim(line);
            if (stripped.empty()) continue;
            CsvRow r;
            r.line_no = line_no;
            r.cols = Split_CSV(stripped);
            if (r.cols.size() != expected_cols) {
                throw std::runtime_error(tag + " CSV format error: expected exactly " +
                                         std::to_string(expected_cols) + " columns, got " +
                                         std::to_string(r.cols.size()) + " at line " +
                                         std::to_string(line_no) + " of " + path);
            }
            rows.push_back(std::move(r));
        }
        return rows;
    }

    // control_signal.csv: first non-empty line, exactly `expected_cols` numeric columns.
    inline std::vector<double> Read_Control_Signal(const std::string& path,
                                                   size_t expected_cols,
                                                   const std::string& tag) {
        std::ifstream in(path);
        if (!in.is_open()) {
            throw std::runtime_error(tag + " Cannot open CSV: " + path);
        }
        std::string line;
        while (std::getline(in, line)) {
            line = Trim(line);
            if (!line.empty()) break;
        }
        if (line.empty()) {
            throw std::runtime_error(tag + " control_signal.csv is empty: " + path);
        }
        const auto c = Split_CSV(line);
        if (c.size() != expected_cols) {
            throw std::runtime_error(tag + " CSV format error: control_signal.csv expects exactly " +
                                     std::to_string(expected_cols) + " columns, got " +
                                     std::to_string(c.size()) + ". File: " + path);
        }
        std::vector<double> raw(expected_cols, 0.0);
        for (size_t i = 0; i < expected_cols; ++i) {
            raw[i] = Parse_Double(c[i], 1, static_cast<int>(i), tag);
        }
        return raw;
    }

    inline bool File_Exists(const std::string& p) {
        std::ifstream f(p);
        return f.good();
    }

    // ------------------------------------------------------------------
    // Numerics
    // ------------------------------------------------------------------
    inline bool Is_Zero(double v, double abs_eps = 1e-8) {
        return std::isfinite(v) && (std::abs(v) <= abs_eps);
    }

    inline double Sign_function(double v, double eps = 1e-12) {
        if (!std::isfinite(v) || std::abs(v) <= eps) return 0.0;
        return (v > 0.0) ? 1.0 : -1.0;
    }

    inline double Sanitize(double v) {
        return std::isfinite(v) ? v : 0.0;
    }

    inline double Clamp(double v, double lim) {
        if (!std::isfinite(v)) return 0.0;
        if (v >  lim) return  lim;
        if (v < -lim) return -lim;
        return v;
    }

    inline int64_t Safe_Round_Step(double t, double dt) {
        if (!(dt > 0.0) || !std::isfinite(dt)) return 0;
        return static_cast<int64_t>(std::llround(t / dt));
    }

    inline double Unwrap_To_Near(double angle, double reference) {
        while (angle - reference >  Pi) angle -= 2.0 * Pi;
        while (angle - reference < -Pi) angle += 2.0 * Pi;
        return angle;
    }

    // ------------------------------------------------------------------
    // Gazebo helpers
    // ------------------------------------------------------------------

    // Mass and Izz (about the CoM, link axes) of a link entity.
    inline bool Get_Link_Mass_Izz(const gz::sim::EntityComponentManager& ecm,
                                  gz::sim::Entity link,
                                  double& mass, double& izz) {
        const auto* c = ecm.Component<gz::sim::components::Inertial>(link);
        if (!c) return false;
        mass = c->Data().MassMatrix().Mass();
        izz  = c->Data().MassMatrix().Izz();
        return true;
    }

    // Model "mubot_XYZ" -> "XYZ" (same rule as the classic plugins).
    inline std::string Model_Id(const std::string& model_name) {
        return (model_name.size() > 6) ? model_name.substr(6) : model_name;
    }

    // Resolves <Database root>/<model_id>. Root is taken from, in order:
    //   1. <database_dir> in the plugin's SDF block
    //   2. MUBOT_DATABASE_DIR environment variable
    //   3. <share dir of des_mubot>/Database  (if ament_index_cpp is available)
    inline std::string Resolve_Database_Path(const std::shared_ptr<const sdf::Element>& sdf,
                                             const std::string& model_name) {
        std::string root;
        if (sdf && sdf->HasElement("database_dir")) {
            root = sdf->Get<std::string>("database_dir");
        }
        if (root.empty()) {
            if (const char* env = std::getenv("MUBOT_DATABASE_DIR")) root = env;
        }

        #if MUBOT_HAVE_AMENT
            if (root.empty()) {
                root = ament_index_cpp::get_package_share_directory("des_mubot") + "/Database";
            }
        #endif
        if (root.empty()) {
            throw std::runtime_error(
                "Cannot locate Database directory: set <database_dir> in the plugin SDF "
                "or the MUBOT_DATABASE_DIR environment variable.");
        }
        while (root.size() > 1 && root.back() == '/') root.pop_back();
        return root + "/" + Model_Id(model_name);
    }
}  // namespace mubot
