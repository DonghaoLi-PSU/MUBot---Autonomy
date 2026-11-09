/* Hydrodynamics Force Plugin for MUBot Project
 * Gazebo Harmonic (gz-sim8) system plugin, with rollout drift metrics.

 * To increase accuracy, the density of fluid and robot are amplified 1000 times, 
 * so applying 'scaling'for torque output
 *
 * Created by Donghao Li
 * Ported to gz-sim (ISystemConfigure + ISystemPreUpdate).
 * 
 * Hydrodynamic plugin is paused for first step to let simulator reset link acc
 */

#include "mubot_sim/mubot_common.hh"

#include <gz/common/Console.hh>
#include <gz/math/Pose3.hh>
#include <gz/math/Vector3.hh>
#include <gz/plugin/Register.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Types.hh>
#include <gz/sim/EntityComponentManager.hh>

#include <sdf/Element.hh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>
#include <cstdio>

using gz::math::Pose3d;
using gz::math::Vector3d;

namespace mubot
{
namespace
{
    constexpr double max_Y     = 42e-3;
    constexpr double max_Theta = 0.79;

    constexpr double A_LIM     = max_Y     * 4.0 * Pi * Pi;
    constexpr double ALPHA_LIM = max_Theta * 4.0 * Pi * Pi;

    // Limit max force/torque output to avoid collapse
    inline void Wrench_Clamp(double m, double izz,
                             double fx_raw, double fy_raw, double tz_raw,
                             double a_lim, double alpha_lim, double frequency,
                             double& fx_out, double& fy_out, double& tz_out)
    {
        fx_raw = Sanitize(fx_raw);
        fy_raw = Sanitize(fy_raw);
        tz_raw = Sanitize(tz_raw);

        constexpr double M_EPS   = 1e-5;
        constexpr double IZZ_EPS = 1e-12;

        m   = std::max(m,   M_EPS);
        izz = std::max(izz, IZZ_EPS);

        // Max force  = max_y_acc*mass*f
        // Max torque = max_ang_acc*izz*f
        fx_out = Clamp(fx_raw, a_lim * m * frequency * 6.0 / 20.0); 
        fy_out = Clamp(fy_raw, a_lim * m * frequency * 6.0 / 20.0);
        tz_out = Clamp(tz_raw, alpha_lim * izz * frequency * 6.0 / 5.0);

        constexpr double F_HARD = 1e10;
        constexpr double T_HARD = 1e10;
        fx_out = Clamp(fx_out, F_HARD);
        fy_out = Clamp(fy_out, F_HARD);
        tz_out = Clamp(tz_out, T_HARD);
    }
}  // anonymous namespace

class HydroPlugin
    : public gz::sim::System,
      public gz::sim::ISystemConfigure,
      public gz::sim::ISystemReset,
      public gz::sim::ISystemPreUpdate
{
public:
    HydroPlugin() = default;

    ~HydroPlugin() override {
        if (force_out_.is_open())  force_out_.close();
        if (disp_out_.is_open())   disp_out_.close();
        if (metric_out_.is_open()) metric_out_.close();
    }

    void Configure(const gz::sim::Entity& _entity,
                   const std::shared_ptr<const sdf::Element>& _sdf,
                   gz::sim::EntityComponentManager& _ecm,
                   gz::sim::EventManager& /*_eventMgr*/) override
    {
        try {
            model_ = gz::sim::Model(_entity);
            if (!model_.Valid(_ecm)) {
                throw std::runtime_error("[Hydro] Configure(): entity is not a valid model.");
            }

            path_ = Resolve_Database_Path(_sdf, model_.Name(_ecm));

            Load_Hydro_params(_ecm, path_ + "/hydro_param.csv");
            Load_Control_Frequency(_ecm, path_ + "/control_signal.csv");
            Prepare_Eval_Window();
            Reset_Drift_Metrics();

            configured_ = true;
        }
        catch (const std::exception& e) {
            gzerr << e.what() << " -- Hydro plugin disabled." << std::endl;
            configured_ = false;
        }
    }

    void PreUpdate(const gz::sim::UpdateInfo& _info,
                   gz::sim::EntityComponentManager& _ecm) override
    {
        if (!configured_ || _info.paused) return;

        const double dt = std::chrono::duration<double>(_info.dt).count();
        if (dt > 0.0 && std::isfinite(dt)) step_size_ = dt;

        const double simu_time = std::chrono::duration<double>(_info.simTime).count();
        const int64_t simu_step   = Safe_Round_Step(simu_time, step_size_);
        const int64_t record_step = Safe_Round_Step(record_time_, step_size_);

        //-------------------------------------
        // Debugging
        //-------------------------------------
        // if (simu_step <=3){
        //     gzerr<<"after reset: "<<after_reset_<<std::endl;
        // }

        // if (simu_step % 2000 == 0) {
        //     gzerr << " Time: " << simu_time << std::endl;
        // }
        // if (simu_step == 20) {
        //     gzerr << "Hydro Start" << std::endl;
        // }

        const size_t N = links_.size();
        if (N == 0 || params_.size() != N) return;


        double tail_pressure = 0.0;

        std::fill(react_x_.begin(), react_x_.end(), 0.0); std::fill(react_y_.begin(), react_y_.end(), 0.0); std::fill(react_t_.begin(), react_t_.end(), 0.0);
        std::fill(resis_x_.begin(), resis_x_.end(), 0.0); std::fill(resis_y_.begin(), resis_y_.end(), 0.0); std::fill(resis_t_.begin(), resis_t_.end(), 0.0);
        std::fill(compe_x_.begin(), compe_x_.end(), 0.0); std::fill(compe_y_.begin(), compe_y_.end(), 0.0); std::fill(compe_t_.begin(), compe_t_.end(), 0.0);
        std::fill(recor_x_.begin(), recor_x_.end(), 0.0); std::fill(recor_y_.begin(), recor_y_.end(), 0.0); std::fill(recor_t_.begin(), recor_t_.end(), 0.0);

        //Calculating hydrodynamic force for segment
        for (size_t i = 0; i < N; ++i) {
            const HydroParam& P = params_[i];
            auto& link = links_[i];

            const auto pose = link.WorldPose(_ecm);
            const auto wlv  = link.WorldLinearVelocity(_ecm);
            const auto wla  = link.WorldLinearAcceleration(_ecm);
            const auto wav  = link.WorldAngularVelocity(_ecm);
            const auto waa  = link.WorldAngularAcceleration(_ecm);
            // Make sure all kinetic can be readed.
            if (!pose || !wlv || !wla || !wav || !waa) continue;

            WP_[i] = *pose;
            const double theta = WP_[i].Rot().Yaw();
            if (!(P.M_total > 1e-9)) continue;

            const Vector3d WLV = *wlv;
            // const Vector3d WLA = *wla;
            const Vector3d WAV = *wav;
            // const Vector3d WAA = *waa;

            const Vector3d WLA = after_reset_ ? Vector3d::Zero : *wla;
            const Vector3d WAA = after_reset_ ? Vector3d::Zero : *waa;

            const double vel_yaw = WAV.Z();
            const double acc_yaw = WAA.Z();

            const double vel_x_0 = WLV.X() * std::cos(theta) + WLV.Y() * std::sin(theta);
            const double vel_y_0 = WLV.Y() * std::cos(theta) - WLV.X() * std::sin(theta);
            const double vel_y_l = vel_y_0 + vel_yaw * P.lt;

            const double acc_x   = WLA.X() * std::cos(theta) + WLA.Y() * std::sin(theta);
            const double acc_y   = -WLA.X() * std::sin(theta) + WLA.Y() * std::cos(theta);
            const double acc_y_0 = acc_y - P.cog * acc_yaw;

            // Reactive force model
            react_x_[i] = Seg_Cm * ( +P.M_S     * std::pow(vel_yaw, 2)
                                     +P.M_total * vel_y_0 * vel_yaw );

            react_y_[i] = Seg_Cm * ( +P.M_total * vel_x_0 * vel_yaw
                                     -P.M_total * acc_y_0
                                     -P.M_S     * acc_yaw
                                     -P.M_l     * vel_x_0 * vel_y_l
                                     +P.M_0     * vel_x_0 * vel_y_0 );

            react_t_[i] = -Seg_Cm * ( +P.M_l * P.lt * vel_x_0 * vel_y_l
                                      +P.M_S * acc_y_0
                                      -P.M_S * vel_x_0 * vel_yaw );

            compe_x_[i] = Comp_ratio * P.M_total * acc_x;
            compe_y_[i] = Comp_ratio * P.M_total * acc_y;
            compe_t_[i] = Comp_ratio * P.M_total * acc_y * P.cog;
            
            // Pressure force is only applied to tail tip. 
            if (i == N - 1) {
                tail_pressure = -Seg_Cp * Seg_Cm * P.M_l * std::pow(vel_y_l, 2);
                react_x_[i] += tail_pressure;
            }

            // Resistive force: friction
            resis_x_[i] = (-0.5 * rho_f * c_f) * std::fabs(vel_x_0) * vel_x_0 * P.IP;

            // Resistive force: drag. integrated along the segment(consistant depth)
            double Seg_lc = 0.0;
            if (!Is_Zero(vel_yaw)) {
                Seg_lc = -vel_y_0 / vel_yaw;
            }

            const double s_vy0 = Sign_function(vel_y_0);
            if (Is_Zero(vel_yaw)) {
                resis_y_[i] = (-0.5 * rho_f * c_d) * std::fabs(vel_y_0) * vel_y_0 * P.IH0;
                resis_t_[i] = (-0.5 * rho_f * c_d) * std::fabs(vel_y_0) * vel_y_0 * P.IH1;
            } else if (Is_Zero(vel_y_0)) {
                resis_y_[i] = (-0.5 * rho_f * c_d) * std::fabs(vel_yaw) * vel_yaw * P.IH2;
                resis_t_[i] = (-0.5 * rho_f * c_d) * std::fabs(vel_yaw) * vel_yaw * P.IH3;
            } else if (Seg_lc < 0.0 || Seg_lc >= P.lt) {
                resis_y_[i] = (-0.5 * rho_f * c_d) * s_vy0 *
                              ( std::pow(vel_y_0,2) * P.IH0
                              + 2.0 * vel_yaw * vel_y_0 * P.IH1
                              + std::pow(vel_yaw,2) * P.IH2 );
                resis_t_[i] = (-0.5 * rho_f * c_d) * s_vy0 *
                              ( std::pow(vel_y_0,2) * P.IH1
                              + 2.0 * vel_yaw * vel_y_0 * P.IH2
                              + std::pow(vel_yaw,2) * P.IH3 );
            } else {
                resis_y_[i] = (-0.5 * rho_f * c_d * P.h) * s_vy0 *
                              ( std::pow(vel_y_0,2) * (2.0 * Seg_lc - P.lt)
                              + vel_y_0 * vel_yaw * (2.0 * std::pow(Seg_lc,2) - std::pow(P.lt,2))
                              + std::pow(vel_yaw,2) * (2.0 / 3.0 * std::pow(Seg_lc,3) - 1.0 / 3.0 * std::pow(P.lt,3)) );
                resis_t_[i] = (-0.5 * rho_f * c_d * P.h) * s_vy0 *
                              ( std::pow(vel_y_0,2) * (std::pow(Seg_lc,2) - 0.5 * std::pow(P.lt,2))
                              + vel_y_0 * vel_yaw * (4.0 / 3.0 * std::pow(Seg_lc,3) - 2.0 / 3.0 * std::pow(P.lt,3))
                              + std::pow(vel_yaw,2) * (0.5 * std::pow(Seg_lc,4) - 0.25 * std::pow(P.lt,4)) );
            }

            const double fx_raw = react_x_[i] + resis_x_[i] + compe_x_[i];
            const double fy_raw = react_y_[i] + resis_y_[i] + compe_y_[i];
            const double tz_raw = react_t_[i] + resis_t_[i] + compe_t_[i];

            // Prevent extrem external force
            double fx = 0.0, fy = 0.0, tz = 0.0;
            Wrench_Clamp(mass_[i], izz_[i], fx_raw, fy_raw, tz_raw,
                         A_LIM, ALPHA_LIM, frequency_, fx, fy, tz);
            recor_x_[i] = fx;
            recor_y_[i] = fy;
            recor_t_[i] = tz;
            
            const Vector3d f_world = WP_[i].Rot().RotateVector(Vector3d(fx, fy, 0.0));
            link.AddWorldWrench(_ecm, f_world, Vector3d(0.0, 0.0, tz),Vector3d(-P.cog,0,0));
        }

        // if (simu_step == 10) {                 // first step after reset = the state the rollout starts from
        //     std::ofstream out(path_ + "/init_state.csv", std::ios::out);
        //     if (out.is_open()) {
        //         out << std::scientific << std::setprecision(6);
        //         out << "link,x,y,yaw,vx,vy,wz,ax,ay,alpha_z\n";
        //         for (size_t i = 0; i < N; ++i) {
        //             const auto wlv = links_[i].WorldLinearVelocity(_ecm);
        //             const auto wla = links_[i].WorldLinearAcceleration(_ecm);
        //             const auto wav = links_[i].WorldAngularVelocity(_ecm);
        //             const auto waa = links_[i].WorldAngularAcceleration(_ecm);
        //             if (!wlv || !wla || !wav || !waa) continue;
        //             out << i << ","
        //                 << WP_[i].Pos().X() << "," << WP_[i].Pos().Y() << "," << WP_[i].Rot().Yaw() << ","
        //                 << wlv->X() << "," << wlv->Y() << "," << wav->Z() << ","
        //                 << wla->X() << "," << wla->Y() << "," << waa->Z() << "\n";
        //         }
        //     }
        // }

        after_reset_ = false;
        // MUBot state recording
        if ((simu_step >= record_step) && (simu_time <= eval_end_time_ + 0.5)) {
            Record_File_Create();

            disp_out_ << simu_time << ",";
            for (size_t i = 0; i < N; ++i) {
                disp_out_ << WP_[i].Pos().X() << "," << WP_[i].Pos().Y() << "," << WP_[i].Rot().Yaw() << ",";
            }
            const double y_tail = WP_[N - 1].Pos().Y() + params_[N - 1].lt * std::sin(WP_[N - 1].Rot().Yaw());
            const double x_tail = WP_[N - 1].Pos().X() + params_[N - 1].lt * std::cos(WP_[N - 1].Rot().Yaw());
            disp_out_ << x_tail << ',' << y_tail << "\n";

            force_out_ << simu_time << ",";
            for (size_t i = 0; i < N; ++i) {
                force_out_ << react_x_[i] << "," << resis_x_[i] << "," << compe_x_[i] << ",";
                force_out_ << react_y_[i] << "," << resis_y_[i] << "," << compe_y_[i] << ",";
                force_out_ << react_t_[i] << "," << resis_t_[i] << "," << compe_t_[i] << ",";
            }
            force_out_ << tail_pressure << "\n";

            Update_Drift_Metrics(_ecm, simu_time);
        }

        if (simu_time >= eval_end_time_) {
            Write_Drift_Summary_If_Needed(simu_time);
            const std::string term = path_ + "/terminate.txt";
            if (!File_Exists(term)) {
                std::ofstream(term, std::ios::out).close();
                if (disp_out_.is_open())  disp_out_.flush();
                if (force_out_.is_open()) force_out_.flush();
                gzmsg << "Hydro Terminate Created" << std::endl;
            }
        }
    }

    void Reset(const gz::sim::UpdateInfo& _info,
            gz::sim::EntityComponentManager& _ecm) override
    {
        try {
            // per-rollout files are re-created (truncated) on the next record
            for (auto* f : {&force_out_, &disp_out_, &metric_out_})
                if (f->is_open()) f->close();
            record_files_initialized_ = false;
            after_reset_ = true;

            std::remove((path_ + "/terminate.txt").c_str());

            Load_Control_Frequency(_ecm, path_ + "/control_signal.csv");
            Prepare_Eval_Window();
            Reset_Drift_Metrics();
            configured_ = true;

            gzmsg << "[Hydro] reset, sim time = "
                << std::chrono::duration<double>(_info.simTime).count() << std::endl;

            // ack for Python; written last, after everything above succeeded
            std::ofstream(path_ + "/reset_ack.txt", std::ios::out).close();
        }
        catch (const std::exception& e) {
            gzerr << e.what() << " -- Hydro plugin disabled." << std::endl;
            configured_ = false;
        }
    }
    private:
    struct HydroParam {
        double M_total = 0.0;
        double M_S     = 0.0;
        double M_0     = 0.0;
        double M_l     = 0.0;
        double cog     = 0.0;
        double lt      = 0.0;
        double h       = 0.0;
        double IH0     = 0.0;
        double IH1     = 0.0;
        double IH2     = 0.0;
        double IH3     = 0.0;
        double IP      = 0.0;
        // double cf      = 0.0;
        // double cd      = 0.0;
    };

    struct DriftMetrics {
        bool initialized = false;
        bool summary_written = false;
        size_t sample_count = 0;

        double start_time = 0.0;
        double end_time = 0.0;

        double x0 = 0.0;
        double y0 = 0.0;
        double cogvel_x_sum = 0.0;
        double lat_sq_sum = 0.0;
        double max_abs_lat = 0.0;

        double yaw0 = 0.0;
        double yaw_prev = 0.0;
        double yaw_last = 0.0;

        double forward_x = 1.0;
        double forward_y = 0.0;
        double left_x = 0.0;
        double left_y = 1.0;
    };

    // Links are looked up by name with same order of CSV
    void Load_Hydro_params(gz::sim::EntityComponentManager& ecm, const std::string& csv_path) {
        const auto model_links = model_.Links(ecm);
        if (model_links.empty()) {
            throw std::runtime_error("[Hydro] Plugin has 0 links.");
        }

        const auto rows = Read_CSV_Rows(csv_path, 13, "[Hydro]");

        if (rows.size() != model_links.size()) {
            throw std::runtime_error("[Hydro] CSV/URDF mismatch: CSV rows = " +
                                     std::to_string(rows.size()) +
                                     ", model links = " + std::to_string(model_links.size()) +
                                     ". They must match exactly.");
        }

        links_.clear();
        link_names_.clear();
        params_.clear();
        mass_.clear();
        izz_.clear();

        std::unordered_set<std::string> csv_seen;
        for (const auto& row : rows) {
            const auto& c = row.cols;
            const int line_no = row.line_no;
            const std::string name = c[0];
            if (name.empty()) {
                throw std::runtime_error("[Hydro] CSV format error: empty link name at line " +
                                         std::to_string(line_no));
            }
            if (!csv_seen.insert(name).second) {
                throw std::runtime_error("[Hydro] CSV format error: duplicate link name '" + name +
                                         "' at line " + std::to_string(line_no));
            }

            const gz::sim::Entity e = model_.LinkByName(ecm, name);
            if (e == gz::sim::kNullEntity) {
                throw std::runtime_error("[Hydro] CSV/URDF mismatch: csv link '" + name +
                                         "' (line " + std::to_string(line_no) +
                                         ") not found in model.");
            }

            HydroParam p;
            p.M_total = Parse_Double(c[1],  line_no, 1,  "[Hydro]");
            p.M_S     = Parse_Double(c[2],  line_no, 2,  "[Hydro]");
            p.M_0     = Parse_Double(c[3],  line_no, 3,  "[Hydro]");
            p.M_l     = Parse_Double(c[4],  line_no, 4,  "[Hydro]");
            p.cog     = Parse_Double(c[5],  line_no, 5,  "[Hydro]");
            p.lt      = Parse_Double(c[6],  line_no, 6,  "[Hydro]");
            p.h       = Parse_Double(c[7],  line_no, 7,  "[Hydro]");
            p.IH0     = Parse_Double(c[8],  line_no, 8,  "[Hydro]");
            p.IH1     = Parse_Double(c[9],  line_no, 9,  "[Hydro]");
            p.IH2     = Parse_Double(c[10], line_no, 10, "[Hydro]");
            p.IH3     = Parse_Double(c[11], line_no, 11, "[Hydro]");
            p.IP      = Parse_Double(c[12], line_no, 12, "[Hydro]");
            // p.cf      = Parse_Double(c[13], line_no, 13, "[Hydro]");
            // p.cd      = Parse_Double(c[14], line_no, 14, "[Hydro]");

            double m = 0.0, izz = 0.0;
            if (!Get_Link_Mass_Izz(ecm, e, m, izz) || !std::isfinite(m) || !std::isfinite(izz)) {
                throw std::runtime_error("[Hydro] Link '" + name +
                                         "' has invalid mass/inertia for acceleration clamp.");
            }

            gz::sim::Link link(e);
            link.EnableVelocityChecks(ecm, true);
            link.EnableAccelerationChecks(ecm, true);

            links_.push_back(link);
            link_names_.push_back(name);
            params_.push_back(p);
            mass_.push_back(m);
            izz_.push_back(izz);
        }

        const size_t N = links_.size();
        WP_.assign(N, Pose3d());
        react_x_.assign(N, 0.0); react_y_.assign(N, 0.0); react_t_.assign(N, 0.0);
        resis_x_.assign(N, 0.0); resis_y_.assign(N, 0.0); resis_t_.assign(N, 0.0);
        compe_x_.assign(N, 0.0); compe_y_.assign(N, 0.0); compe_t_.assign(N, 0.0);
        recor_x_.assign(N, 0.0); recor_y_.assign(N, 0.0); recor_t_.assign(N, 0.0);

        gzerr << "[Hydro] CSV and URDF matched" << std::endl;
    }

    void Load_Control_Frequency(gz::sim::EntityComponentManager& ecm, const std::string& csv_path) {
        const size_t N = model_.Joints(ecm).size();
        if (N == 0) {
            throw std::runtime_error("[Hydro] Model has 0 joints.");
        }
        const auto raw = Read_Control_Signal(csv_path, 2 * N, "[Hydro]");
        frequency_ = std::fabs(raw[2 * N - 1]);
        if (!std::isfinite(frequency_) || frequency_ <= 0.0) {
            throw std::runtime_error("[Hydro] Invalid frequency in control_signal.csv: " +
                                     std::to_string(frequency_));
        }
    }

    // Getting full oscillation cycle
    void Prepare_Eval_Window() {
        eval_period_ = 1.0 / frequency_;
        const double available = terminate_time_ - record_time_;
        const int n_cycles = static_cast<int>(std::floor(available / eval_period_));
        if (n_cycles < 1) {
            throw std::runtime_error("[Hydro] No full oscillation period inside [record_time, terminate_time].");
        }
        eval_end_time_ = record_time_ + static_cast<double>(n_cycles) * eval_period_;
        gzmsg << "[Hydro] eval window: [" << record_time_ << ", " << eval_end_time_
              << "]  period=" << eval_period_ << "  cycles=" << n_cycles << std::endl;
    }

    void Reset_Drift_Metrics() {
        drift_ = DriftMetrics();
    }

    void Record_File_Create() {
        if (record_files_initialized_) return;

        metric_out_.open(path_ + "/record/drift_metrics.csv", std::ios::out);
        if (!metric_out_.is_open())
            throw std::runtime_error("[Hydro] Cannot create drift_metrics.csv");
        metric_out_ << std::scientific << std::setprecision(6);
        metric_out_ << "CoGVel_x,abs_lateral,rms_lat,heading_drift\n";

        disp_out_.open(path_ + "/record/displace_record.csv", std::ios::out);
        if (!disp_out_.is_open())
            throw std::runtime_error("[Hydro] Cannot create displace_record.csv");
        disp_out_ << std::scientific << std::setprecision(6);

        force_out_.open(path_ + "/record/force_record.csv", std::ios::out);
        if (!force_out_.is_open())
            throw std::runtime_error("[Hydro] Cannot create force_record.csv");
        force_out_ << std::scientific << std::setprecision(6);

        gzerr << "Hydro Recording Start" << std::endl;
        record_files_initialized_ = true;
    }

    bool Compute_Mass_Weighted_State(const gz::sim::EntityComponentManager& ecm,
                                     double& xc, double& yc, double& cogvel_x,
                                     double& head_x, double& head_y,
                                     double& tail_x, double& tail_y,
                                     double& mean_yaw) const
    {
        const size_t N = links_.size();
        if (N == 0) return false;

        double total_mass = 0.0;
        xc = 0.0;
        yc = 0.0;
        cogvel_x = 0.0;
        mean_yaw = 0.0;

        for (size_t i = 0; i < N; ++i) {
            const double m = mass_[i];
            if (!(m > 0.0) || !std::isfinite(m)) return false;

            const auto pose = links_[i].WorldPose(ecm);
            const auto vel  = links_[i].WorldLinearVelocity(ecm);
            if (!pose || !vel) return false;

            xc += pose->Pos().X() * m;
            yc += pose->Pos().Y() * m;
            cogvel_x += vel->X() * m;
            mean_yaw += pose->Rot().Yaw() * m;
            total_mass += m;
        }

        if (!(total_mass > 1e-12)) return false;

        xc /= total_mass;
        yc /= total_mass;
        cogvel_x /= total_mass;
        mean_yaw /= total_mass;

        const auto head = links_.front().WorldPose(ecm);
        const auto tail = links_.back().WorldPose(ecm);
        if (!head || !tail) return false;
        head_x = head->Pos().X();
        head_y = head->Pos().Y();
        tail_x = tail->Pos().X();
        tail_y = tail->Pos().Y();
        return true;
    }

    void Update_Drift_Metrics(const gz::sim::EntityComponentManager& ecm, double simu_time) {
        if (simu_time < record_time_ || simu_time > eval_end_time_) return;

        double xc = 0.0, yc = 0.0, cogvel_x = 0.0;
        double head_x = 0.0, head_y = 0.0, tail_x = 0.0, tail_y = 0.0, mean_yaw = 0.0;
        if (!Compute_Mass_Weighted_State(ecm, xc, yc, cogvel_x, head_x, head_y, tail_x, tail_y, mean_yaw)) {
            return;
        }

        if (!drift_.initialized) {
            drift_.initialized = true;
            drift_.sample_count = 0;
            drift_.start_time = simu_time;
            drift_.end_time = simu_time;
            drift_.x0 = xc;
            drift_.y0 = yc;
            drift_.cogvel_x_sum = 0.0;
            drift_.lat_sq_sum = 0.0;
            drift_.max_abs_lat = 0.0;

            const double dx = head_x - tail_x;
            const double dy = head_y - tail_y;
            const double norm = std::sqrt(dx * dx + dy * dy);
            if (norm > 1e-12) {
                drift_.forward_x = dx / norm;
                drift_.forward_y = dy / norm;
            } else {
                drift_.forward_x = 1.0;
                drift_.forward_y = 0.0;
            }
            drift_.left_x = -drift_.forward_y;
            drift_.left_y =  drift_.forward_x;

            drift_.yaw0 = mean_yaw;
            drift_.yaw_prev = mean_yaw;
            drift_.yaw_last = mean_yaw;
        }

        mean_yaw = Unwrap_To_Near(mean_yaw, drift_.yaw_prev);
        drift_.yaw_prev = mean_yaw;
        drift_.yaw_last = mean_yaw;
        drift_.end_time = simu_time;

        const double rel_x = xc - drift_.x0;
        const double rel_y = yc - drift_.y0;
        const double lat = rel_x * drift_.left_x + rel_y * drift_.left_y;

        drift_.cogvel_x_sum += cogvel_x;
        drift_.lat_sq_sum += lat * lat;
        drift_.max_abs_lat = std::max(drift_.max_abs_lat, std::abs(lat));
        drift_.sample_count += 1;
    }

    void Write_Drift_Summary_If_Needed(double simu_time) {
        if (!record_files_initialized_ || drift_.summary_written || simu_time < eval_end_time_) return;

        drift_.summary_written = true;

        double mean_cogvel_x = 0.0;
        double rms_lat = 0.0;
        double heading_drift = 0.0;
        double abs_lateral = 0.0;

        if (drift_.initialized && drift_.sample_count > 0) {
            mean_cogvel_x = drift_.cogvel_x_sum / static_cast<double>(drift_.sample_count);
            rms_lat = std::sqrt(drift_.lat_sq_sum / static_cast<double>(drift_.sample_count));
            heading_drift = std::abs(drift_.yaw_last - drift_.yaw0);
            abs_lateral = drift_.max_abs_lat;
        }
        gzmsg << "Drift Summary created" << std::endl;

        metric_out_ << mean_cogvel_x << ","
                    << abs_lateral << ","
                    << rms_lat << ","
                    << heading_drift << "\n";
        metric_out_.flush();
    }

private:
    gz::sim::Model model_{gz::sim::kNullEntity};
    bool configured_ = false;

    double frequency_ = 10.0;
    double step_size_ = 0.001;
    const double record_time_ = 4.0;
    const double terminate_time_ = 6.1;
    
    // Hydrodynamic constants
    static constexpr double scale_factor = 1000.0;
    static constexpr double rho_f = 1000.0 * scale_factor;
    static constexpr double Comp_ratio = 0.5;
    static constexpr double c_d = 2.25;
    static constexpr double c_f = 0.06;
    static constexpr double Seg_Cm = 1.0;
    static constexpr double Seg_Cp = 1.0;
    double eval_period_ = 0.0;
    double eval_end_time_ = 0.0;

    std::string path_;
    bool record_files_initialized_ = false;

    std::vector<gz::sim::Link> links_;
    std::vector<std::string> link_names_;
    std::vector<HydroParam> params_;
    std::vector<double> mass_, izz_;   // cached inertial properties per link

    std::vector<Pose3d> WP_;
    std::vector<double> react_x_, react_y_, react_t_;
    std::vector<double> resis_x_, resis_y_, resis_t_;
    std::vector<double> compe_x_, compe_y_, compe_t_;
    std::vector<double> recor_x_, recor_y_, recor_t_;

    DriftMetrics drift_;

    std::ofstream force_out_;
    std::ofstream disp_out_;
    std::ofstream metric_out_;

    bool after_reset_ = false;
};
}  // namespace mubot

GZ_ADD_PLUGIN(mubot::HydroPlugin,
              gz::sim::System,
              mubot::HydroPlugin::ISystemConfigure,
              mubot::HydroPlugin::ISystemPreUpdate,
              mubot::HydroPlugin::ISystemReset)

GZ_ADD_PLUGIN_ALIAS(mubot::HydroPlugin, "mubot::HydroPlugin")
