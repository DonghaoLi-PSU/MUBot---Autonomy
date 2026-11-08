/* Plugin for MUBot Magnetic-electric Actuator & Spring  (Gazebo Harmonic / gz-sim8)
 * 
 * Joints 0..s_num-3 contain an active actuator and a passive spring
 * Joint  s_num-2    contains a passive spring only
 *
 * To increase accuracy, the density of fluid and robot are amplified 1k times, 
 * so applying 'scaling'for torque output
 * 
 * Original (Gazebo Classic) by Donghao Li, March 23rd 2021.
 * Ported to a gz-sim System plugin (ISystemConfigure + ISystemPreUpdate).
 */

#include "mubot_sim/mubot_common.hh"

#include <gz/common/Console.hh>
#include <gz/plugin/Register.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/Joint.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Types.hh>

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

namespace mubot
{
namespace
{
    constexpr double max_power = 10000.0;  // controlled by voltage; adjust as needed
    constexpr double max_Theta = 0.79;
    constexpr double max_F     = 20.0;

    constexpr double ALPHA_LIM = max_Theta * 4.0 * Pi * Pi * max_F * max_F;  // max torque for actuator (20Hz oscillation)

    // Clamp max torque at actuator
    inline void AccelClampTorque(double izz_eff, double tz_raw, double& tz_out, double alpha_lim)
    {
        constexpr double IZZ_EPS = 1e-12;
        izz_eff = std::max(izz_eff, IZZ_EPS);
        tz_out = Clamp(Sanitize(tz_raw), alpha_lim * izz_eff);
        constexpr double T_HARD = 1e10;
        tz_out = Clamp(tz_out, T_HARD);  // hard fuse
    }
}  // anonymous namespace

class ActuatorPlugin
    : public gz::sim::System,
      public gz::sim::ISystemConfigure,
      public gz::sim::ISystemPreUpdate
{
public:
    ActuatorPlugin() = default;

    ~ActuatorPlugin() override {
        if (act_out_.is_open())  act_out_.close();
        if (volt_out_.is_open()) volt_out_.close();
    }

    void Configure(const gz::sim::Entity& _entity,
                   const std::shared_ptr<const sdf::Element>& _sdf,
                   gz::sim::EntityComponentManager& _ecm,
                   gz::sim::EventManager& /*_eventMgr*/) override
    {
        try {
            model_ = gz::sim::Model(_entity);
            if (!model_.Valid(_ecm)) {
                throw std::runtime_error("[Actuator] Configure(): entity is not a valid model.");
            }

            // Joint stiffness contained in sdf, no plugin anymore.             
            if (_sdf) {
                explicit_spring_ = _sdf->Get<bool>("explicit_spring", false).first;
            }

            path_ = Resolve_Database_Path(_sdf, model_.Name(_ecm));

            // Load joint_param.csv and enforce name match
            Load_Joint_Params(_ecm, path_ + "/joint_param.csv");

            // Load control_signal.csv (one line)
            Load_Control_Signal(path_ + "/control_signal.csv");

            configured_ = true;
            gzerr << "[Actuator] CSV and URDF matched" << std::endl; // One time check
        }
        catch (const std::exception& e) {
            gzerr << e.what() << " -- Actuator plugin disabled." << std::endl;
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

        const size_t N = joints_.size();
        if (N == 0) return;

        std::fill(pos_.begin(), pos_.end(), 0.0);
        std::fill(vel_.begin(), vel_.end(), 0.0);
        std::fill(voltage_.begin(), voltage_.end(), 0.0);
        std::fill(torque_rec_.begin(), torque_rec_.end(), 0.0);
        std::fill(torque_act_.begin(), torque_act_.end(), 0.0);
        std::fill(torque_spring_.begin(), torque_spring_.end(), 0.0);

        // compute and apply actuator torques
        for (size_t i = 0; i < N; ++i) {
            auto& j = joints_[i];
            const auto p = j.Position(_ecm);
            const auto v = j.Velocity(_ecm);
            pos_[i] = (p && !p->empty()) ? (*p)[0] : 0.0;
            vel_[i] = (v && !v->empty()) ? (*v)[0] : 0.0;

            const double volt = amp_[i] * std::sin(2.0 * Pi * (frequency_ * simu_time + phase_[i]));
            voltage_[i] = Clamp(volt, 15.0);   // clamp voltage

            const double ac = joint_params_[i].ac_param;

            torque_act_[i] = ((voltage_[i] - ac * vel_[i]) / 90.0) * ac * scaling;
            torque_spring_[i] = explicit_spring_ ? (-joint_params_[i].sf_param * pos_[i]) : 0.0; //either in sdf or plugin

            const double torque_raw = torque_act_[i] + torque_spring_[i];
            double torque_clamped = 0.0;
            AccelClampTorque(izz_eff_[i], torque_raw, torque_clamped, ALPHA_LIM);
            torque_rec_[i] = torque_clamped;   // for logging
            j.SetForce(_ecm, {torque_clamped});
        }

        // recording after record_step
        if (kRecord && simu_step >= record_step) {
            Files_Exists();

            act_out_ << simu_time;
            for (size_t i = 0; i < N; ++i) {
                act_out_ << "," << torque_act_[i] << "," << torque_spring_[i] << "," << torque_rec_[i];
            }
            act_out_ << "\n";

            volt_out_ << simu_time;
            for (size_t i = 0; i < N; ++i) {
                volt_out_ << "," << voltage_[i];
            }
            volt_out_ << "\n";
        }
    }

private:
    // joint_param.csv row: name, sf_param, ac_param, stiff_ratio
    struct JointParam {
        double ac_param = 0.0;
        double sf_param = 0.0;
        double stiff_ratio = 0.0;
    };

    //  set joints_ (Joints are looked up by name).
    void Load_Joint_Params(gz::sim::EntityComponentManager& ecm, const std::string& csv_path) {
        const auto model_joints = model_.Joints(ecm);
        if (model_joints.empty()) {
            throw std::runtime_error("[Actuator] Model has 0 joints.");
        }

        const auto rows = Read_CSV_Rows(csv_path, 4, "[Actuator]");

        // Row count must match joint count exactly
        if (rows.size() != model_joints.size()) {
            throw std::runtime_error("[Actuator] CSV/URDF mismatch: joint_param.csv rows = " +
                                     std::to_string(rows.size()) +
                                     ", model joints = " + std::to_string(model_joints.size()) +
                                     ".");
        }

        joints_.clear();
        joint_names_.clear();
        joint_params_.clear();
        izz_eff_.clear();

        std::unordered_set<std::string> csv_seen;
        for (const auto& row : rows) {
            const auto& c = row.cols;
            const int line_no = row.line_no;
            const std::string name = c[0]; //joint name
            if (name.empty()) {
                throw std::runtime_error("[Actuator] CSV format error: empty joint name at line " +
                                         std::to_string(line_no));
            }
            if (!csv_seen.insert(name).second) {
                throw std::runtime_error("[Actuator] CSV format error: duplicate joint name '" + name +
                                         "' at line " + std::to_string(line_no));
            }

            const gz::sim::Entity je = model_.JointByName(ecm, name);
            if (je == gz::sim::kNullEntity) {
                throw std::runtime_error("[Actuator] CSV/URDF mismatch: csv joint '" + name +
                                         "' (line " + std::to_string(line_no) +
                                         ") not found in model.");
            }

            JointParam p;
            p.sf_param    = Parse_Double(c[1], line_no, 1, "[Actuator]");
            p.ac_param    = Parse_Double(c[2], line_no, 2, "[Actuator]");
            p.stiff_ratio = Parse_Double(c[3], line_no, 3, "[Actuator]");

            gz::sim::Joint joint(je);
            joint.EnablePositionCheck(ecm, true);
            joint.EnableVelocityCheck(ecm, true);

            // Getting min Izz for torque clamp using min izz
            const auto parent_name = joint.ParentLinkName(ecm);
            const auto child_name  = joint.ChildLinkName(ecm);
            const gz::sim::Entity pe = parent_name ? model_.LinkByName(ecm, *parent_name)
                                                   : gz::sim::kNullEntity;
            const gz::sim::Entity ce = child_name ? model_.LinkByName(ecm, *child_name)
                                                  : gz::sim::kNullEntity;
            if (pe == gz::sim::kNullEntity || ce == gz::sim::kNullEntity) {
                throw std::runtime_error("[Actuator] Joint '" + name + "' missing parent/child link.");
            }

            double mass_parent = 0.0, izz_parent = 0.0, mass_child = 0.0, izz_child = 0.0;
            //  izz_parent <- parent, izz_child <- child.
            if (!Get_Link_Mass_Izz(ecm, pe, mass_parent, izz_parent)) izz_parent = 0.0;
            if (!Get_Link_Mass_Izz(ecm, ce, mass_child, izz_child)) izz_child = 0.0;
            const double izz_eff = std::min(izz_child, izz_parent);
            if (!std::isfinite(izz_eff)) {
                throw std::runtime_error("[Actuator] Joint '" + name +
                                         "' has invalid moment of inertia for acceleration clamp.");
            }

            joints_.push_back(joint);
            joint_names_.push_back(name);
            joint_params_.push_back(p);
            izz_eff_.push_back(izz_eff);
        }

        // Prepare arrays sized exactly to joint count
        const size_t N = joints_.size();
        amp_.assign(N, 0.0);        phase_.assign(N, 0.0);   voltage_.assign(N, 0.0);
        pos_.assign(N, 0.0);        vel_.assign(N, 0.0);
        torque_act_.assign(N, 0.0); torque_spring_.assign(N, 0.0); torque_rec_.assign(N, 0.0);
    }

    // Getting amp_, phase_ and frequency_ 
    void Load_Control_Signal(const std::string& csv_path) {
        const size_t N = joints_.size();
        const size_t expected = 2 * N; //N amp, N-1 phase, 1 frequency
        const auto raw = Read_Control_Signal(csv_path, expected, "[Actuator]");

        // Packing:
        // amp[0] = abs(raw[0]); phase[0] = 0;
        // for i>=1: amp[i] = abs(raw[2*i-1]); phase[i] = raw[2*i];
        // frequency = raw[2*N - 1]
        amp_[0] = std::fabs(raw[0]);
        phase_[0] = 0.0;

        double amp_sum = amp_[0];
        for (size_t i = 1; i < N; ++i) {
            amp_[i] = std::fabs(raw[2 * i - 1]);
            phase_[i] = raw[2 * i];
            amp_sum += amp_[i];
        }
        frequency_ = std::fabs(raw[2 * N - 1]);

        // Power limiting: turn down all amp at same ratio
        if (amp_sum > max_power) {
            const double ratio = max_power / amp_sum;
            for (size_t i = 0; i < N; ++i) amp_[i] *= ratio;
        }

        if (!std::isfinite(frequency_) || frequency_ <= 0.0) {
            throw std::runtime_error("[Actuator] Invalid frequency in control_signal.csv: " +
                                     std::to_string(frequency_));
        }
    }

    void Files_Exists() {
        if (output_files_initialized_) return;
        act_out_.open(path_ + "/record/actuator_output.csv", std::ios::out);
        if (!act_out_.is_open())
            throw std::runtime_error("[Actuator] Cannot create actuator_output.csv");
        act_out_ << std::scientific << std::setprecision(6);
        volt_out_.open(path_ + "/record/voltage_output.csv", std::ios::out);
        if (!volt_out_.is_open())
            throw std::runtime_error("[Actuator] Cannot create voltage_output.csv");
        volt_out_ << std::scientific << std::setprecision(6);
        output_files_initialized_ = true;
    }

    // Set true to enable actuator_output.csv & voltage_output.csv logging
    static constexpr bool kRecord = true;

    gz::sim::Model model_{gz::sim::kNullEntity};
    bool configured_ = false;
    bool explicit_spring_ = false;

    std::string path_;

    double step_size_ = 0.001;
    const double record_time_ = 4.0;
    const double stop_time_   = 6.1;
    const double scaling      = 1000.0; 

    bool output_files_initialized_ = false;

    // Joints, in joint_param.csv order
    std::vector<gz::sim::Joint> joints_;
    std::vector<std::string> joint_names_;
    std::vector<JointParam> joint_params_;
    std::vector<double> izz_eff_;   

    // Output files
    std::ofstream act_out_;
    std::ofstream volt_out_;

    // Control signals
    double frequency_ = 0.0;
    std::vector<double> amp_;
    std::vector<double> phase_;

    // Runtime / logging
    std::vector<double> voltage_;
    std::vector<double> pos_;
    std::vector<double> vel_;
    std::vector<double> torque_act_;
    std::vector<double> torque_rec_;
    std::vector<double> torque_spring_;
};
}  // namespace mubot

GZ_ADD_PLUGIN(mubot::ActuatorPlugin,
              gz::sim::System,
              mubot::ActuatorPlugin::ISystemConfigure,
              mubot::ActuatorPlugin::ISystemPreUpdate)

GZ_ADD_PLUGIN_ALIAS(mubot::ActuatorPlugin, "mubot::ActuatorPlugin")
