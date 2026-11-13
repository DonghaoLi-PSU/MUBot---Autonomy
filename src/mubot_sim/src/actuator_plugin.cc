/* Actuator plugin for the MUBot magnetic-electric actuators and joint springs
 * (Gazebo Harmonic / gz-sim8 System plugin: Configure, Reset, PreUpdate).
 *
 * Every joint has an actuator. Each joint also has a passive spring, either applied here
 * (<explicit_spring>true</explicit_spring>) or defined in the model file
 * (<axis><dynamics><spring_stiffness>), in which case it must be false here.
 *
 * The density of the fluid and of the robot is amplified 1000x for numerical accuracy,
 * so the torques are multiplied by 'scaling'. The actuator torque is
 *      tau = (V - ac * w) / 90 * ac * scaling
 * and, together with the optional explicit spring torque -sf * q * scaling, is clamped by a
 * maximum angular acceleration (AccelClampTorque) before it is applied with SetForce.
 *
 * Parameters: Database/<id>/joint_param.csv, one row per joint:
 *      joint_name, sf_param (spring), ac_param (actuator), stiff_ratio
 *
 * Plugin block options (all optional)
 *      <database_dir>       folder that holds Database/<id>/
 *      <explicit_spring>    apply the joint spring here (default false)
 *      <external_control>   drive the voltages from Python (default false)
 *      <idle_timeout_s>     external mode: the sim stops waiting for a request after this long (30)
 *      <reply_timeout_s>    external mode: the service gives up waiting for the sim after this long (10)
 *
 * Two control modes
 * -----------------
 * 1) Open loop (default)
 *      V_i = amp_i * sin(2*pi*(f*t + phase_i)), parameters from Database/<id>/control_signal.csv
 *      (packing: amp_0, then (amp_i, phase_i) for i = 1..N-1, then the frequency).
 *
 * 2) External lock-step control (<external_control>true</external_control>)
 *      A gz-transport service  /model/<model_name>/step  (gz.msgs.Double_V -> gz.msgs.Double_V)
 *      drives the simulation. The sim thread blocks inside PreUpdate whenever it has no steps
 *      left, so the world cannot advance beyond the steps that were requested.
 *
 *      Request  [ n_steps, v... ]
 *                 len(v) == N          the voltage vector is held for all n_steps
 *                 len(v) == N*n_steps  step-major sequence, v[k*N + i] = joint i at step k
 *                 n_steps == 0         advance nothing, only return the current observation
 *                 n_steps == -1        RELEASE: stop waiting for requests (voltage 0). Call it before
 *                                      pausing or resetting the world, because a waiting sim thread
 *                                      cannot process world-control messages. A world reset re-arms it.
 *      Reply    [ status, sim_time, iteration,
 *                 joint_pos[N], joint_vel[N],
 *                 head_x, head_y, head_yaw, head_vx, head_vy, head_wz,
 *                 cog_x, cog_y, cogvel_x, cogvel_y, cog_yaw,     (mass-weighted over the links)
 *                 tail_x, tail_y ]                                (origin of the last link)
 *                 length = 3 + 2N + 13
 *                 status: 0 ok, 1 invalid request, 2 busy, 3 aborted by a world reset,
 *                         4 simulation is paused (unpause first; the service answers at once)
 *                 The observation is the state after the last requested step, and `iteration`
 *                 advances by exactly n_steps between two consecutive replies.
 *      The service call fails (no reply) if the sim does not answer within <reply_timeout_s>.
 *
 * Reset: closes the debug files, re-arms the step gate (external mode) or re-reads
 * control_signal.csv (open loop), and sets the joint positions and velocities to zero.
 * Debug files (record/actuator_output.csv, record/voltage_output.csv) are written from t = 4.0 s.
 *
 * Original (Gazebo Classic) by Donghao Li, March 23rd 2021.
 * Ported to a gz-sim System plugin.
 */

#include "mubot_sim/mubot_common.hh"

#include <gz/common/Console.hh>
#include <gz/math/Pose3.hh>
#include <gz/math/Vector3.hh>
#include <gz/msgs/double_v.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/Joint.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Types.hh>
#include <gz/transport/Node.hh>
#include <gz/sim/components/Inertial.hh>

#include <sdf/Element.hh>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <stdexcept>
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
      public gz::sim::ISystemReset,
      public gz::sim::ISystemPreUpdate
{
public:
    ActuatorPlugin() = default;

    ~ActuatorPlugin() override {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            stop_ = true;
        }
        cv_.notify_all();
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
                explicit_spring_  = _sdf->Get<bool>("explicit_spring", false).first;
                external_control_ = _sdf->Get<bool>("external_control", false).first;
                gzerr<<"Plugin Spring: "<< explicit_spring_<<" External control: "<< external_control_<<std::endl;
                idle_timeout_  = std::chrono::milliseconds(static_cast<int64_t>(
                    1000.0 * _sdf->Get<double>("idle_timeout_s", 30.0).first));
                reply_timeout_ = std::chrono::milliseconds(static_cast<int64_t>(
                    1000.0 * _sdf->Get<double>("reply_timeout_s", 10.0).first));
            }

            path_ = Resolve_Database_Path(_sdf, model_.Name(_ecm));

            // Load joint_param.csv and enforce name match
            Load_Joint_Params(_ecm, path_ + "/joint_param.csv");

            if (external_control_) {
                Setup_External_Control(_ecm);
            } else {
                // Load control_signal.csv (one line)
                Load_Control_Signal(path_ + "/control_signal.csv");
            }

            configured_ = true;
            gzerr << "[Actuator] CSV and URDF matched" << std::endl; // One time check
        }
        catch (const std::exception& e) {
            gzerr << e.what() << " -- Actuator plugin disabled." << std::endl;
            configured_ = false;
        }
    }

    void Reset(const gz::sim::UpdateInfo& /*_info*/,
               gz::sim::EntityComponentManager& _ecm) override
    {
        try {
            if (act_out_.is_open())  act_out_.close();
            if (volt_out_.is_open()) volt_out_.close();
            output_files_initialized_ = false;

            if (external_control_) {
                std::lock_guard<std::mutex> lk(mtx_);
                steps_left_ = 0;
                ext_idx_ = 0;
                has_request_ = false;
                release_ = false;
                gave_up_ = false;                // re-arm the gate
                std::fill(ext_now_.begin(), ext_now_.end(), 0.0);
                if (waiting_reply_) {            // a request was in flight: tell the client
                    reply_.Clear();
                    reply_.add_data(3.0);
                    waiting_reply_ = false;
                    reply_ready_ = true;
                    cv_.notify_all();
                }
            } else {
                Load_Control_Signal(path_ + "/control_signal.csv");
            }

            for (auto& j : joints_) {
                j.ResetPosition(_ecm, {0.0});
                j.ResetVelocity(_ecm, {0.0});
            }
            configured_ = true;
        }
        catch (const std::exception& e) {
            gzerr << e.what() << " -- Actuator plugin disabled." << std::endl;
            configured_ = false;
        }
    }

    void PreUpdate(const gz::sim::UpdateInfo& _info,
                   gz::sim::EntityComponentManager& _ecm) override
    {
        paused_.store(_info.paused);
        if (!configured_ || _info.paused) return;

        const size_t N = joints_.size();
        if (N == 0) return;

        if (external_control_) Gate_Step(_ecm, _info);

        const double dt = std::chrono::duration<double>(_info.dt).count();
        if (dt > 0.0 && std::isfinite(dt)) step_size_ = dt;

        const double simu_time = std::chrono::duration<double>(_info.simTime).count();
        const int64_t simu_step   = Safe_Round_Step(simu_time, step_size_);
        const int64_t record_step = Safe_Round_Step(record_time_, step_size_);

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

            const double volt = external_control_
                ? ext_now_[i]
                : amp_[i] * std::sin(2.0 * Pi * (frequency_ * simu_time + phase_[i]));
            voltage_[i] = Clamp(Sanitize(volt), 15.0);   // clamp voltage

            const double ac = joint_params_[i].ac_param;

            torque_act_[i] = ((voltage_[i] - ac * vel_[i]) / 90.0) * ac * scaling;
            torque_spring_[i] = explicit_spring_ ? (-joint_params_[i].sf_param * pos_[i]) * scaling : 0.0; //either in sdf or plugin

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

    // ------------------------------------------------------------------
    // external step control
    // ------------------------------------------------------------------
    void Setup_External_Control(gz::sim::EntityComponentManager& ecm) {
        const size_t N = joints_.size();
        ext_now_.assign(N, 0.0);

        // "head" for the observation 
        const auto parent = joints_[0].ParentLinkName(ecm);
        if (parent) {
            const gz::sim::Entity he = model_.LinkByName(ecm, *parent);
            if (he != gz::sim::kNullEntity) {
                head_ = gz::sim::Link(he);
                head_.EnableVelocityChecks(ecm, true);
                have_head_ = true;
            }
        }

        auto add_link = [&](const std::string& lname) {
            const gz::sim::Entity e = model_.LinkByName(ecm, lname);
            if (e == gz::sim::kNullEntity) return;
            gz::sim::Link l(e);
            l.EnableVelocityChecks(ecm, true);
            const auto* in = ecm.Component<gz::sim::components::Inertial>(e);
            chain_.push_back(l);
            chain_mass_.push_back(in ? in->Data().MassMatrix().Mass() : 0.0);
        };
        chain_.clear(); chain_mass_.clear();
        if (parent) add_link(*parent);
        for (size_t i = 0; i < N; ++i) {
            const auto c = joints_[i].ChildLinkName(ecm);
            if (c) add_link(*c);
        }
        for (size_t i = 0; i < chain_mass_.size(); ++i)
        {
            gzmsg << "[Actuator] chain link " << i << " mass " << chain_mass_[i] << std::endl;
        }

        const std::string topic = "/model/" + model_.Name(ecm) + "/step";
        if (!node_.Advertise(topic, &ActuatorPlugin::On_Step, this)) {
            throw std::runtime_error("[Actuator] cannot advertise service " + topic);
        }
        gzmsg << "[Actuator] external control, service: " << topic << std::endl;
    }

    // Runs on a gz-transport thread. One request at a time; blocks until the sim thread
    // has run the requested steps and produced the observation.
    bool On_Step(const gz::msgs::Double_V& req, gz::msgs::Double_V& rep)
    {
        const size_t N = joints_.size();
        rep.Clear();

        std::unique_lock<std::mutex> svc(svc_mtx_, std::try_to_lock);
        if (!svc.owns_lock()) { rep.add_data(2.0); return true; }            // busy

        if (req.data_size() < 1) { rep.add_data(1.0); return true; }
        const double n_raw = req.data(0);

        // n_steps == -1: RELEASE. The sim thread blocks inside PreUpdate while it waits for the
        // next request, and in that state it cannot process world-control messages (pause, reset,
        // stop). Release makes it stop waiting, so pause/reset/shutdown work. A world reset
        // re-arms the gate.
        if (n_raw == -1.0) {
            std::lock_guard<std::mutex> lk(mtx_);
            release_ = true;
            cv_.notify_all();
            rep.add_data(0.0);
            return true;
        }

        if (!std::isfinite(n_raw) || n_raw < 0.0 || n_raw > 1.0e6) { rep.add_data(1.0); return true; }
        const int64_t n = static_cast<int64_t>(std::llround(n_raw));
        const size_t len = static_cast<size_t>(req.data_size() - 1);
        const bool observe_only = (n == 0);        // voltages, if any, are ignored
        const bool hold = (len == N);
        const bool seq  = (n > 0 && len == N * static_cast<size_t>(n));
        if (!observe_only && !hold && !seq) { rep.add_data(1.0); return true; }

        if (paused_.load()) { rep.add_data(4.0); return true; }

        std::vector<double> volts;
        if (!observe_only) {
            volts.resize(len);
            for (size_t k = 0; k < len; ++k) volts[k] = Sanitize(req.data(static_cast<int>(k + 1)));
        }

        std::unique_lock<std::mutex> lk(mtx_);
        pending_seq_ = std::move(volts);
        pending_n_ = n;
        has_request_ = true;
        reply_ready_ = false;
        cv_.notify_all();

        if (!cv_.wait_for(lk, reply_timeout_, [this] { return reply_ready_ || stop_; }) || !reply_ready_) {
            has_request_ = false;
            return false;
        }
        rep = reply_;
        reply_ready_ = false;
        return true;
    }

    // Sim thread. On return, ext_now_ holds the voltage for this step (or zeros if the
    // controller went away).
    void Gate_Step(const gz::sim::EntityComponentManager& ecm,
                   const gz::sim::UpdateInfo& info)
    {
        const size_t N = joints_.size();
        if (gave_up_) {
            std::fill(ext_now_.begin(), ext_now_.end(), 0.0);
            return;
        }

        while (steps_left_ == 0) {
            std::unique_lock<std::mutex> lk(mtx_);

            // The previous request is finished (or this is the very first step):
            // the ECM now holds the state after its last step -> answer it.
            if (waiting_reply_) {
                Build_Observation(ecm, info, reply_);
                waiting_reply_ = false;
                reply_ready_ = true;
                cv_.notify_all();
            }

            const bool got = cv_.wait_for(lk, idle_timeout_,
                                          [this] { return has_request_ || stop_ || release_; });
            if (got && release_ && !has_request_ && !stop_) {
                // Released by the client: stop gating (voltage 0) until the next world reset,
                // so the sim thread is free to handle pause / reset / shutdown messages.
                release_ = false;
                gave_up_ = true;
                std::fill(ext_now_.begin(), ext_now_.end(), 0.0);
                return;
            }
            if (!got || stop_) {
                gave_up_ = true;
                gzerr << "[Actuator] no step request for "
                      << std::chrono::duration<double>(idle_timeout_).count()
                      << " s -- external control abandoned, voltage = 0." << std::endl;
                std::fill(ext_now_.begin(), ext_now_.end(), 0.0);
                return;
            }

            ext_seq_      = std::move(pending_seq_);
            steps_left_   = pending_n_;
            ext_idx_      = 0;
            has_request_  = false;
            waiting_reply_ = true;
        }

        const bool hold = (ext_seq_.size() == N);
        for (size_t i = 0; i < N; ++i) {
            ext_now_[i] = hold ? ext_seq_[i] : ext_seq_[static_cast<size_t>(ext_idx_) * N + i];
        }
        ++ext_idx_;
        --steps_left_;
    }

    void Build_Observation(const gz::sim::EntityComponentManager& ecm,
                           const gz::sim::UpdateInfo& info,
                           gz::msgs::Double_V& out) const
    {
        out.Clear();
        out.add_data(0.0);                                                   // status
        out.add_data(std::chrono::duration<double>(info.simTime).count());   // sim_time
        out.add_data(static_cast<double>(info.iterations));                  // iteration

        const size_t N = joints_.size();
        for (size_t i = 0; i < N; ++i) {
            const auto p = joints_[i].Position(ecm);
            out.add_data((p && !p->empty()) ? (*p)[0] : 0.0);
        }
        for (size_t i = 0; i < N; ++i) {
            const auto v = joints_[i].Velocity(ecm);
            out.add_data((v && !v->empty()) ? (*v)[0] : 0.0);
        }

        double hx = 0, hy = 0, hyaw = 0, hvx = 0, hvy = 0, hwz = 0;
        if (have_head_) {
            const auto pose = head_.WorldPose(ecm);
            const auto lv   = head_.WorldLinearVelocity(ecm);
            const auto av   = head_.WorldAngularVelocity(ecm);
            if (pose) { hx = pose->Pos().X(); hy = pose->Pos().Y(); hyaw = pose->Rot().Yaw(); }
            if (lv)   { hvx = lv->X(); hvy = lv->Y(); }
            if (av)   { hwz = av->Z(); }
        }
        for (double d : {hx, hy, hyaw, hvx, hvy, hwz}) out.add_data(d);

        double cx = 0, cy = 0, cvx = 0, cvy = 0, cyaw = 0, tx = 0, ty = 0, mt = 0;
        for (size_t i = 0; i < chain_.size(); ++i) {
            const auto pose = chain_[i].WorldPose(ecm);
            const auto lv   = chain_[i].WorldLinearVelocity(ecm);
            const double m  = chain_mass_[i];
            if (!pose) continue;
            cx += pose->Pos().X() * m;  cy += pose->Pos().Y() * m;
            cyaw += pose->Rot().Yaw() * m;
            if (lv) { cvx += lv->X() * m; cvy += lv->Y() * m; }
            mt += m;
        }
        if (mt > 1e-12) { cx /= mt; cy /= mt; cvx /= mt; cvy /= mt; cyaw /= mt; }
        if (!chain_.empty()) {
            if (const auto tp = chain_.back().WorldPose(ecm)) { tx = tp->Pos().X(); ty = tp->Pos().Y(); }
        }
        for (double d : {cx, cy, cvx, cvy, cyaw, tx, ty}) out.add_data(d);   // 7 new values
    }

    // ------------------------------------------------------------------
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

    // Recording parameters
    std::vector<gz::sim::Link> chain_;     // head, then child link of each joint
    std::vector<double> chain_mass_;

    // Debugging Output files
    std::ofstream act_out_;
    std::ofstream volt_out_;

    // Open-loop control signals
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

    // ---- external step control state ----
    bool external_control_ = false;
    gz::transport::Node node_;
    std::atomic<bool> paused_{true};        // last value seen in PreUpdate; the sim starts paused
    gz::sim::Link head_{gz::sim::kNullEntity};
    bool have_head_ = false;
    std::chrono::milliseconds idle_timeout_{30000};
    std::chrono::milliseconds reply_timeout_{10000};

    std::mutex svc_mtx_;                    // one service call at a time
    std::mutex mtx_;                        // guards everything below marked 
    std::condition_variable cv_;
    bool stop_ = false;                     // 
    bool has_request_ = false;              // 
    bool release_ = false;                  //  client asked the gate to stop waiting
    std::vector<double> pending_seq_;       // 
    int64_t pending_n_ = 0;                 // 
    bool waiting_reply_ = false;            //  a request is being executed
    bool reply_ready_ = false;              // 
    gz::msgs::Double_V reply_;              // 
    bool gave_up_ = false;                  // (m, but only the sim thread reads it in the hot path)

    // sim-thread only
    int64_t steps_left_ = 0;
    int64_t ext_idx_ = 0;
    std::vector<double> ext_seq_;
    std::vector<double> ext_now_;
};
}  // namespace mubot

GZ_ADD_PLUGIN(mubot::ActuatorPlugin,
              gz::sim::System,
              mubot::ActuatorPlugin::ISystemConfigure,
              mubot::ActuatorPlugin::ISystemReset,
              mubot::ActuatorPlugin::ISystemPreUpdate)

GZ_ADD_PLUGIN_ALIAS(mubot::ActuatorPlugin, "mubot::ActuatorPlugin")