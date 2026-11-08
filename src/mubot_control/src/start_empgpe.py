#!/usr/bin/env python

from MU_EV.controller.mubot_rob_env import MubotEnv # type: ignore
from MU_EV.controller.EM_pgpe import EMPGPE # type: ignore
import numpy as np
import rospy, rospkg
import pandas as pd
import random, os, time, sys, shutil
from pathlib import Path
from datetime import datetime, timedelta

class MubotPGPE:
    def __init__(self,file_path, max_sim_time, wk_id,morph_ID=None):
        self.env = MubotEnv(file_path, max_sim_time, wk_id)
        self.file_path = file_path+"/"
        if morph_ID is not None:
            self.morph_ID = morph_ID
            self.morph_path = Path(self.file_path).parent.parent/ 'Population_data' / str(morph_ID)
        pass

    def run_pgpe(self,wk_id,act_num):
        ######################################################################################################
        if os.path.exists(self.file_path+'cpg_history.csv'):
            os.remove(self.file_path+'cpg_history.csv')
        if os.path.exists(self.file_path+'rollout_history.csv'):
            os.remove(self.file_path+'rollout_history.csv')
        if os.path.exists(self.file_path+'reward_history.csv'):
            os.remove(self.file_path+'reward_history.csv')
        ###############################################################################################
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('des_mubot')
        self.wk_id = wk_id
    # ######################################################################################################          # number of actuators
        ### initialize PGPE parameters
        if act_num<=4:
            n_rollout = 40;     n_episodes= 15*2
        elif act_num<=7:
            n_rollout = 60;     n_episodes= 17*2
        else:
            n_rollout = 80;     n_episodes= 20*2
        ################################################################
        ########## Debug
        # rospy.logerr("DEBUG PGPE rollout and episode: ");   n_rollout = 1;         n_episodes= 1;
        ################################################################

        n_select  = int(n_rollout/2);                           n_param   = act_num*2       # number of parameter (amp, (amp+phase)*n, fre)
        record_mu = np.zeros((n_episodes+1,n_param));           record_sigma = np.zeros((n_episodes+1,n_param))
        record_reward = np.zeros((n_episodes+1,n_rollout));     record_cpg = np.zeros(((n_episodes+1)*n_rollout,n_param))
        temp_input = np.zeros(n_param+1);                       record_sta = np.zeros(((n_episodes+1)*n_rollout,4))
        #######################################################################################################
        ### Initial mu and sigma
        record_mu[0,0]   = random.uniform(10,15);       record_sigma[0,0]   = 5       
        for i in range(act_num-1):
            record_mu[0,2*i+1] = random.uniform(10,15); record_sigma[0,2*i+1] = 5;    #Amplitute for i-th actuator
            record_mu[0,2*i+2] = random.uniform(0,1);   record_sigma[0,2*i+2] = 0.3;  #Phase     for i+1-th actuator   
        record_mu[0,-1] = random.uniform(0,10);         record_sigma[0,-1] = 2.5;     #Frequency
        #######################################################################################################
        # Start PGPE learning
        empgpe = EMPGPE(n_param,n_rollout,act_num,n_select)
        self._csv_write('cpg_history.csv', np.hstack((record_mu[0,:],record_sigma[0,:])),mode='w')
        for x in range(n_episodes):
            record_cpg[x*n_rollout:(x+1)*n_rollout,:] = empgpe.sample_params(record_mu[x,:], record_sigma[x,:])
            record_cpg[x*n_rollout:(x+1)*n_rollout,-1] = np.clip(record_cpg[x*n_rollout:(x+1)*n_rollout,-1],1.0,None)
            ######################################################################################################
            for y in range(n_rollout):
                ts = time.time()
                dt = datetime.utcfromtimestamp(ts) - timedelta(hours=4)
                print(f"\033[32m{dt.strftime('%m-%d %H:%M:%S')}  ######## \033[31m  E :{x+1}/{n_episodes} \033[93m  R :{y+1}/{n_rollout}\033[0m")
                temp_input = record_cpg[x*n_rollout+y,:]
                self._csv_write('control_signal.csv', temp_input,mode='w')
                ##### run Gazebo and get reward
                # record_reward[x,y] = self.env.step()
                ##########################################################
                self.env.step()
                record_sta[x*n_rollout+y,:] = pd.read_csv(self.file_path+"drift_metrics.csv").to_numpy()
                record_reward[x,y] = self.env.reward()
                ##########################################################
                mo_y = 'w' if ((x==0) and (y==0)) else 'a'
                self._csv_write('rollout_history.csv', np.hstack((temp_input,record_reward[x,y])),mode=mo_y)
                self._csv_write('drift_history.csv',record_sta[x*n_rollout+y,:], mode=mo_y)
                # self.env._cleanup_ros_logs()
            ### Update mu and sigma
            record_mu[x+1,:], record_sigma[x+1,:], _= empgpe.learn(record_cpg[x*n_rollout:(x+1)*n_rollout,:],record_reward[x,:],record_mu[x,:],record_sigma[x,:])
            ### Logging
            self._csv_write('cpg_history.csv', np.hstack((record_mu[x+1,:],record_sigma[x+1,:]))) 
            mo_x = 'w' if x==0 else 'a'
            self._csv_write('reward_history.csv', record_reward[x,:],mode=mo_x)
            ### Reward statistics
            mean_reward = np.mean(record_reward[x,:]); std_reward  = np.std(record_reward[x,:])
            if (std_reward<=mean_reward*0.1) and (x>=n_episodes*0.5):
                rospy.logerr("Early converged at episode: "+str(x+1))
                break
        final_max = np.argmax(record_reward[x,:]); final_reward = record_reward[x,final_max]; final_cpg = record_cpg[x*n_rollout+final_max,:]
        if self.morph_ID is not None and os.path.exists(self.file_path + 'gait_record.csv'):
            shutil.move(self.file_path + 'gait_record.csv', self.morph_path / 'gait_.csv')
        self.env._cleanup_ros_logs()
        return final_reward,final_cpg
    
    def continue_pgpe(self,wk_id,act_num):
        ###############################################################################################
        # rospy.init_node('mubot_evolve', anonymous=False, log_level=rospy.INFO)
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('des_mubot')
        self.wk_id = wk_id
        ######################################################################################################          # number of actuators
        ### initialize PGPE parameters
        if act_num<=4:
            n_rollout = 40
            n_episodes= 15*2
        elif act_num<=7:
            n_rollout = 60
            n_episodes= 20*2
        else:
            n_rollout = 80
            n_episodes= 25*2

        n_select  = int(n_rollout/2);                       n_param   = act_num*2                               # number of parameter (amp, (amp+phase)*n, fre)
        record_mu = np.zeros((n_episodes+1,n_param));       record_sigma = np.zeros((n_episodes+1,n_param))
        record_reward = np.zeros((n_episodes+1,n_rollout)); record_cpg = np.zeros(((n_episodes+1)*n_rollout,n_param))
        # episode_mu = np.zeros(n_param);                     episode_sigma = np.zeros(n_param)
        temp_input = np.zeros(n_param+1)
        #######################################################################################################
        ### Retrieve history
        cpg_hist     = pd.read_csv(self.file_path+'cpg_history.csv',    header=None,index_col=None).to_numpy()
        ep_comp = cpg_hist.shape[0]-1
        #abandon incomplete rollouts in the last episode
        reward_hist  = pd.read_csv(self.file_path+'reward_history.csv', header=None,index_col=None).to_numpy()[:ep_comp,:]
        pd.DataFrame(reward_hist).to_csv(self.file_path+'reward_history.csv',float_format='%.8f',header=None,index=None,mode='w')
        rollout_hist = pd.read_csv(self.file_path+'rollout_history.csv',header=None,index_col=None).to_numpy()[:ep_comp*n_rollout,:]
        pd.DataFrame(rollout_hist).to_csv(self.file_path+'rollout_history.csv',float_format='%.8f',header=None,index=None,mode='w')

        record_mu[:ep_comp+1,:] = cpg_hist[:ep_comp+1,:n_param]
        record_sigma[:ep_comp+1,:] = cpg_hist[:ep_comp+1:,n_param:]
        record_reward[:ep_comp,:] = reward_hist
        record_cpg[:rollout_hist.shape[0],:] = rollout_hist[:rollout_hist.shape[0],:-1]
        #######################################################################################################
        # Start PGPE learning
        empgpe = EMPGPE(n_param,n_rollout,act_num,n_select)
        # print(f"Continue PGPE from episode {ep_comp+1}/{n_episodes}")
        if ep_comp==n_episodes:
            x = ep_comp-1
        # x = ep_comp
        for x in range(ep_comp,n_episodes):
            record_cpg[x*n_rollout:(x+1)*n_rollout,:] = empgpe.sample_params(record_mu[x,:], record_sigma[x,:])
            ######################################################################################################
            
            for y in range(n_rollout):
                ts = time.time()
                dt = datetime.utcfromtimestamp(ts) - timedelta(hours=4)
                print(f"\033[32m{dt.strftime('%m-%d %H:%M:%S')}  ######## \033[31m  E :{x+1}/{n_episodes} \033[93m  R :{y+1}/{n_rollout}\033[0m")
                # rospy.logwarn("#####E :"+ str(x+1)+"/"+str(n_episodes)+"  R :"+ str(y+1)+"/"+str(n_rollout))
                temp_input = record_cpg[x*n_rollout+y,:]
                self._csv_write('control_signal.csv', temp_input,mode='w')
                ##### run Gazebo and get reward
                record_reward[x,y] = self.env.step()
                ##########################################################
                if (x==0) and (y==0):
                    self._csv_write('rollout_history.csv', np.hstack((temp_input,record_reward[x,y])),mode='w')
                else:
                    self._csv_write('rollout_history.csv', np.hstack((temp_input,record_reward[x,y])))
            ### Update mu and sigma
            record_mu[x+1,:], record_sigma[x+1,:], _= empgpe.learn(record_cpg[x*n_rollout:(x+1)*n_rollout,:],record_reward[x,:],record_mu[x,:],record_sigma[x,:])
            ### Logging
            self._csv_write('cpg_history.csv', np.hstack((record_mu[x+1,:],record_sigma[x+1,:]))) 
            if x ==0:
                self._csv_write('reward_history.csv', record_reward[x,:],mode='w')
            else:
                self._csv_write('reward_history.csv', record_reward[x,:])
            ### Reward statistics
            mean_reward = np.mean(record_reward[x,:]); std_reward  = np.std(record_reward[x,:])
            # rospy.logerr(f" Mean Reward: {mean_reward:.3e} Std Reward: {std_reward:.3e}")
            # rospy.logerr("DEBUG early stop terminated")
            if (std_reward<=mean_reward*0.1) and (x>=n_episodes*0.5):
                rospy.logerr("Early converged at episode: "+str(x+1))
                break
        

        # rospy.logerr("DEBUG reward tuned")
        final_max = np.argmax(record_reward[x,:]); final_reward = record_reward[x,final_max]; final_cpg = record_cpg[x*n_rollout+final_max,:]
        # Fix:
        if self.morph_ID is not None and os.path.exists(self.file_path + 'gait_record.csv'):
            os.makedirs(self.morph_path , exist_ok=True)
            shutil.move(self.file_path + 'gait_record.csv', self.morph_path / 'gait_.csv')
        return final_reward,final_cpg
    
    def one_pepg(self,wk_id,morph_ID,act_num):
        ###############################################################################################
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('des_mubot')
        self.wk_id = wk_id
        ####################################################################################################### 
        ### initialize PGPE parameters        
        n_param   = act_num*2;        temp_input = np.zeros(n_param+1);        reward = 0.0
        #######################################################################################################
        # Start PGPE learning
        rospy.logwarn("##### Rerun for morph ID: "+str(morph_ID)+" #####")
        df = pd.read_csv(self.file_path+'/controller.csv',header=None,index_col=None).to_numpy().flatten()
        temp_input = df
        self._csv_write('control_signal.csv', temp_input,mode='w')
        ##### run Gazebo and get reward
        # reward = self.env.gait_step()
        self.env.step()
        reward = self.env.reward()
        rospy.logerr(f" Reward:  {reward:.6e}")
        ###############################################################################################
        return reward
    
    def _csv_write(self, filename, data, mode='a'):
        """Write CSV and guarantee the data is flushed to disk before returning.
        Prevents Gazebo plugin from reading a partially-written file."""
        df_data = pd.DataFrame(data.reshape(1, -1))
        filepath = self.file_path + filename
        with open(filepath, mode, newline='') as f:
            df_data.to_csv(f, float_format='%.8e', header=None, index=None)
            f.flush()
            os.fsync(f.fileno())   # ← guarantees kernel buffer is written to disk
    
    # def _csv_write(self, filename, data, mode='a'):
    #     """Write a single row to a CSV file. Uses explicit context manager to guarantee fd closure."""
    #     df_data = pd.DataFrame(data.reshape(1, -1))
    #     with open(self.file_path + filename, mode, newline='') as f:
    #         df_data.to_csv(f, float_format='%.8f', header=None, index=None)
    
    # def single_rollout(self,control_signal):
    #     self._csv_write('control_signal.csv', control_signal,mode='w')
    #     reward = self.env.step()
    #     rospy.logerr(f" Reward:  {reward*1000:.3e}")
    #     return reward

    # def single_episode(self):

    #     return    
        
    # def _csv_write(self,filename, data, mode='a'):
    #     df_data = pd.DataFrame(data.reshape(1,-1))
    #     # print(df_data.shape)
    #     df_data.to_csv(self.file_path+filename,float_format='%.8f',header=None,index=None,mode=mode)
    #     return
