#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 09:59:13 2019

@author: abernal
"""

"""
    1. decision_freq = This is the # of times the agent will make decision for the day. If the length of the episode is T=365 days and I make 1 decision per day (starting from beginning of day) then this parameter is 1. I assume that if freq=h, then agent makes decisions at uniformly spreadout times, ie 1/h time
    
    2. A = integer BOM matrix, resources x products
"""

import numpy as np
import bisect as bis
#from scipy.optimize import fsolve as invf

class Simulator(object):
    
    EPS = 1e-5
    
    def __init__(self, A, profit, capacity_res, num_prod=3, num_resour=4, T=365, max_service_time=5, st_dist=[.3,.25,.35,.05,.05], max_adv_res=10, ar_dist=[.25,.20,.2,.1,.08,.06,.05,.03,.01,.01,.01], lambd = None, HPP=True, decision_freq=1, p1=[1,2,1], p2=[4,7,5]):
        self.HRS = 24 # This depends on whether the stores are open 24 hrs. If not, change this to appropriate setting
        self.p1 = np.array(p1)
        self.p2 = np.array(p2)
        self.num_prod = num_prod
        self.num_resour = num_resour
        self.profit = np.array(profit)
        self.capacity_res = np.array(capacity_res)
        self.A = np.array(A) # Bill of materials matrix
        self.T = T # Episode length = T days
        self.max_service_time = max_service_time
        self.st_dist = st_dist
        self.max_adv_res = max_adv_res
        self.ar_dist = ar_dist
        self.CDF_st = np.cumsum(self.st_dist)
        self.CDF_ar = np.cumsum(self.ar_dist)
        self.decision_freq = decision_freq
        self.factor = self.HRS / self.decision_freq
        self.HPP = HPP
        self.lambd = None if lambd is None else np.array(lambd) # Assuming you want to use Homogenous PP
        
        # This part is for simulation
        self.done = False
        self.rewards = 0.0
        self.cumulative_rewards = 0.0
        self.penalty = -max(profit)*100
        # Tagging the customers. Customers who get scheduled get tagged
        self.tag = 1
        self.current_period = 0 # <= decision_freq, not really but close
        self.current_day = 0
        self.NUM_IN_SYSTEM = [0]*self.num_prod
        self.total_calls = 0
        self.per_blocks = 0 # blocks that occur in the period
        self.total_blocks = 0
        self.FUTURE_EVENTS = []
        self.num_customers = 0
        if HPP:
            self.total_lam = sum(self.lambd)
            self.prod_prob = self.lambd / self.total_lam
            self.CDFprod = np.cumsum(self.prod_prob)
            self.gen = self.yieldHPP()
        else: # Case when NHPP is used
            pass
        self.current_time = next(self.gen)
        
    def reset(self):
        self.rewards = 0.0
        # Tagging the customers. Customers who get scheduled get tagged
        self.tag = 1
        self.cumulative_rewards = 0.0
        self.gen.close()
        self.gen = self.yieldHPP()
        self.current_time = next(self.gen)
        self.current_period = 0 # <= decision_freq
        self.current_day = 0
        self.NUM_IN_SYSTEM = [0]*len(self.num_prod)
        self.total_calls = 0
        self.total_blocks = 0
        self.FUTURE_EVENTS = []
        self.num_customers = 0
        
    # Note that after each event times, the CDF function we invert changes
    # Lam = integral of lambda, in other words, the expected mean. Should be non-decreasing
    def yieldNHPP(self, Lam):
        ti = 0.0
        uni = np.random.rand()
        F_ti = lambda t: 1 - np.exp(-Lam(ti+t) + Lam(ti)) - uni   # This is the function to invert
        while True:
            ti = invf(F_ti , ti+np.random.rand())
            yield ti
            uni = np.random.rand()
            F_ti = lambda t: 1 - np.exp(-Lam(ti+t) + Lam(ti)) - uni
    
    # Theres 2 dimensions to the arrival rate lam, time and price. if HPP, then lam is constant wrt time, but is it constant or fluctuates with price.      
    def yieldHPP(self):
        while True:
            arrival_time = np.random.exponential(1.0/self.total_lam) # In hours
            prod = bis.bisect(self.CDFprod , np.random.rand())
            yield arrival_time
            yield prod
    
    # Actions are the prices for each product, i.e. profit. The RL rewards depend on this vector        
    def apply_action(self , price_vec):
        self.profit = np.array(price_vec)
        if sum(self.profit == self.p2) == self.num_prod:
            self.profit -= .01
        F = map(self.create_f, self.lambd, self.p1, self.p2) # Creates function for each product
        F = list(F) # F is iniitially a map object so Im converting it to a list of functions
        self.lambd = np.array([F[i](self.profit[i]) for i in range(self.num_prod)])
        self.total_lam = sum(self.lambd)
        self.prod_prob = self.lambd / self.total_lam
        self.CDFprod = np.cumsum(self.prod_prob)
        
    def create_f(self,lam,p1,p2):
        def f(p):
            ft = lam*np.exp(-(p-p1))/(1-np.exp(-(p2-p1)))
            st = lam*np.exp(-(p2-p1))/(1-np.exp(-(p2-p1)))
            return ft - st
        return f        
        
    def sample_service(self):
        uni = np.random.rand()
        d = bis.bisect(self.CDF_st , uni) # If uni exactly equals one of the points in CDF, then it chooses to the left, not right
        return d+1
    
    def sample_reservation(self):
        uni = np.random.rand()
        d = bis.bisect(self.CDF_ar , uni) # If uni exactly equals one of the points in CDF, then it chooses to the left, not right
        return d
    
    
    def yield_next_state(self):
        pass
        
    # Get current state, i.e. resources remaining
    def yield_state(self):
        self.simulate_process(self.current_period+1)
        remaining_resources = self.capacity_res - np.matmul(self.A, self.NUM_IN_SYSTEM)
        self.current_period += 1
        return remaining_resources , self.rewards , self.done
        
    # I have to run the process to get the next state    
    def simulate_process(self , period):
        time = period * self.factor 
        self.rewards = 0.0
        self.per_blocks = 0

        while self.current_time <= time :
            
            # Figure out what product request arrived
            prod = next(self.gen) 
            
            # Counting total number of calls to the system
            self.total_calls += 1
                    
            # Customer service time and reservation time
            svt = self.sample_service()
            adv = self.sample_reservation()
            arrival_time = ((self.current_time + self.HRS*adv) // self.HRS) * self.HRS if adv > 0 else self.current_time
            depart_time = arrival_time + self.HRS*svt - Simulator.EPS if adv > 0 else ((self.current_time + self.HRS*adv) // self.HRS) * self.HRS + self.HRS*svt - Simulator.EPS
            
            # Checking to see if the customer ca be scheduled
            dummy_FUTURE_EVENTS = self.FUTURE_EVENTS[:]
            
            if dummy_FUTURE_EVENTS:
                idx = len(dummy_FUTURE_EVENTS)
                """ 
                In this scenario, customer will arrive way into the future where
                the customer is guaranteed to not be blocked
                """ 
                if dummy_FUTURE_EVENTS[-1][0] <= arrival_time:
                    ADD = [arrival_time , 1, self.tag, prod]
                    bis.insort(self.FUTURE_EVENTS, ADD)
                    ADD = [depart_time , -1, self.tag, prod]
                    bis.insort(self.FUTURE_EVENTS, ADD)
                    self.tag += 1
                    self.num_customers += 1
                
                # In this scenario, customer will arrive near the end of the schedule
                # so I have to check if there will be strictly less than C reserved 
                # customers in the interval [arrival_time , last customer reserved]
                
                elif dummy_FUTURE_EVENTS[-1][0] <= depart_time and dummy_FUTURE_EVENTS[0][0] <= arrival_time:
                    dummy_NUM_IN_SYSTEM = self.NUM_IN_SYSTEM[:]
                    dummy_current_time = dummy_FUTURE_EVENTS[0][0]
                    row = 0
                    FLAG = False
                    while dummy_current_time <= arrival_time:
                        which_prod = dummy_FUTURE_EVENTS[row][3]
                        dummy_NUM_IN_SYSTEM[which_prod] += dummy_FUTURE_EVENTS[row][1] 
                        row += 1
                        dummy_current_time = dummy_FUTURE_EVENTS[row][0]
                    
                    #  At this point, current_time >= arrival_time
                    for k in range(row, idx):
                        cap = np.matmul(self.A, dummy_NUM_IN_SYSTEM)  # vector not a matrix. cap stands for capacity
                        if sum(cap + self.A.T[prod] <= self.capacity_res) < self.num_resour: 
                            FLAG = True
                            break
                        which_prod = dummy_FUTURE_EVENTS[k][3]
                        dummy_NUM_IN_SYSTEM[which_prod] +=  dummy_FUTURE_EVENTS[k][1]
                    
                   # This is where we SCHEDULE or not     
                    if FLAG == False: 
                        ADD = [arrival_time , 1, self.tag, prod]
                        bis.insort(self.FUTURE_EVENTS, ADD)
                        ADD = [depart_time , -1, self.tag, prod]
                        bis.insort(self.FUTURE_EVENTS, ADD)
                        self.tag += 1
                        self.num_customers += 1
                    else:   # if FLAG == True
                        self.total_blocks += 1
                        self.per_blocks += 1
                        self.rewards += self.penalty
                        self.cumulative_rewards += self.penalty 
               
                # In this scenario, customer will arrive in the middle of the schedule
                # so I have to check if there will be strictly less than C reserved 
                # customers in the interval [arrival_time , depart_time]
                 
                elif dummy_FUTURE_EVENTS[0][0] <= arrival_time:
                    dummy_NUM_IN_SYSTEM = self.NUM_IN_SYSTEM[:]
                    dummy_current_time = dummy_FUTURE_EVENTS[0][0]
                    row = 0
                    FLAG = False
                    while dummy_current_time <= arrival_time:
                        which_prod = dummy_FUTURE_EVENTS[row][3]
                        dummy_NUM_IN_SYSTEM[which_prod] += dummy_FUTURE_EVENTS[row][1]
                        row += 1
                        dummy_current_time = dummy_FUTURE_EVENTS[row][0]
#                    else:
                    # At this point, current_time >= arrival_time
                    while dummy_current_time <= depart_time:
                        cap = np.matmul(self.A, dummy_NUM_IN_SYSTEM)  # vector not a matrix. 
                        if sum(cap + self.A.T[prod] <= self.capacity_res) < self.num_resour: 
                            FLAG = True
                            break
                        
                        which_prod = dummy_FUTURE_EVENTS[row][3]
                        dummy_NUM_IN_SYSTEM[which_prod] += dummy_FUTURE_EVENTS[row][1] 
                        row += 1
                        dummy_current_time = dummy_FUTURE_EVENTS[row][0]
                    
                    # This is where we SCHEDULE or not   
                    if FLAG == False: 
                        ADD = [arrival_time , 1, self.tag, prod]
                        bis.insort(self.FUTURE_EVENTS, ADD)
                        ADD = [depart_time , -1, self.tag, prod]
                        bis.insort(self.FUTURE_EVENTS, ADD)
                        self.tag += 1
                        self.num_customers += 1
                    else:   # if FLAG == True
                        self.total_blocks += 1
                        self.per_blocks += 1
                        self.rewards += self.penalty
                        self.cumulative_rewards += self.penalty
                
                # The case when current reserving customer starts before schedule and ends after schedule
                elif dummy_FUTURE_EVENTS[-1][0] <= depart_time and dummy_FUTURE_EVENTS[0][0] >= arrival_time:
                    dummy_NUM_IN_SYSTEM = self.NUM_IN_SYSTEM[:]
#                    dummy_current_time = dummy_FUTURE_EVENTS[0][0]
                    row = 0
                    FLAG = False
                    for k in range(row, idx):
                        cap = np.matmul(self.A, dummy_NUM_IN_SYSTEM)  # vector not a matrix. cap stands for capacity
                        if sum(cap + self.A.T[prod] <= self.capacity_res) < self.num_resour: 
                            FLAG = True
                            break
                        which_prod = dummy_FUTURE_EVENTS[k][3]
                        dummy_NUM_IN_SYSTEM[which_prod] +=  dummy_FUTURE_EVENTS[k][1]
                   
                    if FLAG == False: 
                        ADD = [arrival_time , 1, self.tag, prod]
                        bis.insort(self.FUTURE_EVENTS, ADD)
                        ADD = [depart_time , -1, self.tag, prod]
                        bis.insort(self.FUTURE_EVENTS, ADD)
                        self.tag += 1
                        self.num_customers += 1
                    else:   # if FLAG == True
                        self.total_blocks += 1
                        self.per_blocks += 1
                        self.rewards += self.penalty
                        self.cumulative_rewards += self.penalty
                # Case when current reserving customer starts before schedule and ends in the middle
                elif dummy_FUTURE_EVENTS[-1][0] >= depart_time > dummy_FUTURE_EVENTS[0][0] and dummy_FUTURE_EVENTS[0][0] > arrival_time:   
                    dummy_NUM_IN_SYSTEM = self.NUM_IN_SYSTEM[:]
                    dummy_current_time = arrival_time
                    row = 0
                    FLAG = False
                    while dummy_current_time <= depart_time:
                        cap = np.matmul(self.A, dummy_NUM_IN_SYSTEM)  # vector not a matrix. 
                        if sum(cap + self.A.T[prod] <= self.capacity_res) < self.num_resour: 
                            FLAG = True
                            break
                        which_prod = dummy_FUTURE_EVENTS[row][3]
                        dummy_NUM_IN_SYSTEM[which_prod] += dummy_FUTURE_EVENTS[row][1] 
                        dummy_current_time = dummy_FUTURE_EVENTS[row][0]
                        row += 1
                        
                    if FLAG == False: 
                        ADD = [arrival_time , 1, self.tag, prod]
                        bis.insort(self.FUTURE_EVENTS, ADD)
                        ADD = [depart_time , -1, self.tag, prod]
                        bis.insort(self.FUTURE_EVENTS, ADD)
                        self.tag += 1
                        self.num_customers += 1
                    else:   # if FLAG == True
                        self.total_blocks += 1
                        self.per_blocks += 1
                        self.rewards += self.penalty
                        self.cumulative_rewards += self.penalty

                else: # This is the part where dummy_FUTURE_EVENTS[0][0] >= depart_time, in other words, current reserving customer will leave before the next true schedule arrival
                    cap = np.matmul(self.A, self.NUM_IN_SYSTEM)  # Notice that dummy is not used
                    if sum(cap + self.A.T[prod] <= self.capacity_res) < self.num_resour: 
                        self.total_blocks += 1
                        self.per_blocks += 1
                        self.rewards += self.penalty
                        self.cumulative_rewards += self.penalty
                    else:
                        ADD = [arrival_time , 1, self.tag, prod]
                        bis.insort(self.FUTURE_EVENTS, ADD)
                        ADD = [depart_time , -1, self.tag, prod]
                        bis.insort(self.FUTURE_EVENTS, ADD)
                        self.tag += 1
                        self.num_customers += 1
                        
            else:
                ADD = [arrival_time , 1, self.tag, prod]
                bis.insort(self.FUTURE_EVENTS, ADD)
                ADD = [depart_time , -1, self.tag, prod]
                bis.insort(self.FUTURE_EVENTS, ADD)
                self.tag += 1
                self.num_customers += 1
                    
            """
            We are done checking with the current customer. Now we have to go thru
            the actual events and update the system. Customers might leave or arrive 
            to the system. Therefore, we will add/subtract the customers who arrive/depart
            until the next customer who calls for reservation
            """
            
            next_Reservation_Time = self.current_time + next(self.gen)
            # This is when lambd is so small that next reserve time is sooo long
            if next_Reservation_Time >= time + self.HRS: 
                next_Reservation_Time = time + Simulator.EPS
            if self.FUTURE_EVENTS:
                while self.FUTURE_EVENTS[0][0] <= next_Reservation_Time:
                    self.current_time = self.FUTURE_EVENTS[0][0]
                    if self.FUTURE_EVENTS[0][1] == 1:
                        which_prod = self.FUTURE_EVENTS[0][3]
                        self.NUM_IN_SYSTEM[which_prod] += 1
                        self.rewards += self.profit[which_prod]
                        self.cumulative_rewards += self.profit[which_prod]
                        del self.FUTURE_EVENTS[0]
                        if not self.FUTURE_EVENTS: 
                            break
                    else:
                        which_prod = self.FUTURE_EVENTS[0][3]
                        self.NUM_IN_SYSTEM[which_prod] -= 1
                        del self.FUTURE_EVENTS[0]
                        if not self.FUTURE_EVENTS: 
                            break
            self.current_time = next_Reservation_Time
    
        if period == self.T*self.decision_freq:
            blk_prob = float(self.total_blocks)/self.total_calls 
            self.done = True
        else:
            blk_prob = -1
            self.done = False
    
        return blk_prob  
  
    
A = np.array([[1, 1, 1],
       [3, 2, 3],
       [1, 3, 1],
       [3, 1, 1]])
profit = np.array([1,2,1])
capacity_res = np.array([12,40,30,25])
lambd = np.array([4,8,4])/24

sim = Simulator(A=A, profit=profit, capacity_res=capacity_res, lambd=lambd)
sim.apply_action([1.,2.,1.])
sim.simulate_process(sim.current_period+1)
sim.apply_action([4.,7.,5.])
sim.simulate_process(sim.current_period+2)