import numpy as np
import numpy.random as rnd
import pandas as pd
from lifelines import KaplanMeierFitter
import time
import random
import math
from mpi4py import MPI
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
kmf = KaplanMeierFitter() 

epoch=1
nan=np.nan # represents impossible actions
n_iterations = 100 #10000
learning_rate0 = 0.01
learning_rate_decay = 0.01 #0.1
discount_rate = 0.9
totite=10000

time_inter=80 #(240 months)
T=1
doc_path='/public/home/dku_gpengzhan/personalizedHR/SA-ANN/data'
clusterornot=1 # 1 clust, 0 single
noffixed=2

if clusterornot==0:
    typo=str(size)+'-single'+str(noffixed)
elif clusterornot==1:
    typo=str(size)+'-2cluster'+str(noffixed)
else:
    typo=str(size)+'-4cluster'+str(noffixed)

group_size=2**(clusterornot)

msize=4282 #4282
exp_com=20 
# Reward
#1 is smooth transfer, 2 is change position, 3 is average stay time, 4 is popularity, 5 is reputation    

s1, s2=0.8, 0.2

#weight=[[0.2, 0.2, 0.2, 0.2, 0.2]]
weight=[[0.25, 0.25, 0.25, 0.25]]
suffer_time=4
penalty=0.5
def random_no_repeat1(numbers, count):
    
    number_list = list(numbers)
    random.shuffle(number_list)
    return number_list[:count]
    
def reward(a,W): 
    e=0
    for i in range(len(a)):
        e+=a[i]*W[i]
    return e
    
def duration(sp):
    k=4
    while True:
        p = rnd.choice(range(2), p=[STA[sp][k],1-STA[sp][k]]) # choose an poss
        if p==1:
            return int(k)
        else:
            k+=1 # #of 3 months
    
def finrew(cur_score,next_score,s,sp,ad_pro,vis):   
    if cur_score>=next_score:
        decr=penalty
    else:
        decr=1
    vR=0
    new_r1=next_score*decr
    i, start=sp, 0
    if sp in vis:
        ad_pro=1
        start=8*vis.count(sp)
    st_day=duration(sp)
    for j in range(start,st_day+start):
        if j<suffer_time:
          vR+=new_r1*3*STA[i][j]*ad_pro
        else:
          if j<99:
              vR+=new_r1*3*STA[i][j]
    return vR, st_day

# function for define the p of choosing action i    
def pforac(s,a,msize,T):
    p=np.zeros(msize)
    #p[a]=1*T[s][a]
    p[a]=1
    p[s]=1-p[a]
    return list(p)

def calscore(cur_pos, last_pos, i):
    temp=pos_per[i]
    li_temp=list(temp)
    temp.sort()
    if cur_pos==last_pos: # next pos occupied the majority in next comp
        score=s1*a_R1[i]+s2*100
    elif li_temp[cur_pos]>=temp[-3]:
        score=s1*a_R1[i]+s2*80
    else:
        score=s1*a_R1[i]+s2*60
    return score

def path_rew(path, ori_pos):
    ini=list(path)[0]
    old_c, val, v1 = ini, 0, 1
    cur_pos, last_pos = ori_pos, ori_pos
    score=calscore(cur_pos, last_pos, ini)
    cur_score, last_score= score, score
    idx, i, ad_pro=0, 0, 1
    all_pos=[ori_pos]
    vis=[old_c]
    accumu_v=[val]
    while i<len(path):
        if path[i]==old_c:
            if idx>3:
                ad_pro=1
            val+=v1*score*3*STA[path[i]][idx]*v1*ad_pro
            idx+=1 
            i+=1
            accumu_v.append(val)
        else:
            cur_pos, last_pos = rnd.choice(range(26),p=posi_cha[last_pos]), cur_pos
            all_pos.append(cur_pos)
            ad_pro=SIMI[path[i]][old_c]
            old_c=path[i]
            if old_c not in vis:
                vis.append(old_c)
                idx=0
            else:
                idx=8
            score=calscore(cur_pos, last_pos, path[i])
            cur_score, last_score= score, cur_score
            if last_score>=cur_score:
                v1=penalty
            else:
                v1=1  
        #print (score,val,cur_pos)
    return val, all_pos, accumu_v
# read company   
compid=open(doc_path+"/recom.txt",'r')  
content2=compid.readlines()[:msize]
comp=[]
for i in content2:
    i=i.split('\n')
    comp.append(i[0])
# 2 K-MEANS group
compid=open(doc_path+"/simi.txt",'r')  
content2=compid.readlines()[:msize]
perc=[]
for i in content2:
    i=i.split('\n')
    perc.append(i[0])
cm2=[]  
for i in range(len(perc)):
    sing=perc[i].split(' ')
    temp=np.zeros(26)
    for j in range(26):
        temp[j]=float(sing[j])
    cm2.append(temp)
if clusterornot==0:
    pos_per=cm2
kmeans = KMeans(n_clusters=100, random_state=0).fit(cm2)
##rought divide
totgroup=[]
for i in range(100):
    g=[]
    for j in range(len(kmeans.labels_)):     
        if kmeans.labels_[j]==i:
            g.append(j)
    totgroup.append(g)

FEA=np.zeros([msize,4])
df=pd.read_csv(doc_path+'/v2reward.csv')[:msize]
for i in range(4):
    fea=df[str(i)].tolist()
    for j in range(len(fea)):
        FEA[j][i]=fea[j]  
## normalize stay time 
maxtime=np.amax(FEA, axis=0)[1]
FEA[:,1]=FEA[:,1]/maxtime


# stay probability
L=3  # a season
numofsta=int(20*12/L)
compid=open(doc_path+"/obs.txt",'r')  
content2=compid.readlines()[:msize]
oobs=[]
for i in content2:
    i=i.split('\n')
    oobs.append(i[0])
obs=[]
for i in oobs:
    i=i.split('[')[1]
    i=i.split(']')[0]
    b=np.fromstring(i, dtype=int, sep=',')
    obs.append(list(b))
##stay time
compid=open(doc_path+"/staytime.txt",'r')  
content2=compid.readlines()[:msize]
osta=[]
for i in content2:
    i=i.split('\n')
    osta.append(i[0])
sta=[]
for i in osta:
    i=i.split('[')[1]
    i=i.split(']')[0]
    b=np.fromstring(i, dtype=int, sep=',')
    sta.append(list(b))

posi_cha=np.genfromtxt(doc_path+'/posi_change.csv',  delimiter=",")[1::] 
transfer_record=np.genfromtxt(doc_path+'/transferrecord.csv',  delimiter=",")[1::] 
  
def data_preprocess(weight):

    # 1. Calculate Reward
    R=np.zeros(msize)
    for i in range(msize):
        R[i]=reward(FEA[i],weight)
        
    # 2. Divide Company based on Score
    if clusterornot!=0: 
        detailgroup=[]
        for i in range(len(totgroup)):
            l=len(totgroup[i])
            if l<=group_size:
                detailgroup.append(totgroup[i])
            else:
                R_c=[]
                for j in totgroup[i]:
                    R_c.append(R[j])
                Z = [x for _,x in sorted(zip(R_c,totgroup[i]))]
                Z.reverse()
                #print(Z)
                for j in range(0,l//group_size):
                    E=Z[j*group_size:(j+1)*group_size]
                    detailgroup.append(E)
                if l%group_size!=0:
                    detailgroup.append(Z[l//group_size*group_size::])
    else:
        detailgroup=[[i] for i in range(len(comp))] 
        
    # 1.1 Normalize Reward
    maxreward=np.amax(R)
    minreward=np.amin(R) 
    R1=[]
    for i in R:
        ele=(i-minreward)/(maxreward-minreward)*100
        R1.append(ele)
    if clusterornot==0:
        a_R1=np.array(R1)
    else:
        a_R1=np.array(R1)
        c_R1=[np.mean(a_R1[i]) for i in detailgroup]
        maxreward=max(c_R1)
        minreward=min(c_R1) 
        R1=[(i-minreward)/(maxreward-minreward)*100 for i in c_R1]
        c_R1=R1
        a_R1=np.array(c_R1)
                     
    # 3. Calculate Stay Probability
    STA=np.zeros([len(detailgroup),100])
    ind=0
    for i in detailgroup:
        temobs=[]
        temsta=[]
        for j in i:
            temobs+=obs[j]
            temsta+=sta[j]
        if len(temsta)>0:
            temobs=np.asarray(temobs)
            temsta=np.asarray(temsta)       
            kmf.fit(temsta, temobs,label='Kaplan Meier Estimate') 
            for k in range(numofsta):
                if k<4:
                    STA[ind][k]=1
                else:
                    stpo=kmf.predict(L*30*k) # stay possibility                    
                    STA[ind][k]=stpo
        ind+=1
    '''
    # 4. Predict Stay Days based on STA
    STD=np.zeros([len(detailgroup)])
    for i in range(len(detailgroup)):
        j=4
        while True:
            p = rnd.choice(range(2), p=[STA[i][j],1-STA[i][j]]) # choose an poss
            if p==1:
                STD[i]=j 
                break
            else:
                j+=1 # #of 3 months
    '''
    # 4. Transfer Rank
    trans_rank=[]
    for i in range(len(detailgroup)):
        temp=[j for j in range(len(detailgroup))]
        temp2=[]
        idx=0
        for ele in detailgroup[i]:
            if idx==0:
                temp2=np.array(transfer_record[ele])
            else:
                temp2+=np.array(transfer_record[ele])
            idx+=1
        if clusterornot==0:
            temp3=temp2
        else:
            temp3=[]
            for j in range(len(detailgroup)):
                val=0
                for ele in detailgroup[j]:
                    val+=temp2[ele]
                temp3.append(val)
        t2=[x for _, x in sorted(zip(temp3,temp), key=lambda pair: pair[0], reverse=True)]
        trans_rank.append(t2) 
    # 5. Calculate Similarity
    SIMI=np.zeros([len(detailgroup),len(detailgroup)])
    if clusterornot!=0:
        pos_per=[]
        for i in range(len(detailgroup)):
            temp1=np.zeros([26]) 
            for ele in detailgroup[i]:
                sing=perc[ele].split(' ')
                for k in range(26):
                    temp1[k]+=float(sing[k])/len(detailgroup[i]) 
            pos_per.append(temp1)
            SIMI[i][i]=1
            for j in range(i+1,len(detailgroup)):
                temp2=np.zeros([26])
                for ele in detailgroup[j]:
                    sing=perc[ele].split(' ')
                    for k in range(26):
                        temp2[k]+=float(sing[k])/len(detailgroup[j])  
                SIMI[i][j]=cosine_similarity([temp1], [temp2])[0][0] 
                SIMI[j][i]=cosine_similarity([temp1], [temp2])[0][0]
    else:
        df=pd.read_csv(doc_path+'/V3simimatrix.csv')
        pos_per=cm2
        for i in range(len(detailgroup)):
            simi=df[str(i)].tolist()
            for j in range(i,len(detailgroup)):
                msimi='{:.4}'.format(simi[j])
                SIMI[j][i]=msimi 
                SIMI[i][j]=msimi 
    return a_R1, detailgroup, STA, trans_rank, SIMI, pos_per  

tot_nrew, tot_group, tot_STA, tot_rank, tot_SIMI, tot_posper=[], [], [], [], [], []
for ele in weight:
    e1, e2, e3, e4, e5, e6=data_preprocess(ele)
    tot_nrew.append(e1)
    tot_group.append(e2)
    tot_STA.append(e3)
    tot_rank.append(e4)
    tot_SIMI.append(e5)
    tot_posper.append(e6)
    

R2=[]
turning_ind=0 
a_R1, detailgroup, STA, trans_rank, SIMI, pos_per= tot_nrew[turning_ind], tot_group[turning_ind], tot_STA[turning_ind], tot_rank[turning_ind], tot_SIMI[turning_ind], tot_posper[turning_ind]
STD=np.zeros([len(detailgroup)])
for i in range(len(detailgroup)):
    suff,rew=0, 0
    new_r1=s1*a_R1[i]+s2*100
    k=4
    while True:
        p = rnd.choice(range(2), p=[STA[i][k],1-STA[i][k]]) # choose an poss
        if p==1: 
            STD[i]=k
            break
        else:
            k+=1 # #of 3 months
    for j in range(int(k)):
        if j<suffer_time:
            suff+=new_r1*3*STA[i][j]
        else:
            rew+=new_r1*3*STA[i][j]
    R2.append([suff, rew])          
path_score=[]
se_c=[]
ttotherinf=[]
#print(size)
for worker in range(size):
    if rank==worker:
########
        for ep in range(epoch):
            comp_ti=[i for i in range(time_inter)]
            random.seed(ep) #10
            fixtime_list = random.sample(range(0, time_inter), noffixed*2)
            fixtime_list.sort()
            fle_timeint=int(time_inter)
            ttfix=[]
            for i in range(noffixed):
                temp=[j for j in range(fixtime_list[2*i], fixtime_list[2*(i+1)-1])]
                ttfix+=temp
                comp_ti=[i for i in comp_ti if i not in temp]
                fl_timeint=time_inter-len(temp)
            random.seed(ep+rank*10) #10
            complist=[i for i in range(len(detailgroup))]
            ini=complist[random.randint(0, len(complist)-1)]
            ori_pos=random.randint(0, 25)
            #ini=chosen_com[-rank-10]
            se_c.append([ini, ori_pos])
            curcom_list=[ini]
            ttrew, inirew=sum(R2[ini])/(3*time_inter), sum(R2[ini])/(3*time_inter) # inital average score
            inicol=[ini]
            while len(curcom_list)<exp_com:
                ele=random.choice(complist)
                if ele not in curcom_list:
                    curcom_list.append(ele)
            for ite in range(totite): #10^6
                length=len(curcom_list)
                possible_actions = [[i for i in range(length)]]*length
                for i in range(length):
                    p_a=list(possible_actions[i])
                    #p_a.remove(i)
                    possible_actions[i]=p_a
                    
                Q = np.full((length, length), -np.inf) # -inf for impossible actions
                for state, actions in enumerate(possible_actions):
                    Q[state, actions] = 0.0 # Initial value = 0.0, for all possible actions
                s = 0 # start in state 0 
                row_idx = np.array(curcom_list)
                p_SIMI=SIMI[row_idx]
                p_R1=a_R1[row_idx]  
                #p_R2=a_R2[row_idx]  
                p_STD=STD[row_idx] 
                worktime=p_STD[0]
                cur_pos=ori_pos   
                cur_score=s1*a_R1[ini]+s2*100
                vis=[curcom_list[s]]           
                for iteration in range(n_iterations):   
                    # random selection
                    pa1 = possible_actions[s] # all the s' at state s
                    a = rnd.choice(pa1) # randomly choose a action
                    #sp = rnd.choice(range(length),p=pforac(s,a,length,p_SIMI)) # choose next state
                    sp=a
                    ad_pro=p_SIMI[s][sp]
                    next_pos=rnd.choice(range(26),p=posi_cha[cur_pos]) # predict position in next company
                    #Calculate Company Score
                    temp=pos_per[curcom_list[sp]]
                    li_temp=list(temp)
                    temp.sort()
                    if next_pos==cur_pos or li_temp[next_pos]>=temp[-3]: # next pos occupied the majority in next comp
                        next_score=s1*p_R1[sp]+s2*100
                    else:
                        next_score=s1*p_R1[sp]+s2*50    
                    reward, st_da = finrew(cur_score,next_score,curcom_list[s],curcom_list[sp],ad_pro,vis)
                    learning_rate = learning_rate0 / (1 + iteration * learning_rate_decay)
                    Q[s, sp] = (1-learning_rate) * Q[s, sp] +  learning_rate * (
                    reward + discount_rate * np.max(Q[sp])
                    )
                    s = sp # move to next state 
                    vis.append(curcom_list[s])
                    #easily_trans=list(set(easily_trans))
                    cur_pos=next_pos   
                    cur_score=next_score
                    worktime+=st_da
                    if worktime>=fl_timeint:
                        s=0
                        cur_pos=ori_pos 
                        cur_score=s1*a_R1[ini]+s2*100
                        worktime=p_STD[0]
                        vis=[curcom_list[s]]
                resu=np.argmax(Q, axis=1)
                fin_path, p_ini, cur_c, cur_t=[], 0, ini, 0  
                cur_d=int(STD[curcom_list[p_ini]]) 
                for i in range(time_inter):    
                    if i not in ttfix:                                             
                        fin_path.append(cur_c) # selected company
                        cur_t+=1
                        if cur_t>=cur_d:
                            p_ini=resu[p_ini] 
                            cur_t, cur_d, cur_c=0, int(STD[curcom_list[p_ini]]), curcom_list[p_ini]
                    else:
                        fin_path.append(ini)
                av_re, new_pos, new_accumu=path_rew(fin_path, ori_pos)   
                av_re=av_re/(3*time_inter)
                dE=av_re-ttrew
                rate=np.float64(dE/T)
                if dE>=0 or random.uniform(0, 1) < math.exp(rate) :
                    inicol=list(fin_path)
                    inirew=av_re
                    fin_jobs=new_pos
                    curcom_list=list(set(fin_path))
                    opt_accumu=new_accumu
                    while len(curcom_list)<exp_com:
                        if ini not in curcom_list:
                            curcom_list=[ini]+curcom_list
                        else:
                            ele=random.choice(complist)
                            if ele not in curcom_list:                           
                                curcom_list.append(ele)
                    
                else: 
                    random.seed(ite)
                    curcom_list=list(set(inicol))
                    while len(curcom_list)<exp_com:
                        if ini not in curcom_list:
                            curcom_list=[ini]+curcom_list
                        else:
                            ele=random.choice(complist)
                            if ele not in curcom_list:                           
                                curcom_list.append(ele)
                
                cind=curcom_list.index(ini)
                curcom_list[0], curcom_list[cind]=ini, curcom_list[0]                      
                T=T*0.99
                ttrew=inirew
                                                  
########     
        comm.Barrier()
        totp_score=comm.gather([inirew], root=0)
        tot_inf=comm.gather(opt_accumu, root=0)
        if rank==0:
            f1=open('/public/home/dku_gpengzhan/personalizedHR/SA-ANN/result/'+typo+'Reibar.txt','w')
            f2=open('/public/home/dku_gpengzhan/personalizedHR/SA-ANN/result/'+typo+'Reiaccumu.txt','w')
            fin_score=[]
            for i in range(size):
                for j in range(epoch):
                    fin_score.append(totp_score[i][j])
            print(np.mean(fin_score))
            
            for i in fin_score:
                f1.write(str(i)+'\n')
            
            for i in range(time_inter):
                for j in range(size):
                    ele=tot_inf[j][i]                        
                    f2.write(str(ele)+'\n')
