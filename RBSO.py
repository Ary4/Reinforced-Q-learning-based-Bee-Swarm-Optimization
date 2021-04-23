import numpy as np
import os, glob, random, copy, sys, time, re
#from collections import defaultdict
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, \
ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.svm import SVC
#from sklearn.preprocessing import StandardScaler
#from oauth2client.client import GoogleCredentials
#from google.colab import drive
import pandas as pd

import xlsxwriter

#gdrive_path = '/content/gdrive'
#drive.mount(gdrive_path)

class Bee :
    def __init__(self,id,problem,locIterations):
        self.id=id
        self.data=problem
        self.solution=[]
        self.fitness= 0.0
        self.reward = 0.0
        self.locIterations=locIterations
        self.action = []
    
    def localSearch(self):
        best=self.fitness
        #done=False
        lista=[j for j, n in enumerate(self.solution) if n == 1]
        indice =lista[0]
        
        for itr in range(self.locIterations):
            while(True):
                pos=-1
                oldFitness=self.fitness
                for i in range(len(self.solution)):
                    
                    if ((len(lista)==1) and (indice==i) and (i < self.data.nb_attribs-1)):
                        i+=1
                    self.solution[i]= (self.solution[i] + 1) % 2
                    
                    quality = self.data.evaluate(self.solution)
                    if (quality >best):
                        pos = i
                        best=quality
                    self.solution[i]= (self.solution[i]+1) % 2
                    self.fitness = oldFitness 
                if (pos != -1):
                    self.solution[pos]= (self.solution[pos]+1)%2
                    self.fitness = best
                else:
                    break
            for i in range(len(self.solution)):
                oldFitness=self.fitness
                if ((len(lista)==1) and (indice==i) and (i < self.data.nb_attribs-1)):
                    i+=1
                self.solution[i]= (self.solution[i] + 1) % 2
                quality = self.data.evaluate(self.solution)
                if (quality<best):
                    self.solution[i]= (self.solution[i] + 1) % 2
                    self.fitness = oldFitness


    def ql_localSearch(self):
        
        iterations = int(self.locIterations/4) if self.locIterations >= 4 else 1
        for itr in range(iterations):
       
            state = self.solution.copy()

            next_state, action = self.data.ql.step(self.data,state)
            acc_state = self.data.evaluate(state)
            acc_new_state = self.data.evaluate(next_state)

            if (acc_state < acc_new_state):
                self.reward = acc_new_state
            elif (acc_state > acc_new_state):
                self.reward = acc_new_state - acc_state
            else :
                if (self.data.nbrUn(state) > self.data.nbrUn(next_state) ):
                    self.reward = 0.5 * acc_new_state
                else :
                    self.reward = -0.5 * acc_new_state

            self.fitness = self.data.ql.get_q_value(self.data,state,action)
            self.data.ql.learn(self.data,state,action,self.reward,next_state)
            self.solution = next_state.copy()
        
       
    def setSolution(self,solution):
        self.solution=solution.copy()
        self.fitness=self.data.evaluate(solution)
    
    def Rand(self,num): 
        res = [] 
        res = np.random.choice([0,1],size=(num,),p=[2./10,8./10]).tolist()
  
        return res



class Swarm :
    def __init__(self,problem,flip,maxChance,nbrBees,maxIterations,locIterations):
        self.data=problem
        self.flip=flip
        self.maxChance=maxChance
        self.nbChance=maxChance
        self.nbrBees=nbrBees
        self.maxIterations=maxIterations
        self.locIterations=locIterations
        self.beeList=[]
        self.refSolution = Bee(-1,self.data,self.locIterations)
        self.refSolution.setSolution(self.refSolution.Rand(self.data.nb_attribs))
        self.bestSolution = self.refSolution
        self.tabou=[]

    def searchArea(self):    
        i=0
        h=0
        
        self.beeList=[]
        while((i<self.nbrBees) and (i < self.flip) ) :
            #print ("First method to generate")
            
            solution=self.refSolution.solution.copy()
            k=0
            while((self.flip*k+h) < len(solution)):
                solution[self.flip*k +h] = ((solution[self.flip*k+h]+1) % 2)
                k+=1
            newBee=Bee(i,self.data,self.locIterations)
            #newBee.solution = copy.deepcopy(solution)
            newBee.solution = solution.copy()
            self.beeList.append(newBee)
            
            i+=1
            h=h+1
        h=0
        
        while((i<self.nbrBees) and (i< 2*self.flip )):
            #print("Second method to generate")

            solution=self.refSolution.solution.copy()
            k=0
            while((k<int(len(solution)/self.flip)) and (self.flip*k+h < len(solution))):
                solution[int(self.data.nb_attribs/self.flip)*h+k] = ((solution[int(self.data.nb_attribs/self.flip)*h+k]+1)%2)
                k+=1
            newBee=Bee(i,self.data,self.locIterations)
            #newBee.solution = copy.deepcopy(solution)
            newBee.solution = solution.copy()
            self.beeList.append(newBee)
            
            i+=1
            h=h+1
        while (i<self.nbrBees):
            #print("Random method to generate")
            solution= self.refSolution.solution.copy()
            indice = random.randint(0,len(solution)-1)
            solution[indice]=((solution[indice]+1) % 2)
            newBee=Bee(i,self.data,self.locIterations)
            #newBee.solution = copy.deepcopy(solution)
            newBee.solution = solution.copy()
            self.beeList.append(newBee)
            i+=1
        for bee in (self.beeList):
            lista=[j for j, n in enumerate(bee.solution) if n == 1]
            if (len(lista)== 0):
                bee.setSolution(bee.Rand(self.data.nb_attribs))
                
    def selectRefSol(self,typeOfAlgo):
      typeOfAlgo = typeOfAlgo
      if (typeOfAlgo == 0):
        self.beeList.sort(key=lambda Bee: Bee.fitness, reverse=True)
        bestQuality=self.beeList[0].fitness
        if(bestQuality>self.bestSolution.fitness):
            self.bestSolution=self.beeList[0]
            self.nbChance=self.maxChance
            return self.bestSolution
        else:
            if(  (len(self.tabou)!=0) and  bestQuality > (self.tabou[len(self.tabou)-1].fitness)  ):
                self.nbChance=self.maxChance
                return self.bestBeeQuality(typeOfAlgo)
            else:
                self.nbChance-=1
                if(self.nbChance > 0): 
                    return self.bestBeeQuality(typeOfAlgo)
                else :
                    return self.bestBeeDiversity()
      
      elif (typeOfAlgo == 1):
        self.beeList.sort(key=lambda Bee: Bee.reward, reverse=True)
        bestQuality=self.beeList[0].reward
        if(bestQuality>self.bestSolution.reward):
            self.bestSolution=self.beeList[0]
            self.nbChance=self.maxChance
            return self.bestSolution
        else:
            if(  (len(self.tabou)!=0) and  bestQuality > (self.tabou[len(self.tabou)-1].reward)  ):
                self.nbChance=self.maxChance
                return self.bestBeeQuality(typeOfAlgo)
            else:
                self.nbChance-=1
                if(self.nbChance > 0): 
                    return self.bestBeeQuality(typeOfAlgo)
                else :
                    return self.bestBeeDiversity()                  

    def distanceTabou(self,bee):
        distanceMin=self.data.nb_attribs
        for i in range(len(self.tabou)):
            cpt=0
            for j in range(self.data.nb_attribs):
                if (bee.solution[j] != self.tabou[i].solution[j]) :
                      cpt +=1
            if (cpt<=1) :
                return 0
            if (cpt < distanceMin) :
                distanceMin=cpt
        return distanceMin
    
    def bestBeeQuality(self,typeOfAlgo):
        distance = 0
        i=0
        pos=-1
        while(i<self.nbrBees):
            if (typeOfAlgo == 0):
              max_val=self.beeList[i].fitness
            if (typeOfAlgo == 1):
              max_val=self.beeList[i].reward  

            nbUn=self.data.nbrUn(self.beeList[i].solution)
            while((i<self.nbrBees) and (self.data.evaluate(self.beeList[i].solution) == max_val)):
                distanceTemp=self.distanceTabou(self.beeList[i])
                nbUnTemp = self.data.nbrUn(self.beeList[i].solution)
                if(distanceTemp > distance) or ((distanceTemp == distance) and (nbUnTemp < nbUn)):
                    if((distanceTemp==distance) and (nbUnTemp<nbUn)):
                        print("We pick the solution with less features")
                    nbUn=nbUnTemp
                    distance=distanceTemp
                    pos=i
                i+=1
            if(pos!=-1) :
                return self.beeList[pos]
        bee= Bee(-1,self.data,self.locIterations)
        bee.setSolution(bee.Rand(self.data.nb_attribs))
        return bee
            
    def bestBeeDiversity(self):
        max_val=0
        for i in range(len(self.beeList)):
            if (self.distanceTabou(self.beeList[i])> max_val) :
                max_val = self.distanceTabou(self.beeList[i])
        if (max_val==0):
            bee= Bee(-1,self.data,self.locIterations)
            bee.setSolution(bee.Rand(self.data.nb_attribs))
            return bee
        i=0
        while(i<len(self.beeList) and self.distanceTabou(self.beeList[i])!= max_val) :
            i+=1
        return self.beeList[i]
    
    def bso(self,typeOfAlgo):
        i=0
        while(i<self.maxIterations):
            #print("refSolution is : ",self.refSolution.solution)
            self.tabou.append(self.refSolution)
            #print("Iteration N° : ",i)
            
            self.searchArea()

            #La recherche locale
            
            for j in range(self.nbrBees):
              if (typeOfAlgo == 0):
                self.beeList[j].localSearch()
              elif (typeOfAlgo == 1):
                for episode in range(self.locIterations):
                  self.beeList[j].ql_localSearch()
                #print( "Q-value of bee " + str(j) + " solution is : " + str(self.beeList[j].fitness))
            self.refSolution = self.selectRefSol(typeOfAlgo)
            i+=1
        print("[BSO parameters used]\n")
        print("Type of algo : {0}".format(typeOfAlgo))
        print("Flip : {0}".format(self.flip))
        print("MaxChance : {0}".format(self.maxChance))
        print("Nbr of Bees : {0}".format(self.nbrBees))
        print("Nbr of Max Iterations : {0}".format(self.maxIterations))
        print("Nbr of Loc Iterations : {0}\n".format(self.locIterations))
        print("Best solution found : ",self.bestSolution.solution)
        print("Number of features used : {0}".format(self.data.nbrUn(self.bestSolution.solution)))
        print("Accuracy : {0:.2f} ".format(self.bestSolution.fitness*100))
        #print("Return (Q-value) : ",self.bestSolution.fitness)
        return self.bestSolution.fitness*100, self.data.nbrUn(self.bestSolution.solution)

    
    def str_sol(self,mlist):
        result = ''
        for element in mlist:
            result += str(element)
        return result

      
class QLearning:
    def __init__(self,nb_atts,actions):
        self.actions = actions
        self.alpha = 0.1 # Facteur d'apprentissage
        self.gamma = 0.9
        self.epsilon = 0.1
        self.q_table = [ {} for i in range(nb_atts) ] #defaultdict(lambda : [0.0,0.0,0.0,0.0])

    def get_max_value(self,data,state,actions_vals):
        max_val = 0.0
        arg_max = 0
        for i in actions_vals:
            if self.get_q_value(data,state,i) >= max_val:
                max_val = self.get_q_value(data,state,i)
                arg_max = i
        if max_val == 0:
            arg_max = np.random.choice(actions_vals)
        return max_val,arg_max


    def get_q_value(self,data,state,action):
        if not self.str_sol(state) in self.q_table[self.nbrUn(state)]:
            self.q_table[self.nbrUn(state)][self.str_sol(state)] = {}

        if not str(action) in self.q_table[self.nbrUn(state)][self.str_sol(state)]:
            self.q_table[self.nbrUn(state)][self.str_sol(state)][str(action)] = data.evaluate(self.get_next_state(state,action))
            self.q_table[self.nbrUn(state)][self.str_sol(state)][str(action)] = 0
            
        return self.q_table[self.nbrUn(state)][self.str_sol(state)][str(action)]

    def set_q_value(self,state,action,val):
        self.q_table[self.nbrUn(state)][self.str_sol(state)][str(action)] = val

    def step(self,data,state):
        if np.random.uniform() > self.epsilon :
            action_values = self.actions
            argmax_actions=[] 
            for ac in action_values :

                ac_state_q_val = self.get_q_value(data,state,ac)
                if ( ac_state_q_val >= self.get_max_value(data,state,action_values)[0] ):
                    #print("Q-value for action :" + str(ac) + " is " + str(ac_state_q_val))
                    argmax_actions.append(ac)

            #print("This is argmax list : ",argmax_actions)
            if len(argmax_actions) != 0:
              next_action = np.random.choice(argmax_actions) 
            else:
              next_action = np.random.choice(action_values) 
            next_state = self.get_next_state(state,next_action)
            #print("The next state is :",next_state)
            
            #reward = data.evaluate(next_state)
        else :
            next_action = np.random.choice(self.actions)
            next_state = self.get_next_state(state,next_action)
            
            #reward = reward = data.evaluate(next_state)
            
        if self.epsilon > 0 :
            self.epsilon -= 0.0001 
        if self.epsilon < 0 :
            self.epsilon = 0

        return next_state, next_action #, reward


    def get_next_state(self,state,action):
        next_state = state.copy()
        next_state[action] = (next_state[action]+1) % 2
        if (self.nbrUn(next_state) != 0):
          return next_state
        else:
          return state
    
    def learn(self,data,current_state,current_action,reward,next_state):
        #print("current state : " + self.str_sol(current_state) + "| current action : " + str(current_action) + "| reward : "+ str(reward) + "| next state : "+ self.str_sol(next_state))
        
        next_action = self.step(data,next_state)[1] # step returns 3 values : next_state, next_action, and the reward
        new_q = reward + self.gamma * self.get_q_value(data,next_state,next_action)  #[0] is to pick q-value instead of [1] which is the accuracy of the new state 
        self.set_q_value(current_state,current_action,(1 - self.alpha)*self.get_q_value(data,current_state,current_action) + self.alpha*new_q)  

    #@staticmethod
    def str_sol(self,mlist):
        result = ''
        for element in mlist:
            result += str(element)
        return result

    def nbrUn(self,solution):
        return len([i for i, n in enumerate(solution) if n == 1])

      
      
class FsProblem :
    def __init__(self,data,qlearn):
        self.data=data
        self.nb_attribs= len(self.data.columns)-1 
        self.outPuts=self.data.iloc[:,self.nb_attribs]
        self.ql = qlearn
        self.nb_actions = len(self.ql.actions)
        self.classifier = KNeighborsClassifier(n_neighbors=1)

    def evaluate2(self,solution):
        list=[i for i, n in enumerate(solution) if n == 1]
        if (len(list) == 0):
            return 0
         
        df = self.data.iloc[:,list]
        array=df.values
        nb_attribs =len(array[0])
        X = array[:,0:nb_attribs]
        Y = self.outPuts
        train_X, test_X, train_y, test_y = train_test_split(X, Y, 
                                                    random_state=0,
                                                    test_size=0.1
                                                    )
        classifier = KNeighborsClassifier(n_neighbors=1)
        classifier.fit(train_X,train_y)
        predict= classifier.predict(test_X) 
        return metrics.accuracy_score(predict,test_y)
    
    def evaluate(self,solution):
        list=[i for i, n in enumerate(solution) if n == 1]
        if (len(list)== 0):
            return 0
        df = self.data.iloc[:,list]        
        array=df.values
        nbrAttributs =len(array[0])
        X = array[:,0:nbrAttributs]
        Y = self.outPuts
        cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
        results = cross_val_score(self.classifier, X, Y, cv=cv,scoring='accuracy')
        #print("\n[Cross validation results]\n{0}".format(results))
        return results.mean()
    
    def nbrUn(self,solution):
        return len([i for i, n in enumerate(solution) if n == 1])

class FSData:

    def __init__(self,location,nbr_exec):
        
        self.location = location
        self.nb_exec = nbr_exec
        self.dataset_name = re.search('[A-Za-z\-]*.csv',self.location)[0].split('.')[0]
        self.df = pd.read_csv(self.location,header=None)
        self.ql = QLearning(len(self.df.columns),self.attributs_to_flip(len(self.df.columns)-1))
        self.fsd = FsProblem(self.df,self.ql)
        
        self.classifier_name = str(type(self.fsd.classifier)).strip('< > \' class ').split('.')[3]
        path ='./'
        self.instance_name = str(time.strftime("%d-%m-%Y_%H-%M_", time.localtime()) + self.dataset_name + '_' + self.classifier_name)
        log_filename = str(path + '/logs/'+ self.instance_name)
        
        log_file = open(log_filename + '.txt','w+')
        sys.stdout = log_file
        
        print("[START] Dataset" + self.dataset_name + "description \n")
        print("Shape : " + str(self.df.shape) + "\n")
        print(self.df.describe())
        print("\n[END] Dataset" + self.dataset_name + "description\n")
        print("[START] Ressources specifications\n")
        #!cat /proc/cpuinfo
        print("[END] Ressources specifications\n")

        
        sheet_filename = str(path + '/sheets/'+ self.instance_name )
        self.workbook = xlsxwriter.Workbook(sheet_filename + '.xlsx')
        
        self.worksheet = self.workbook.add_worksheet(self.classifier_name)
        self.worksheet.write(0,0,'Iteration')
        self.worksheet.write(0,1,'Accuracy')
        self.worksheet.write(0,2,'N_Features')
        self.worksheet.write(0,3,'Time')
    
    def attributs_to_flip(self,nb_att):
      
        return list(range(nb_att))
    
    def run(self,typeOfAlgo,flip,maxChance,nbrBees,maxIterations,locIterations):
        t_init = time.time()
        
        for itr in range(1,self.nb_exec+1):
          print ("Execution N°{0}".format(str(itr)))
          self.ql = QLearning(len(self.df.columns),self.attributs_to_flip(len(self.df.columns)-1))
          self.fsd = FsProblem(self.df,self.ql)
          swarm = Swarm(self.fsd,flip,maxChance,nbrBees,maxIterations,locIterations)
          t1 = time.time()
          best = swarm.bso(typeOfAlgo)
          t2 = time.time()
          print("Time elapsed for execution N°{0} : {1:.2f} s\n".format(itr,t2-t1))
          self.worksheet.write(itr, 0, itr)
          self.worksheet.write(itr, 1, "{0:.2f}".format(best[0]))
          self.worksheet.write(itr, 2, best[1])
          self.worksheet.write(itr, 3, "{0:.3f}".format(t2-t1))
          
        t_end = time.time()
        print ("Total execution time for dataset {0} is {1:.2f} s".format(self.dataset_name,t_end-t_init))
        self.workbook.close()

        
# Main program

# Prepare the dataset

dataset = "Iris"
# dataset ="Sonar"
data_loc_path = "./"
location = data_loc_path + dataset + ".csv"

# Params init

typeOfAlgo = 0
nbr_exec = 1
flip = 5
maxChance = 5
nbrBees = 10
maxIterations = 2
locIterations = 2

instance = FSData(location,nbr_exec)
instance.run(typeOfAlgo,flip,maxChance,nbrBees,maxIterations,locIterations)