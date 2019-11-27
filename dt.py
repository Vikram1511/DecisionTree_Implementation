import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from pandas.api.types import is_string_dtype,is_numeric_dtype
import operator
import sys
import json
import copy
import random

nodeCount=0
acc_prune_hist=[]
node_count_hist=[]
def preprocessing(train,test,valid):
    m,n = train.shape
    m_test = test.shape[0]
    m_valid = valid.shape[0]
    df = pd.concat([train,test,valid],axis=0)
    print(df.shape)
    df = regexOperation(df)
    #fnlwgt
    # std_fnlwgt = df['Fnlwgt'].std()
    # threshold_min = df['Fnlwgt'].mean()-0.25*std_fnlwgt
    # threshold_max = df['Fnlwgt'].mean()+0.25*std_fnlwgt
    bins_age = pd.IntervalIndex.from_tuples([(0, 35),(35,120)])
    bins_hours = pd.IntervalIndex.from_tuples([(0,40),(40,50),(50,150)])
    bins_edu = pd.IntervalIndex.from_tuples([(0,9),(9,14),(14,20)])
    # bins_fnlwgt=pd.IntervalIndex.from_tuples([(df['Fnlwgt'].min()-100,threshold_min),(threshold_min,threshold_max),(threshold_max,df['Fnlwgt'].max()+100)])
    # z =pd.cut(df['Fnlwgt'].tolist(),bins_fnlwgt)
    # z.categories=['low','medium','high']
    # df['Fnlwgt']=z
    # df['Fnlwgt'] = df['Fnlwgt'].astype('str')
    x = pd.cut(df['Age'].tolist(),bins_age)
    x.categories = ['upto35','above35']
    df['Age'] = x
    df['Age'] = df['Age'].astype('str')
    y = pd.cut(df['Hour per Week'].tolist(),bins_hours)
    y.categories = ['upto40','40to50','above50']
    df['Hour per Week'] = y
    df['Hour per Week'] = df['Hour per Week'].astype('str')
    ed = pd.cut(df['Education Number'].tolist(),bins_edu)
    ed.categories=['upto8','9to13','14above']
    df['Education Number'] = ed
    # df['Education Number']= df['Education Number'].where(df['Education Number']>10,'others')
    df['Education Number'] =df['Education Number'].astype('str')
    # df['Native Country']= df['Native Country'].where(df['Native Country']=='United-States','Not US')   
    df.drop(['Fnlwgt','Education','Capital Gain','Capital Loss','Rich?'],axis=1,inplace=True)
    df['Relationship'].loc[(df['Relationship']!='Husband') & (df['Relationship']!='Wife') ] ='others'
    # df['Marital Status'].loc[(df['Marital Status']!='Married-civ-spouse') ] ='others'
    df = read_inputs(df)
    print(df.shape)
    return df.iloc[:m,:].astype('uint8'),df.iloc[m:m+m_test,:].astype('uint8'),df.iloc[m+m_test:,:].astype('uint8')


def onehotEncoder(array,k_class,column_name,dict_map):

    '''
    @array -  encoded array
    @k_class - number of unique classes in array
    '''
    #to ensure the type of input is one dimensional array
    if(type(array)==list):
        array = np.array(array)
    assert len(array.shape)==1

    #initializing (m,k_classes) dataframe as one hot encoded
    onehotencoded = np.zeros((array.shape[0],k_class))
    array = array.reshape((array.shape[0],))

    #loop for each data point in array
    for i in range(array.shape[0]):
        onehotencoded[i,array[i]-1] = 1
    onehotencoded = pd.DataFrame(onehotencoded,columns=[column_name+"."+dict_map[i][0] for i in range(len(dict_map))])

    return onehotencoded

def decode(array,mapping):

    '''
        @mapping- python dict where keys and values and encoded and true values respectively
        @array - encoded array
    '''
    #convert to list type
    array_list = array.tolist()
    for i in range(array.shape[0]):
        array_list[i] = mapping[array[i]-1]
    return np.array(array_list)

def print_tree(dtree):
    tree_str = json.dumps(dtree, indent=4)
    tree_str = tree_str.replace("\n    ", "\n")
    tree_str = tree_str.replace('"', "")
    tree_str = tree_str.replace(',', "")
    tree_str = tree_str.replace("{", "")
    tree_str = tree_str.replace("}", "")
    tree_str = tree_str.replace("    ", " | ")
    tree_str = tree_str.replace("  ", " ")

    print(tree_str)
    
def encoder(array,unique_classes):

    '''
        @array -  input categorical feature which is to be encoded 
        @unique_classes -  input list of unique classes in array
    '''
    array = array.reshape((array.shape[0],))

    #initializing a ditionary to store mapping of true classes to encoded value
    mapping={}
    for i in range(len(unique_classes)):
        mapping[unique_classes[i]]=i+1
    
    mapping_sortedlist = sorted(mapping.items(), key=operator.itemgetter(1))
    #encoding
    for j in range(array.shape[0]):
        if(array[j] not in list(mapping.keys())):
            array[j] = None
        else:
            array[j] = list(mapping.keys()).index(array[j])+1
    #returning encoded array
    return array.reshape((array.shape[0],)),mapping_sortedlist


def regexOperation(data):
    data.columns = data.columns.str.lstrip()
    for col in data.columns:
        if(is_string_dtype(data[col])):
            data[col] = data[col].str.lstrip()
    return data

def read_inputs(df):
    dataframe = df.copy()
    #list of catergoircal columns
    categorical_columns = [col for col in dataframe.columns if is_string_dtype(dataframe[col])]
    
    #list of numerical columns
    numeric_columns =  [col for col in dataframe.columns if is_numeric_dtype(dataframe[col])]
    
    #list of unique lables for each categorical columns in dataframe
    label_order = [list(dataframe[name].unique()) for name in categorical_columns]
    

    resultant_dataframe = dataframe
    resultant_dataframe.reset_index(inplace=True)
    
    for i,column in enumerate(categorical_columns):
            uniquec = label_order[i]
            feat= dataframe[column].values
            arr,map_train = encoder(feat,uniquec)
            cc = onehotEncoder(arr,len(uniquec),column,map_train)
            resultant_dataframe=pd.concat([resultant_dataframe,cc],axis=1)
            
    categorical_columns2 = [col for col in resultant_dataframe.columns if is_string_dtype(resultant_dataframe[col])]
    resultant_dataframe.drop(categorical_columns2+['Education Number','index'],axis=1,inplace=True)
    print(resultant_dataframe.columns)    
    #print(Y.shape)
    # X = dataframe
    return resultant_dataframe.astype('uint8')




class Node:
    def __init__(self,feature=None,pos_examples=None,neg_examples=None):
        self.pos_examples = pos_examples
        self.neg_examples = neg_examples
        self.feature = feature
        self.label = None
        self.leftchild=None 
        self.rightchild = None 
        self.parent = None
        self.nodeId=None
        self.isLeaf = False

    def set_nodeValue(self,feature,pos_examples,neg_examples):
        self.pos_examples = pos_examples
        self.neg_examples = neg_examples
        self.feature = feature

    def set_leftchild(self,node):
        self.leftchild = node
        node.parent = self
    
    def set_rightchild(self,node):
        self.rightchild =node
        node.parent=self
    
    def get_rightchild(self):
        return self.rightchild
    
    def get_leftchild(self):
        return self.leftchild
    


class DecisionTree:
    def __init__(self,trainx=None,trainy=None,max_depth=None,column_order_list=None,min_sample_leaf=None,validx=None,validy=None):
        self.trainx = trainx
        self.trainy = trainy
        self.validx=validx
        self.validy = validy
        self.max_depth = max_depth
        self.min_sample_leaf =min_sample_leaf
        self.ordered_column_name = column_order_list
        # self.isroot_set = False
        self.root = Node()
        if(self.ordered_column_name is not None):
            self.feature_importance_ = np.zeros((len(self.ordered_column_name),)).tolist()


    def shannonEntropy(self,data):
        '''
        array of shape (m X n)
        '''
        data = data.copy()
        entropy = 0
        pos_data = data[data[:,-1]==1]
        neg_data = data[data[:,-1]==0]
        if(pos_data.shape[0]==0 or neg_data.shape[0]==0):
            return 0
        prob_pos = (pos_data.shape[0]/data.shape[0])
        prob_neg = (neg_data.shape[0])/data.shape[0]
        entropy = - prob_pos*np.log2(prob_pos) - prob_neg*np.log2(prob_neg) 
        return entropy
    

    def gini_index(self,data):
        data = data.copy()
        if(data.shape[0]==0):
            return 0
        entropy = 0
        pos_data = data[data[:,-1]==1]
        neg_data = data[data[:,-1]==0]
        prob_pos = (pos_data.shape[0]/data.shape[0])
        prob_neg = (neg_data.shape[0])/data.shape[0]
        gini_ind=1-np.power(prob_pos,2)-np.power(prob_neg,2)
        return gini_ind

    def bestFeatureSelect(self,data,criterion='entropy'):
        data_copy = data.copy()
        bestinfo_gain = 0
        best_gini = 0
        init_entropy = self.shannonEntropy(data_copy)
        init_gini = self.gini_index(data_copy)
        best_feat =-1
        for feat in range(data_copy.shape[1]-1):
            yes_response = data_copy[data_copy[:,feat]==1]
            no_response = data_copy[data_copy[:,feat]==0]
            prob_yes =0
            prob_no =0
            entropy_yes=0
            entropy_no =0
            if(criterion=='entropy'):
                if(yes_response.shape[0]>0):
                    entropy_yes = self.shannonEntropy(yes_response) 
                    prob_yes = float(yes_response.shape[0]/data_copy.shape[0])
                if(no_response.shape[0]>0):
                    entropy_no = self.shannonEntropy(no_response)  
                    prob_no = float(no_response.shape[0]/data_copy.shape[0])
                
                info_gain= init_entropy -float(prob_yes*(entropy_yes))-float(prob_no*(entropy_no))
                if(info_gain > bestinfo_gain):
                        bestinfo_gain = info_gain
                        best_feat= feat
            if(criterion=='gini'):
                prob_yes = float(yes_response.shape[0]/data_copy.shape[0])
                prob_no = float(no_response.shape[0]/data_copy.shape[0])
                # pos_yes_response = yes_response[yes_response[:,-1]==1]
                # pos_no_response = no_response[no_response[:,-1]==1]
                # p1 = float(pos_yes_response.shape[0]/yes_response.shape[0])
                # p2 = float(pos_no_response.shape[0]/no_response.shape[0])
                # q1 = 1-p1
                # q2=1-p2
                gini_yes = self.gini_index(yes_response)
                gini_no = self.gini_index(no_response)
                # gini_no = np.power(p2,2)+np.power(q2,2)
                gini =  prob_yes*gini_yes + prob_no*gini_no
                gini_gain = init_gini-gini
                if(gini_gain>best_gini):
                    best_gini = gini
                    best_feat = feat
        if(criterion=='gini'):
            self.feature_importance_[best_feat]  = best_gini
            return best_feat,best_gini
        if(criterion=='entropy'):
            self.feature_importance_[best_feat]  = bestinfo_gain
            return best_feat,bestinfo_gain
    

    def majority_count(self,data):
        data = data.copy()
        pos = data[data[:,-1]==1].shape[0]
        neg = data[data[:,-1]==0].shape[0]
        if(pos>neg):
            return 1
        else:
            return 0
    
    def depthTree(self,node):
        if(node==None):
            return 0
        else:
            h_l = self.depthTree(node.leftchild)
            h_r = self.depthTree(node.rightchild)
            if(h_l>h_r):
                return h_l+1
            else:
                return h_r+1
        
    
    def preOrder(self,root):
        if(root!=None):
            print(root.label)
            self.preOrder(root.leftchild)
            self.preOrder(root.rightchild)
    
    def postOrder(self,root):
        if(root!=None):
            self.postOrder(root.leftchild)
            self.postOrder(root.rightchild)
            print(root.label)
    
    def inorder(self,root):
        if(root!=None):
            self.inorder(root.leftchild)
            print(root.label)
            self.inorder(root.rightchild)


    def countLeafNodes(self,root):
        if(root.leftchild==None and root.rightchild==None):
            return 1
        return self.countLeafNodes(root.leftchild)+self.countLeafNodes(root.rightchild)


    def isTree(self,node):
        if(node is None):
            return False
        if(node.isLeaf==True):
            return False
        if(self.depthTree(node)>2):
            return True
        else:
            return False

    def calculateError(self,node,data):
        error=0
        for i in range(data.shape[0]):
            if(data[i,-1]!=node.label):
                error+=1
        return error


    def prune(self,tree,validation_data):
        global acc_prune_hist
        global node_count_hist
        if(validation_data.shape[0]==0):
            return

        if(self.isTree(tree.leftchild)):
            leftset = validation_data[validation_data[:,self.ordered_column_name.index(tree.leftchild.feature)]==0]
            self.prune(tree.leftchild,leftset)

        if(self.isTree(tree.rightchild)):
            rightset = validation_data[validation_data[:,self.ordered_column_name.index(tree.rightchild.feature)]==1]
            self.prune(tree.rightchild,rightset)

        if(self.isTree(tree.leftchild)==False and self.isTree(tree.rightchild)==False):
            leftset = validation_data[validation_data[:,self.ordered_column_name.index(tree.feature)]==0]
            rightset = validation_data[validation_data[:,self.ordered_column_name.index(tree.feature)]==1]
            if(leftset.shape[0]==0):
                 left_error=0
            else:
                left_error = self.calculateError(tree.leftchild,leftset)
            
            if(rightset.shape[0]==0):
                 right_error=0
            else:
                right_error = self.calculateError(tree.rightchild,rightset)

            error_without_merging = np.power(left_error,2)+np.power(right_error,2)
            error_with_merge = np.power(self.calculateError(tree,validation_data),2)

            if(error_with_merge <= error_without_merging):
                print('merging')
                print(error_without_merging)
                print(error_with_merge)
                tree.leftchild=None
                tree.rightchild=None
                acc_prune_hist.append(self.calculateAccuracy(self.validx,self.validy))
                node_count_hist.append(self.countNodes(self.root))
            else:
                return 

            



    def create_decision_tree(self,data,column_list,TreeNode,criterion='gini'):
        global nodeCount
        best_feat,best_val = self.bestFeatureSelect(data,criterion=criterion)
        #counting number of positive and negative instances
        pos_exp = data[data[:,-1]==1].shape[0]
        neg_exp = data[data[:,-1]==0].shape[0]


        #if number of features in data is one(which is class itself) or count of instances are less than min_samples_leaf, return it as leaf node
        if(data.shape[1]==1 or data.shape[0]<self.min_sample_leaf or best_val==0):
            TreeNode.set_nodeValue(None,pos_exp,neg_exp)
            TreeNode.label = self.majority_count(data)
            TreeNode.isLeaf  =True
            return
        

        #if number of unique classes in dataset is one,(impurity is zero), return it as leaf node
        if(len(np.unique(data[:,-1]))==1):
            TreeNode.set_nodeValue(None,pos_exp,neg_exp)
            label = data[0,-1]
            TreeNode.label = label
            TreeNode.isLeaf =True
            return
        


        #getbest feature

        #find the corresponding feature name from column list
        column_name = column_list[best_feat]

        #set the values for current node
        TreeNode.set_nodeValue(column_name,pos_exp,neg_exp)


        #split the dataset into left and right
        right_split = data[data[:,best_feat]==1]
        right_split = np.hstack((right_split[:,:best_feat],right_split[:,best_feat+1:]))

        left_split = data[data[:,best_feat]==0]
        left_split = np.hstack((left_split[:,:best_feat],left_split[:,best_feat+1:]))


        #make leftchild
        left = Node()
        #set it as left child of current node
        TreeNode.set_leftchild(left)
        left.nodeId = nodeCount
        left.label = self.majority_count(left_split)
        nodeCount+=1


        #make rightchild
        right = Node()
        TreeNode.set_rightchild(right)
        right.label = self.majority_count(right_split)
        right.nodeId=nodeCount
        nodeCount+=1



        # if(self.isroot_set==False):
        #     node = Node(column_name)
        #     node.index=0
        #     self.root = node
        #     self.isroot_set = True
        # else:
        #     node = Node(column_name)
        # node.pos_examples = pos_exp
        # node.neg_examples = neg_exp
        # node.label = self.majority_count(data)

        column_list = column_list[:best_feat]+column_list[best_feat+1:]


        self.create_decision_tree(left_split,column_list,left,criterion) 
        self.create_decision_tree(right_split,column_list,right,criterion)
    

    def get_label(self,feat_vector):
        curr_node = self.root
        while(curr_node!=None):
            feat_splitted = curr_node.feature
            if(feat_splitted is not None):

                feat_index = self.ordered_column_name.index(feat_splitted)
                parent_node = curr_node
                if(feat_vector[0,feat_index]==0):
                    curr_node = curr_node.leftchild
                else:
                    curr_node = curr_node.rightchild
                
                if(curr_node==None):
                    if(parent_node.label!=None):
                        return parent_node.label
                    else:
                        if(parent_node.pos_examples>=parent_node.neg_examples):
                            return 1
                        else:
                            return 0
            else:
                    if(curr_node.label!=None):
                        return curr_node.label
                    else:
                        if(curr_node.pos_examples>=curr_node.neg_examples):
                            return 1
                        else:
                            return 0

        return -1

    def predict(self,X):
        prediction = []
        for i in range(X.shape[0]):
            curr_feat_vector = X[i,:].reshape((1,X.shape[1]))
            label = self.get_label(curr_feat_vector)
            prediction.append(label)
        return prediction

    def calculateAccuracy(self,x,y_true):
        y = self.predict(x)
        acc =[1 if i==j else 0 for (i,j) in zip(y,y_true)]
        return float(sum(acc)/len(acc))*100


    def countNodes(self,node):
        if(node.leftchild is not None and node.rightchild is not None):
            return 2 + self.countNodes(node.leftchild) + self.countNodes(node.rightchild)
        return 0

    def searchNode(self,node,nodeid):
        temp=None 
        res=None
        if(node.isLeaf is False):
            if(node.nodeId==nodeid):
                return node
            else:
                res = self.searchNode(node.leftchild,nodeid)
                if(res is None):
                    res = self.searchNode(node.rightchild,nodeid)
                return res
        else:
            return temp
        
    def post_prune(self,pruneNodesNumber,countNode):
        for i in range(pruneNodesNumber):
            x = random.randint(2,countNode)
            tempNode = Node()
            tempNode = self.searchNode(self.root,x)

            if(tempNode is not None):
                tempNode.leftchild = None
                tempNode.rightchild = None
                tempNode.isLeaf = True
                if(tempNode.pos_examples >= tempNode.neg_examples):
                    tempNode.label = 1
                else:
                    tempNode.label = 0



if __name__ == "__main__":
    train_file = sys.argv[1] 
    valid_file = sys.argv[2]
    test_file = sys.argv[3] 
    valid_pred_file = sys.argv[4]
    test_pred_file = sys.argv[5]
    
    
    train_df = pd.read_csv(train_file)
    testdf = pd.read_csv(test_file)
    valid_df = pd.read_csv(valid_file)
    
    trainy = train_df[" Rich?"].values.reshape((train_df.shape[0],1))
    validy = valid_df[" Rich?"].values.reshape((valid_df.shape[0],1))
    
    trainx,testx,validx = preprocessing(train_df,testdf,valid_df)
    # trainx = read_inputs(train)
    # testx = read_inputs(test)
    # validx = read_inputs(valid)
    
    cols = list(trainx.columns)
    trainx = trainx.values
    print(trainx.shape)
    testx = testx.values
    print(testx.shape)
    validx = validx.values
    print(validx.shape)
    train  = np.hstack((trainx,trainy))

    best_tree = DecisionTree(trainx,trainy,5,cols,min_sample_leaf=3,validx=validx,validy=validy)
    best_tree.create_decision_tree(train,cols,best_tree.root,criterion='gini')
    best_accuracy = best_tree.calculateAccuracy(validx,validy)
    acc_prune_hist.append(best_accuracy)
    init_nodes = best_tree.countNodes(best_tree.root)
    node_count_hist.append(init_nodes)
    # print(best_tree.countNodes(best_tree.root))
    # print("initial_best_acc:",best_accuracy)
    bestTree2 = DecisionTree()
    best_tree2= copy.deepcopy(best_tree)

    number_ofNodes =best_tree2.countNodes(best_tree2.root)
    print("pruning")
    best_tree2.prune(best_tree2.root,np.hstack((validx,validy)))
    # pruneFactor = 0.8
    # c=0
    # print('pruning..')
    # while(c<20):
    #     numberOfpruningNodes = round(pruneFactor*number_ofNodes)
    #     pruneTree = DecisionTree()
    #     pruneTree = copy.deepcopy(best_tree2)
    #     print(pruneTree.countNodes(pruneTree.root))
    #     pruneTree.post_prune(numberOfpruningNodes,pruneTree.countNodes(pruneTree.root))
    #     temp_acc = pruneTree.calculateAccuracy(validx,validy)
    #     print("temp acc:",temp_acc)
    #     if(temp_acc>best_accuracy):
    #         print('accuracy improved:',temp_acc)
    #         best_accuracy = temp_acc
    #         best_tree2 = copy.deepcopy(pruneTree)
    #         number_ofNodes =best_tree2.countNodes(best_tree.root)
    #     c=c+1

    pred = best_tree2.predict(trainx)
    pred_test =best_tree2.predict(testx)
    pred_val = best_tree2.predict(validx)
    with open(test_pred_file,"w") as f:
      f.write("\n".join(str(x) for x in pred_test))
       
    with open(valid_pred_file,"w") as f:
      f.write("\n".join(str(x) for x in pred_val))
    print("initial accuracy:",best_accuracy)
    print("initial nodes:",init_nodes)
    print("final nodes:",best_tree2.countNodes(best_tree2.root))
    print("final accuracy:",best_tree2.calculateAccuracy(validx,validy))
    print(acc_prune_hist)
    print(node_count_hist)
    # print("nodeCount:",nodeCount)
    # print("leaf:", best_tree.countNodes(best_tree.root))