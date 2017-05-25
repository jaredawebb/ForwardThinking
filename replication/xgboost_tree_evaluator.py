class xgboost_tree_evaluator():

    '''
    How to use this class:
    
    import xgboost
    import xgboost_tree_evaluator
    
    # Prep data
    
    xgb = xgboost.XGBClassifier(params)
    model = xgb.fit(X, y)
    
    evaluator = xgboost_tree_evaluator()
    
    tree_outputs = evaluator.evaluate_trees(model, data)
    
    NOTE:  X must be a numpy array!  Pandas DataFrames would need more work.
    
    On my machine I was able to evaluate mnist.train.images on 100 trees in
    less than a second.  This should hopefully be fast enough.
    '''
    
    def __init__(self):
        self.tree_dicts = []
        self.outputs = []
    
    def create_tree(self, tree_str):

        decisions = re.findall('[0-9*]+:+.*no=+[0-9]*', tree_str)
        leafs = re.findall('[0-9*]+:+leaf.*', tree_str)

        tree = {}
        for decision in decisions:
            k = int(decision.rsplit(':')[0])

            ineq = re.findall('\[(.*?)\]', decision)[0].split('<')
            p = int(ineq[0][1:])
            v = float(ineq[1])

            yes = int(re.findall('yes=+[0-9]*', decision)[0].split('=')[1])
            no = int(re.findall('no=+[0-9]*', decision)[0].split('=')[1])

            tree[k] = (p, v, yes, no)

        for leaf in leafs:

            k = int(leaf.rsplit(':')[0])
            v = float(leaf.split('=')[1])

            tree[k] = v

        return tree

    def eval_tree(self, tree, images):

        i = 0

        curr = tree[i]
        outputs = []

        for image in images:
            #count = 0
            while type(curr) == tuple:# and count < 10:
                #print(curr)
                if image[curr[0]] < curr[1]:
                    curr = tree[curr[2]]
                else:
                    curr = tree[curr[3]]
                #count+=1
            outputs.append(curr)

        return outputs

    def evaluate_trees(self, model, data):
        
        booster = model.booster()
        trees = booster.get_dump()
        
        for i, tree in enumerate(trees):
            #print(i)
            curr_tree = self.create_tree(tree)
            outputs = self.eval_tree(curr_tree, data)
            
            self.tree_dicts.append(curr_tree)
            self.outputs.append(outputs)
            
        return np.array(self.outputs)    