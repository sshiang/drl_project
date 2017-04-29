import numpy as np
import cPickle

#load SA pairs, obtain all S first.
pkl_file = open('data_mdp/SA_pairs.pkl', 'rb')
data = cPickle.load(pkl_file)

def map_states_actions(datapath):
    states = dict()
    actions = dict()
    transitions = list()

    pkl_file = open(datapath, 'rb')
    data = cPickle.load(pkl_file)

    state_cnt = 0
    action_cnt = 0
    for state in data:
        if states.has_key(state) is False:
            states[state] = state_cnt
            state_cnt += 1
        for action in data[state]:
            action = action.strip()
            if actions.has_key(action) is False:
                actions[action] = action
                action_cnt += 1
    
    pkl_file.close()
    return states, actions

def load_feature(datapath, states, actions):
    state_size = len(states)
    action_size = len(actions)

    MDP_states = dict()
    MDP_feature = list()
#    transition_matrix = np.zeros(state_size, action_size, state_size)

    pkl_file = open(datapath, 'rb')
    SAS = cPickle.load(pkl_file)
    feature = cPickle.load(pkl_file)

    MDP_state_cnt = 0
    for state in SAS:
        for action in SAS[state]:
#            print(action)
#            print(SAS[state][action])
            for i, new_state in enumerate(SAS[state][action]):
                new_state1 = new_state
                new_state = new_state[-1]
                assert states.has_key(new_state) is True
                MDP_state = (states[state], actions[action], states[new_state])
#                assert MDP_states.has_key(MDP_state) is False
                if MDP_states.has_key(MDP_state) is True:
                    print ("====start")
                    print new_state1
                    print state, action, new_state
                    print ("====old")
                    print MDP_feature[MDP_states[MDP_state]]
                    print ("====new")
                    print feature[state][action][i]
                    print ("====end")
                    assert False
                MDP_states[MDP_state] = MDP_state_cnt
                MDP_feature.append(feature[state][action][i])
                MDP_state_cnt += 1
#               feature_matrix.append(MDP_feature)
    print MDP_state_cnt
    pkl_file.close()
    return MDP_states, feature_matrix

if __name__ == "__main__":
    SApairs_path = 'data_mdp/SA_pairs.pkl'
    states, actions = map_states_actions(SApairs_path)
    MDP_states, feature_matrix, transition_matrix = load_feature(SApairs_path, states, actions)
