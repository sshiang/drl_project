import numpy as np
import cPickle
import irl.maxent as maxent

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
                actions[action] = action_cnt
                action_cnt += 1
    
    pkl_file.close()
    return states, actions

def load_feature(datapath, states, actions):
    state_size = len(states)
    action_size = len(actions)

    MDP_states = dict()
    MDP_feature = list()
    MDP_trans = dict()

    pkl_file = open(datapath, 'rb')
    SAS = cPickle.load(pkl_file)
    feature = cPickle.load(pkl_file)

    MDP_state_cnt = 0
    for state in SAS:
        for action in SAS[state]:
#            print(action)
#            print(SAS[state][action])
            trans_state = (states[state], actions[action])
            MDP_trans[trans_state] = list()
            prob = 1.0/float(len(SAS[state][action]))
            for i, new_state in enumerate(SAS[state][action]):
#                new_state1 = new_state
                new_state = new_state[-1]
                assert states.has_key(new_state) is True
                MDP_state = (states[state], actions[action], states[new_state])
#                assert MDP_states.has_key(MDP_state) is False
                MDP_trans[trans_state].append((states[new_state], prob))
                """
                if MDP_states.has_key(MDP_state) is True:
                    print ("====start")
                    print state
                    print new_state1
                    print SAS[state][action]
                    print state, action, new_state
                    print ("====old")
                    print MDP_feature[MDP_states[MDP_state]]
                    print ("====new")
                    print feature[state][action][i]
                    print ("====end")
                    assert False
                """
                if MDP_states.has_key(MDP_state) is False:
                    MDP_states[MDP_state] = MDP_state_cnt
                    MDP_feature.append(np.array(feature[state][action][i]))
                    MDP_state_cnt += 1
    pkl_file.close()
    return MDP_states, MDP_feature, MDP_trans

if __name__ == "__main__":
    SApairs_path = 'data_mdp/SA_pairs.pkl'
    traj_path = 'data_mdp/cas.mdps.train.pkl'
    weights_path = 'weights.pkl'

    traj_file = open(traj_path, 'rb')
    trajectories = cPickle.load(traj_file)
    traj_file.close()

    states, actions = map_states_actions(SApairs_path)
    MDP_states, MDP_feature, MDP_trans = load_feature(SApairs_path, states, actions)
    MDP_feature = np.array(MDP_feature)
    weights = maxent.irl(MDP_feature, len(actions), 0.9, states, actions, MDP_states, MDP_trans,
                        trajectories, epochs=100, learning_rate = 0.1)

    weights_file = open(weights_path, 'wb')
    cPickle.dump(weights, weights_file)
    weights_file.close()
