import numpy as np
import cPickle
import irl.maxent as maxent

def map_states_actions(datapath):
    states = dict()
    inv_states = list()
    actions = dict()
    inv_actions = list()
    transitions = list()

    pkl_file = open(datapath, 'rb')
    data = cPickle.load(pkl_file)

    state_cnt = 0
    action_cnt = 0
    for state in data:
        if states.has_key(state) is False:
            states[state] = state_cnt
            inv_states.append(state)
            state_cnt += 1
        for action in data[state]:
            action = action.strip()
            if actions.has_key(action) is False:
                actions[action] = action_cnt
                inv_actions.append(action)
                action_cnt += 1
    
    pkl_file.close()
    return states, actions, inv_states, inv_actions

def load_feature(datapath, states, actions):
    state_size = len(states)
    action_size = len(actions)

    MDP_states = dict()
    MDP_feature = list()
    MDP_trans = dict()
    inv_MDP_states = list()

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
                    inv_MDP_states.append(MDP_state)
                    MDP_feature.append(np.array(feature[state][action][i]))
                    MDP_state_cnt += 1
    pkl_file.close()
    return MDP_states, MDP_feature, MDP_trans, inv_MDP_states

if __name__ == "__main__":
    SApairs_path = 'data_mdp/SA_pairs.pkl'
    traj_path = 'data_mdp/cas.mdps.test.pkl'
    weights_path = 'weights_1.pkl'

    traj_file = open(traj_path, 'rb')
    trajectories = cPickle.load(traj_file)
    traj_file.close()

    weights_file = open(weights_path, 'rb')
    weights = cPickle.load(weights_file)
    weights_file.close()

    states, actions, inv_states, inv_actions = map_states_actions(SApairs_path)
    MDP_states, MDP_feature, MDP_trans, inv_MDP_states = load_feature(SApairs_path, states, actions)
    MDP_feature = np.array(MDP_feature)

    outputs = list()
    n_actions = len(actions)

    for trajectory in trajectories:
        visit = np.zeros((len(states), len(actions)))
        output = list()
        idx = 0
        now_grid = None
        last_state, last_action = None, None
        verified = False
        while idx < len(trajectory):
            start_state = states[trajectory[idx][0]]
            score = list()
            for action in range(n_actions):
                if MDP_trans.has_key((start_state, action)) is False:
                    continue
                for new_state, _ in MDP_trans[(start_state, action)]:
                    MDP_state = MDP_states[(start_state, action, new_state)]
                    score.append((-np.dot(weights, MDP_feature[MDP_state]), MDP_state))
            score = sorted(score)

            if now_grid != start_state:
                verified = False
            for ans in score:
                MDP_state = inv_MDP_states[ans[1]]
                if "verify" in inv_actions[MDP_state[1]]:
                    if verified is True:
                        continue
                if visit[MDP_state[0], MDP_state[1]] == 1:
                    continue
                nxt_state = MDP_state[2]
                check = False 
                for nxt in range(idx, len(trajectory)):
                    if nxt_state==states[trajectory[nxt][0]] or nxt_state==states[trajectory[-1][2][-1]]:
                        check = True
                        idx = nxt
                        output.append((inv_states[MDP_state[0]], inv_actions[MDP_state[1]], 
                                    inv_states[MDP_state[2]]))
                        visit[MDP_state[0], MDP_state[1]] = 1
                        if "verify" in inv_actions[MDP_state[1]]:
                            verified = True
                        now_grid = MDP_state[2]
                        break
                if check is True:
                    break
            if now_grid == states[trajectory[-1][2][-1]]:
                break
        outputs.append(output)        
        print len(outputs)
        if len(outputs)>5000:
            break
#    for line in outputs:
#        print line
    for i in range(len(outputs)):
        print "===============output"
        print outputs[i]
        print "===============original"
        print trajectories[i]
        print ""
