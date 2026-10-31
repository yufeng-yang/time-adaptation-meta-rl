import unittest
import numpy as np
import torch
from discrete_paper_comparison.train import AblationPolicy, engine


class ProtocolTests(unittest.TestCase):
    def test_absent_inputs_cannot_change_logits(self):
        torch.set_num_threads(1)
        x = torch.randn(6, engine.base.OBS_DIM)
        y = x.clone(); y[:, -1] += 50
        h = torch.randn(32)
        for mode in ('state', 'context'):
            m = AblationPolicy(mode=mode)
            self.assertTrue(torch.equal(m(x,h)[0], m(y,h)[0]))
        for mode in ('state', 'time'):
            m = AblationPolicy(mode=mode)
            self.assertTrue(torch.equal(m(x,h)[0],m(x,h+10)[0]))

    def test_time_hidden_encoder_cannot_see_deadline(self):
        m = AblationPolicy(mode='context')
        x = torch.randn(8, engine.base.OBS_DIM); y=x.clone(); y[:,-1]+=5
        a=torch.zeros(8,dtype=torch.long);r=torch.zeros(8);d=torch.zeros(8)
        h=torch.zeros(32)
        self.assertTrue(torch.equal(m.update_hidden(h,x,a,r,x,d), m.update_hidden(h,y,a,r,y,d)))

    def test_no_context_stays_zero(self):
        x=torch.randn(8,engine.base.OBS_DIM)
        for mode in ('state','time'):
            m=AblationPolicy(mode=mode)
            h=m.update_hidden(torch.ones(32),x,torch.zeros(8,dtype=torch.long),
                              torch.zeros(8),x,torch.zeros(8))
            self.assertEqual(int(h.count_nonzero()),0)

    def test_full_checkpoint_architecture_compatible(self):
        original=engine.TransformerTrialActorCritic()
        m=AblationPolicy(mode='full');m.load_state_dict(original.state_dict())
        x=torch.randn(engine.base.OBS_DIM);h=torch.randn(32)
        self.assertTrue(torch.equal(m(x,h)[0],original(x,h)[0]))

    def test_oracle_ties_and_infeasible_case(self):
        from discrete_paper_comparison.evaluate import oracle_routes
        oracle=oracle_routes(engine.parse_args([]))
        self.assertEqual(oracle['open_D25']['optimal_routes'],['direct','detour'])
        self.assertEqual(oracle['closed_D17']['optimal_routes'],['medium'])
        self.assertEqual(oracle['closed_D25']['optimal_routes'],['detour'])
        self.assertEqual(oracle['closed_D15']['optimal_routes'],['direct','detour'])
        self.assertFalse(oracle['closed_D15']['feasible'])

    def test_decision_after_probe_and_return(self):
        from discrete_paper_comparison.evaluate import selected_route
        self.assertEqual(selected_route([(1,4),(2,4),(3,4),(4,4),(3,4),(3,5)]),'detour')
        self.assertIsNone(selected_route([(3,4),(4,4),(3,4),(3,4)]))

    def test_tasks_are_random_paired_not_stratified(self):
        from discrete_paper_comparison.evaluate import task_set
        a=task_set();self.assertEqual(a,task_set());self.assertEqual(len(a),100)
        self.assertEqual(len({(t['gate'],t['deadline']) for t in a}),6)
        self.assertEqual(len({t['seed'] for t in a}),100)

    def test_ppo_replay_parity_for_each_ablation(self):
        args=engine.parse_args([]);env=engine.make_env(args)
        try:
            for mode in ('state','time','context','full'):
                model=AblationPolicy(mode=mode)
                trial=engine.base.collect_trial(model,env,'closed',4545,0,args,
                    torch.device('cpu'),trial_deadline=17)
                hidden=model.initial_hidden(torch.device('cpu'))
                for ep in trial.episodes:
                    obs,actions,rewards,nxt,dones=engine.base.episode_tensors(ep,torch.device('cpu'))
                    logits,_=model(obs,hidden)
                    logs=torch.distributions.Categorical(logits=logits).log_prob(actions)
                    np.testing.assert_allclose(logs.detach().numpy(),ep.old_log_probs,atol=2e-6)
                    hidden=model.update_hidden(hidden,obs,actions,rewards,nxt,dones)
        finally:env.close()


if __name__=='__main__': unittest.main()
