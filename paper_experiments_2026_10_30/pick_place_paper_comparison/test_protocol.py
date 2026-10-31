import unittest
import numpy as np
import torch
from pick_place_paper_comparison.runtime import ARCHIVE,AblationPolicy,archived_model,engine,archived_env,assert_frozen_imports


class Tests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_time_mask_and_no_history_leak(self):
        x=torch.randn(2,6,49);y=x.clone();y[:,:,21]+=100
        h=torch.randn(2,64);n=torch.tensor([6,6])
        for mode in ('state','context'):
            m=AblationPolicy(mode)
            a=m.forward_episode(x,n,h);b=m.forward_episode(y,n,h)
            self.assertTrue(torch.equal(a[0],b[0]));self.assertTrue(torch.equal(a[2],b[2]))
        z=x.clone();z[:,:-1]+=30
        for mode in ('state','time'):
            m=AblationPolicy(mode)
            a=m.forward_episode(x,n,h);b=m.forward_episode(z,n,h+100)
            self.assertTrue(torch.equal(a[0][:,-1],b[0][:,-1]));self.assertEqual(int(a[2].count_nonzero()),0)

    def test_full_archive_compatibility(self):
        saved=torch.load(ARCHIVE/'half_bc/best.pt',map_location='cpu',weights_only=False)
        m=AblationPolicy();m.load_state_dict(saved['model'])
        original=archived_model.TransformerMetaPolicy();original.load_state_dict(saved['model'])
        x=torch.randn(1,5,49);n=torch.tensor([5]);h=torch.zeros(1,64)
        self.assertTrue(torch.equal(m.forward_episode(x,n,h)[0],original.forward_episode(x,n,h)[0]))
        assert_frozen_imports()

    def test_masked_rollout_replay_parity(self):
        for mode in ('state','time','context','full'):
            m=AblationPolicy(mode);trial=engine.collect_trial(m,archived_env.PickPlaceTrial(seed=900001),65)
            out=engine.replay(m,[trial]);logs=torch.distributions.Categorical(logits=out['logits']).log_prob(out['actions'])
            np.testing.assert_allclose(logs.detach().numpy(),out['old_log_probs'].numpy(),atol=2e-6)

    def test_target_oracle_and_last_step_boundary(self):
        from pick_place_paper_comparison.evaluate import oracle
        e=archived_env.PickPlaceTrial(seed=72);e.reset(30)
        self.assertEqual(oracle(e)['optimal_targets'],['S'])
        for target in range(5):
            e.reset(150);duration=e.duration(4+target,e.home)
            positions={label:tuple(xy[:2]) for label,xy in zip(archived_env.LABELS,e.initial_balls)}
            scores=e._scores.copy();e.reset(duration,offsets=positions,scores=scores)
            _,reward,_,info=e.step(4+target)
            self.assertTrue(info['episode_record']['delivered']);self.assertEqual(reward,scores[target]/4)
        e.reset(150);self.assertEqual(oracle(e)['max_score'],4.)

    def test_uniform_paired_tasks(self):
        from pick_place_paper_comparison.evaluate import tasks
        self.assertEqual(tasks(),tasks());self.assertEqual(len(tasks()),100)
        self.assertTrue(all(30<=t['deadline']<=150 for t in tasks()))

    def test_inspection_cannot_shorten_target_delivery(self):
        for seed in range(10):
            e=archived_env.PickPlaceTrial(seed=seed);e.reset(150)
            for target in range(5):
                direct=e.duration(4+target,e.home)
                for inspect in range(4):
                    contact=e.initial_balls[inspect+1]+[0,0,.005]
                    detour=e.duration(inspect,e.home)+e.duration(4+target,contact)
                    self.assertGreaterEqual(detour,direct)


if __name__=='__main__':unittest.main()
