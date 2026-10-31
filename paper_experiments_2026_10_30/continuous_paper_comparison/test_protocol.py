import unittest
from types import SimpleNamespace
import numpy as np
import torch
from continuous_paper_comparison.runtime import ARCHIVE,AblationPolicy,core,assert_frozen_imports


class Tests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_no_explicit_time_or_context_leak(self):
        x=torch.randn(8,63);y=x.clone();y[:,-1]+=20;h=torch.randn(32)
        for mode in ('state','context'):
            m=AblationPolicy(mode=mode)
            self.assertTrue(torch.equal(m(x,h)[0].logits,m(y,h)[0].logits))
        for mode in ('state','time'):
            m=AblationPolicy(mode=mode)
            self.assertTrue(torch.equal(m(x,h)[0].logits,m(x,h+10)[0].logits))

    def test_trajectory_encoder_time_mask(self):
        def episode(time):
            x=np.ones((10,63),np.float32);x[:,-1]=time
            return SimpleNamespace(observations=x,next_observations=x,actions=np.ones(10,dtype=np.int64),
                                   rewards=np.zeros(10,np.float32),dones=np.zeros(10,np.float32))
        m=AblationPolicy(mode='context');h=torch.zeros(32)
        self.assertTrue(torch.equal(m.update_hidden(h,episode(1)),m.update_hidden(h,episode(50))))
        for mode in ('state','time'):
            self.assertEqual(int(AblationPolicy(mode=mode).update_hidden(h+1,episode(1)).count_nonzero()),0)

    def test_full_archive_weight_compatibility(self):
        saved=torch.load(ARCHIVE/'best_model.pt',map_location='cpu',weights_only=False)
        original=core.TrialPolicy(63);original.load_state_dict(saved['model_state_dict'])
        m=AblationPolicy();m.load_state_dict(saved['model_state_dict'])
        x=torch.randn(63);h=torch.randn(32)
        self.assertTrue(torch.equal(original(x,h)[0].logits,m(x,h)[0].logits))
        assert_frozen_imports()

    def test_uniform_test_tasks_reproducible_and_in_range(self):
        from continuous_paper_comparison.evaluate import tasks
        a=tasks();self.assertEqual(a,tasks());self.assertEqual(len(a),100)
        self.assertTrue(all(75<=t['deadline']<=230 for t in a))
        self.assertGreater(len({t['deadline'] for t in a}),50)
        self.assertEqual({t['gate'] for t in a},{'open','closed'})

    def test_validated_route_reference(self):
        from continuous_paper_comparison.evaluate import reference_decisions
        bank={'direct':dict(steps=70,danger_step=None),'upper':dict(steps=128,danger_step=100),
              'lower':dict(steps=155,danger_step=None)}
        self.assertEqual(set(reference_decisions('open',230,bank)[0]),{'direct','lower'})
        self.assertEqual(reference_decisions('closed',145,bank)[0],['upper'])
        self.assertEqual(reference_decisions('closed',230,bank)[0],['lower'])
        self.assertFalse(reference_decisions('closed',75,bank)[1])
        bank['upper']={'variants':[dict(steps=128,danger_step=100),dict(steps=140,danger_step=None)]}
        self.assertEqual(reference_decisions('closed',145,bank)[2]['upper'],1.)
        self.assertEqual(reference_decisions('closed',130,bank)[2]['upper'],.8)


if __name__=='__main__':unittest.main()
