"""Task-state checks; physics/raycast integration is checked separately."""
from dataclasses import replace
import unittest
import numpy as np
from environment import GateTrial, STATE_DIM
from layout import Config


class FakeBackend:
    name='test-only'
    failed=False
    wall_contact=False
    def set_task(self,doors): self.doors=np.array(doors).copy()
    def reset(self): self.pose=np.zeros(3);self.velocity=np.zeros(3)
    def local_map(self): return {'grid':np.zeros((3,64,64),np.float32)}
    def step(self,command,seconds):
        self.velocity=np.array(command);self.pose+=self.velocity*seconds


class EnvironmentTests(unittest.TestCase):
    def make(self,**kwargs): return GateTrial(FakeBackend(),Config(**kwargs))

    def test_no_hidden_gate_labels_in_observations(self):
        a,b=self.make(),self.make()
        oa,ob=a.reset(doors=(False,False)),b.reset(doors=(True,True))
        self.assertEqual(oa['state'].shape,(STATE_DIM,))
        self.assertEqual(set(oa),{'state','grid'})
        for key in oa: np.testing.assert_array_equal(oa[key],ob[key])

    def test_goal_coordinate_scale_is_independent_of_lidar_range(self):
        env=self.make(sensor_range=2.)
        obs=env.reset()
        self.assertEqual(obs['state'][4],.75)

    def test_teacher_does_not_interpret_out_of_range_misses_as_open(self):
        from teacher import observe_gate
        scan=dict(points=np.zeros((3,3)),distances=np.full(3,np.nan),
                  origins=np.array([0.,0.,.4]),directions=np.tile([1.,0.,0.],(3,1)))
        self.assertIsNone(observe_gate(scan,0,2.))
        self.assertTrue(observe_gate(scan,0,5.))

    def test_pickup_alone_does_not_end_episode(self):
        env=self.make(goal=(.9,0.))
        env.reset(deadline=10.,doors=(False,True))
        env.step([1.,0.])
        obs,reward,done,info=env.step([1.,0.])
        self.assertTrue(info['picked_up']);self.assertTrue(obs['state'][12])
        self.assertFalse(done);self.assertEqual(reward,0.)
        obs,reward,done,info=env.step([-1.,0.])
        self.assertTrue(done);self.assertTrue(info['delivered']);self.assertEqual(reward,1.)
        self.assertFalse(info['trial_done'])

    def test_timeout_then_next_episode_preserves_task(self):
        env=self.make()
        env.reset(deadline=.5,doors=(False,True))
        for index in range(3):
            obs,reward,done,info=env.step([0.,0.])
            self.assertTrue(done);self.assertEqual(reward,0.)
            self.assertEqual(info['trial_done'],index==2)
            if index<2:
                next_obs=env.next_episode()
                np.testing.assert_array_equal(env.backend.doors,[False,True])
                self.assertFalse(next_obs['state'][12]);self.assertEqual(env.elapsed,0.)
        with self.assertRaises(RuntimeError): env.next_episode()

    def test_no_danger_or_random_value_rewards(self):
        env=self.make()
        env.reset(deadline=1.,doors=(True,False))
        self.assertEqual(env.step([0.,0.])[1],0.)
        self.assertNotIn('scores',env.metadata()['task'])
        self.assertNotIn('danger',env.metadata()['task'])

    def test_fallback_reward_and_one_delivery(self):
        env=self.make(fallback_goal=(.9,0.))
        env.reset(deadline=10.)
        env.step([1.,0.]);obs,_,done,info=env.step([1.,0.])
        self.assertEqual(info['pickup_kind'],'fallback')
        self.assertEqual(obs['state'][13],.25)
        _,reward,done,info=env.step([-1.,0.])
        self.assertTrue(done);self.assertEqual(reward,.25)
        self.assertEqual(info['delivered_value'],1.)
        with self.assertRaises(RuntimeError): env.step([0.,0.])

    def test_high_value_is_never_downgraded(self):
        env=self.make(fallback_goal=(.9,0.))
        env.reset(deadline=10.);env.carrying=True;env.carried_value=4.
        env.backend.pose[:2]=[.9,0.]
        env.step([0.,0.])
        self.assertEqual(env.carried_value,4.)

    def test_fall_at_base_does_not_deliver(self):
        env=self.make();env.reset();env.carrying=True;env.backend.failed=True
        _,reward,done,info=env.step([0.,0.])
        self.assertTrue(done);self.assertFalse(info['delivered']);self.assertEqual(reward,0.)

    def test_invalid_controls_and_deadlines_rejected(self):
        env=self.make()
        with self.assertRaises(ValueError): env.reset(deadline=.3)
        env.reset()
        for action in ([0.,0.,0.],[float('nan'),0.]):
            with self.assertRaises(ValueError): env.step(action)


if __name__=='__main__': unittest.main()
