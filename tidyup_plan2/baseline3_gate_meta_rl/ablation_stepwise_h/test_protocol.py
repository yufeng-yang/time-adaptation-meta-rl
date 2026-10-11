"""Causal recurrence, cross-episode memory and recurrent PPO replay checks."""
import argparse
import unittest

import numpy as np
import torch
from torch.distributions import Categorical

from tidyup_plan2.baseline3_gate_meta_rl import train_decay_bc as base
from tidyup_plan2.baseline3_gate_meta_rl.ablation_stepwise_h.train import (
    StepwiseActorCritic, collect_trial, ppo_update,
)


def settings():
    return argparse.Namespace(step_penalty=.01,time_cap=20.,gamma=.99,beta_trial=1.,curriculum_steps=0,
                              ppo_epochs=1,clip_range=.2,value_coef=.5,entropy_coef=.02,max_grad_norm=1.,
                              bc_initial_weight=1.,bc_final_weight=0.,bc_decay_steps=600_000,seed=0)


class MemorySpy:
    def __init__(self):
        self.initializations = 0
        self.inputs, self.terminal_updates = [], []

    def initial_hidden(self, device):
        self.initializations += 1
        return torch.zeros(2,device=device)

    def __call__(self, observation, hidden):
        self.inputs.append(float(hidden[0]))
        return torch.tensor([0.,0.,1.]),torch.tensor(0.)

    def update_transition(self, hidden, observation, action, reward, nxt, done):
        self.terminal_updates.append(done)
        return hidden+1


class StepwiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_every_step_and_terminal_update_cross_episode_reset(self):
        args = settings()
        env = base.make_env(args)
        spy = MemorySpy()
        try:
            for gate in ('open','closed'):
                trial = collect_trial(spy,env,gate,0,0,args,torch.device('cpu'),True,3)
                self.assertEqual([e.deadline for e in trial.episodes],[3]*3)
                self.assertEqual([e.steps for e in trial.episodes],[3]*3)
                self.assertEqual(env.unwrapped.gate_state,gate)
                self.assertEqual(env.unwrapped.episode_in_trial,2)
            self.assertEqual(spy.initializations,2)
            self.assertEqual(spy.inputs,list(map(float,range(9)))*2)
            self.assertEqual(spy.terminal_updates,[0.,0.,1.]*6)
        finally:
            env.close()

    def test_initialization_and_single_transition_match_parent(self):
        torch.manual_seed(7)
        parent = base.TrialActorCritic(32)
        expected_rng = torch.random.get_rng_state()
        torch.manual_seed(7)
        step = StepwiseActorCritic(32)
        torch.testing.assert_close(torch.random.get_rng_state(),expected_rng)
        for name in ('body','actor','critic','transition_encoder','episode_encoder','event_attention'):
            for old,new in zip(getattr(parent,name).parameters(),getattr(step,name).parameters()):
                torch.testing.assert_close(old,new,rtol=0,atol=0)
        hidden = torch.randn(32)
        transition = (torch.randn(1,base.OBS_DIM),torch.tensor([2]),torch.tensor([-.01]),
                      torch.randn(1,base.OBS_DIM),torch.tensor([0.]))
        torch.testing.assert_close(parent.update_hidden(hidden,*transition),step.update_hidden(hidden,*transition),
                                   rtol=1e-5,atol=1e-6)

    def test_online_rollout_matches_full_trial_replay(self):
        args = settings()
        env = base.make_env(args)
        torch.manual_seed(2)
        model = StepwiseActorCritic(32)
        try:
            trial = collect_trial(model,env,'closed',3,0,args,torch.device('cpu'),trial_deadline=7)
            with torch.no_grad():
                logits,values,_ = model.replay_trial(trial)
                actions = torch.as_tensor(np.concatenate([e.actions for e in trial.episodes]))
                logs = Categorical(logits=logits).log_prob(actions)
            torch.testing.assert_close(logs,torch.as_tensor(np.concatenate([e.old_log_probs for e in trial.episodes])),
                                       rtol=1e-5,atol=1e-6)
            torch.testing.assert_close(values,torch.as_tensor(np.concatenate([e.old_values for e in trial.episodes])),
                                       rtol=1e-5,atol=1e-6)
        finally:
            env.close()

    def test_causality_and_gradient_across_terminal(self):
        torch.manual_seed(5)
        model = StepwiseActorCritic(32)
        obs = torch.randn(4,base.OBS_DIM,requires_grad=True)
        actions,rewards,nxt,dones = torch.tensor([0,1,2,0]),torch.randn(4),torch.randn(4,base.OBS_DIM),torch.tensor([0.,1.,0.,1.])
        contexts,_ = model.hidden_sequence(model.initial_hidden(torch.device('cpu')),obs,actions,rewards,nxt,dones)
        changed = rewards.clone()
        changed[2] += 1.
        other,_ = model.hidden_sequence(model.initial_hidden(torch.device('cpu')),obs,actions,changed,nxt,dones)
        torch.testing.assert_close(contexts[:3],other[:3],rtol=0,atol=0)
        self.assertGreater(float((contexts[3]-other[3]).abs().sum().detach()),0.)
        logits,_ = model(obs[3],contexts[3])
        logits.sum().backward()
        # Action 4 is after the first episode's terminal transition at step 2.
        self.assertGreater(float(obs.grad[0].abs().sum()),0.)
        self.assertTrue(torch.isfinite(obs.grad).all())

    def test_joint_ppo_bc_update_is_finite_and_changes_gru(self):
        args = settings()
        env = base.make_env(args)
        torch.manual_seed(0)
        model = StepwiseActorCritic(32)
        optimizer = torch.optim.Adam(model.parameters(),lr=3e-4)
        try:
            trials = [collect_trial(model,env,g,0,0,args,torch.device('cpu'),trial_deadline=5) for g in ('open','closed')]
            demonstrations = base.build_expert_trials(env,args)
            targets,advantages = base.build_meta_targets(trials,args.gamma,args.beta_trial)
            before = model.meta_gru.weight_hh_l0.detach().clone()
            metrics = ppo_update(model,optimizer,trials,targets,advantages,args,torch.device('cpu'),demonstrations,30)
            self.assertTrue(all(np.isfinite(x) for x in metrics.values()))
            self.assertGreater(float((model.meta_gru.weight_hh_l0.detach()-before).abs().sum()),0.)
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
