import csv
import unittest

from . import reproduce


class ReleaseTests(unittest.TestCase):
    def test_current_cli_defaults_use_retained_models_and_range(self):
        from unittest.mock import patch
        from tidyup_plan2.baseline3_gate_meta_rl.continuation_h_comparison import train
        for mode,deadline in (('episode',25),('step',19)):
            with patch('sys.argv',['train.py','--mode',mode]):
                args = train.parse_args()
            self.assertEqual(args.total_steps,645000)
            self.assertEqual(args.closed_long_deadline,deadline)
            self.assertTrue(train.DEFAULT_CHECKPOINTS[mode].exists())
            self.assertTrue(args.output_dir.is_relative_to(reproduce.HERE/'runs'))

    def test_retained_resume_models_restore_optimizer_and_schedule(self):
        import torch
        from tidyup_plan2.baseline3_gate_meta_rl.continuation_h_comparison import train
        torch.set_num_threads(1)
        for mode,path in train.DEFAULT_CHECKPOINTS.items():
            self.assertTrue(path.is_file())
            checkpoint,args,model,optimizer,steps,updates,trials = train.load_runtime(path,mode)
            self.assertEqual(steps,reproduce.STARTS[mode])
            self.assertEqual(trials,updates*args.trials_per_update)
            for name,value in model.state_dict().items():
                self.assertTrue(torch.equal(value,checkpoint['model_state_dict'][name]))
            restored = optimizer.state_dict()
            self.assertEqual(restored['param_groups'],checkpoint['optimizer_state_dict']['param_groups'])
            for key,original in checkpoint['optimizer_state_dict']['state'].items():
                for field,value in original.items():
                    self.assertTrue(torch.equal(restored['state'][key][field],value))
            self.assertAlmostEqual(train.episode.bc_coefficient(steps,args),1-steps/600000)
            self.assertEqual(train.episode.bc_coefficient(600000,args),0)

    def test_closed25_experts_use_only_the_six_final_tasks(self):
        import numpy as np
        from tidyup_plan2.baseline3_gate_meta_rl.continuation_h_comparison import train
        _,args,_,_,_,_,_ = train.load_runtime(train.DEFAULT_CHECKPOINTS['episode'],'episode')
        args.closed_long_deadline = 25
        env = train.episode.make_env(args)
        try:
            experts = train.build_demonstrations(env,args)
        finally:
            env.close()
        self.assertEqual({(t.gate_state,t.episodes[0].deadline) for t in experts},
                         {('open',15),('open',17),('open',19),('closed',15),('closed',17),('closed',25)})
        closed25 = next(t for t in experts if t.gate_state=='closed' and t.episodes[0].deadline==25)
        self.assertEqual([e.steps for e in closed25.episodes],[22,18,18])
        np.testing.assert_allclose([e.episode_return for e in closed25.episodes],[.78,.82,.82],atol=1e-6)

    def test_integrity_and_last_episode_points(self):
        manifest = reproduce.validate_files()
        self.assertEqual(manifest['endpoints'], reproduce.ENDPOINTS)
        self.assertGreater(len(manifest['sha256']), 60)
        for mode, points in reproduce.data()['curves'].items():
            for p in points:
                self.assertFalse(p['source'].startswith('/'))
                cases = {tuple(case) for case in p['task_cases']}
                self.assertEqual(len(cases), 6)
                long_deadline = 25 if mode == 'episode' and p['steps'] >= 551179 else 19
                self.assertIn(('closed', long_deadline), cases)
                self.assertNotIn(('open', 25), cases)

    def test_logs_stop_at_the_retained_models(self):
        for path in (reproduce.HERE / 'training_logs').glob('*.csv'):
            mode = path.name.split('_')[0]
            cutoff = reproduce.STARTS[mode] if 'initial' in path.name else reproduce.ENDPOINTS[mode]
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(rows)
            self.assertLessEqual(max(int(r['total_steps']) for r in rows), cutoff)


if __name__ == '__main__':
    unittest.main()
