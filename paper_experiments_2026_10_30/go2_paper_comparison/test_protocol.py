import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'runtime_source/gates'))
import torch
from model import MetaPolicy,log_probability
from original_model import MetaPolicy as Original
from test_training import TrainingTests
from train import replay
class AblationTests(TrainingTests):
    def test_masks_and_history(self):
        torch.manual_seed(5)
        states=torch.randn(2,5,19);grids=torch.randn(2,5,3,64,64)
        lens=torch.tensor([5,5]);hidden=torch.randn(2,64)
        for mode in ('state','time','context'):
            m=MetaPolicy(mode=mode)
            a=m.forward_episode(states,grids,lens,hidden)[0]
            if mode in ('state','context'):
                changed=states.clone();changed[...,10:12]+=50
                torch.testing.assert_close(a,m.forward_episode(changed,grids,lens,hidden)[0])
            if mode in ('state','time'):
                st=states.clone();gr=grids.clone();st[:,:-1]+=20;gr[:,:-1]+=20
                b=m.forward_episode(st,gr,lens,hidden*20)[0]
                torch.testing.assert_close(a[:,-1],b[:,-1])
    def test_all_variant_replay(self):
        for mode in ('state','time','context','full'):
            self.model=MetaPolicy(mode=mode)
            out=replay(self.model,self.rollout())
            torch.testing.assert_close(log_probability(out['mean'],self.model.log_std,out['raw_actions']),out['log_probs'],atol=2e-5,rtol=2e-5)
    def test_full_exact(self):
        a=Original();b=MetaPolicy(mode='full');b.load_state_dict(a.state_dict())
        s=torch.randn(2,4,19);g=torch.randn(2,4,3,64,64);l=torch.tensor([4,3]);h=a.initial_hidden(2)
        for x,y in zip(a.forward_episode(s,g,l,h),b.forward_episode(s,g,l,h)):torch.testing.assert_close(x,y,rtol=0,atol=0)
if __name__=='__main__':unittest.main()
