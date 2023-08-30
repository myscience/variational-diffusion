import torch
import unittest

from torch import Tensor

from src.module.monotonic import MonotonicLinear
from src.schedule import LearnableSchedule

class ScheduleTest(unittest.TestCase):
    def setUp(self) -> None:
        pass        

    def test_monotonic_linear(self):

        mono = MonotonicLinear(
            3, 3, bias=True,
            gate_func='relu',
            indicator=torch.tensor((+1, +1, -1)),
            act_weight=(1, 1, 1),
        )

        time_zero = torch.rand((10, 3))

        time_add1 = time_zero + torch.tensor((1, 0, 0))
        time_add2 = time_zero + torch.tensor((0, 1, 0))
        time_add3 = time_zero + torch.tensor((0, 0, 1))

        time_sub1 = time_zero - torch.tensor((1, 0, 0))
        time_sub2 = time_zero - torch.tensor((0, 1, 0))
        time_sub3 = time_zero - torch.tensor((0, 0, 1))

        out_zero = mono(time_zero)

        out_add1 = mono(time_add1)
        out_add2 = mono(time_add2)
        out_add3 = mono(time_add3)

        out_sub1 = mono(time_sub1)
        out_sub2 = mono(time_sub2)
        out_sub3 = mono(time_sub3)

        # Layer should be positive monotonic along dim=[0, 1],
        # while being negative monotonic along dim=2
        mono_pos_dim0 = torch.all(out_add1 >= out_zero)
        mono_pos_dim1 = torch.all(out_add2 >= out_zero)
        mono_neg_dim2 = torch.all(out_add3 <= out_zero)

        mono_neg_dim0 = torch.all(out_sub1 <= out_zero)
        mono_neg_dim1 = torch.all(out_sub2 <= out_zero)
        mono_pos_dim2 = torch.all(out_sub3 >= out_zero)
        
        # Check monotonicity
        self.assertTrue(torch.all(mono_pos_dim0))        
        self.assertTrue(torch.all(mono_pos_dim1))        
        self.assertTrue(torch.all(mono_neg_dim2))
        
        self.assertTrue(torch.all(mono_neg_dim0))        
        self.assertTrue(torch.all(mono_neg_dim1))        
        self.assertTrue(torch.all(mono_pos_dim2))        

    def test_schedule(self):

        schedule = LearnableSchedule(
            hid_dim=[50, 50],
            gate_func='relu',
        )

        time = torch.rand((10, 1))
        time_p1 = time + 1

        out_t0 : Tensor = schedule(time)
        out_t1 : Tensor = schedule(time_p1)

        # Check monotonicity
        self.assertTrue(torch.all(out_t0 >= out_t1))