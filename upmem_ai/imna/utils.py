import torch
import numpy as np
import math

def round_to_fixed(input, integer_bits=16, fraction_bits=16): 
    assert integer_bits >= 1, integer_bits
    # TODO: Deal with unsigned tensors where there is no sign bit
    #       which is the case with activations to convolution that 
    #       are usually the output of a Relu layer
    if integer_bits == 1: 
        return torch.sign(input) - 1 
    delta = math.pow(2.0, -(fraction_bits))
    bound = math.pow(2.0, integer_bits-1) 
    min_val = - bound 
    max_val = bound - 1 
    rounded = torch.floor(input / delta) * delta

    clipped_value = torch.clamp(rounded, min_val, max_val)
    return clipped_value 

def get_shift_and_sign(x, rounding='deterministic'):
    sign = torch.sign(x)
    
    x_abs = torch.abs(x)
    shift = round(torch.log(x_abs) / np.log(2), rounding)
    
    return shift, sign    

def round_power_of_2(x, rounding='deterministic'):
    shift, sign = get_shift_and_sign(x, rounding)    
    x_rounded = (2.0 ** shift) * sign
    return x_rounded

def round(x, rounding='deterministic'):
    assert(rounding in ['deterministic', 'stochastic'])
    if rounding == 'stochastic':
        x_floor = x.floor()
        return x_floor + torch.bernoulli(x - x_floor)
    else:
        return x.round()

def clampabs(input, min, max):
    assert(min >= 0 and max >=0)

    input[input > max] = max
    input[input < -max] = -max

    input[(input > torch.zeros_like(input)) & (input < min)] = min
    input[(input < torch.zeros_like(input)) & (input > -min)] = -min
    return input 

class ConcWeight():
    def __init__(self, data=None, base=0, bits=8):
        self.data = data 
        self.base = base
        self.bits = bits


def multipleof8(x: int) -> int:
    return (x + 7) & ~7

def fuseSignsAndShifts(signs: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    assert signs.shape == shifts.shape

    # Print the min and max of signs and shifts for debugging
    # print("Signs min:", signs.min().item(), "max:", signs.max().item())
    # print("Shifts min:", shifts.min().item(), "max:", shifts.max().item())


    # bits [7:6]: sign encoding — 00=zero, 01=positive, 11=negative
    # bits [5:0]: |shift|
    sign_bits = ((signs & 0x03) << 6).to(torch.int8)
    # print("Sign bits min:", sign_bits.min().item(), "max:", sign_bits.max().item())
    shift_part = shifts.abs().to(torch.int8) & 0x3F
    # print("Shift part min:", shift_part.min().item(), "max:", shift_part.max().item())
    out = torch.bitwise_or(sign_bits, shift_part)
    out.contiguous()
    return out
