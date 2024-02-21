## 绘制RMSE散点图, 用来直观的比对ChirlFormer与Baseline的性能差别

import math
import random
import json
import collections
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn

from matplotlib import pyplot as plt
from collections import Counter
from scipy.interpolate import interp1d