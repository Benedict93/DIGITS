# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from . import option_list

def test_pt_import():
    """
    Tests if pytorch can be imported, returns if it went okay and optional error.
    """
    try:
        import torch  # noqa
        return True
    except ImportError:
        return False

pt_enabled = test_pt_import()

if not pt_enabled:
    print('pytorch support disabled.')

if pt_enabled:
    option_list['pytorch'] = {
        'enabled': True
    }
else:
    option_list['pytorch'] = {
        'enabled': False
    }
