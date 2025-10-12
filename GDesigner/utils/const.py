#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path


GDesigner_ROOT = Path(os.path.realpath(os.path.join(os.path.split(__file__)[0], "../..")))

# Alias for AnyMAC naming without changing existing references
AnyMAC_ROOT = GDesigner_ROOT
