#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from pykeyboard import *
from pymouse import *
from sys import *


m = PyMouse()
k = PyKeyboard()

if (sys.argv[1] == "vim"):
    x_dim, y_dim = m.screen_size()
    k.press_keys(['Command','Tab'])
    #m.click(x_dim/2+100, y_dim/2+100)
    #k.type_string('~m')
    #k.type_string(':MarkdownPreview')
else:
    k.press_keys(['Control','Alternate',sys.argv[1]])
    k.release_key('Control')
    k.release_key('Alternate')
    k.release_key(sys.argv[1])

#m.click(404.45623779296875, 50.3455810546875)

#k.press_key('Alternate')
#k.press_key('Control')
#k.press_key('Command')
#k.tap_key('c')

#k.release_key('Alternate')

