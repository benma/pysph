#!/bin/sh

# builds cg.py and cg_gl.py from their headers.
# see http://starship.python.net/crew/theller/ctypes/old/codegen.html

h2xml.py /usr/include/Cg/cg.h -o cg.xml -q -c
xml2py cg.xml > cg.py
h2xml.py /usr/include/Cg/cgGL.h -o cgGL.xml -q -c
xml2py cgGL.xml > cg_gl.py
rm cg.xml cgGL.xml
