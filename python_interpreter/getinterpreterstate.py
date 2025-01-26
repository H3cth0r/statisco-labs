import interpstate
import dis
import math
import testmod

# List imported modules
modules = interpstate.inspect()

dis.dis(testmod.printhi)
dis.dis(interpstate.inspect)
