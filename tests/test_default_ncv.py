import dynamite
from slepc4py import SLEPc
from petsc4py import PETSc
from dynamite.opetator import Operator, Identity
from dynamite.states import State
from dynamite.config import config
from dynamite.computations import evolve

L = 20
config.L = L
t = 10

def build_hamiltonian():
    H = Identity()

if __name__ == '__main__':
    mfn = SLEPc.MFN().create()
    h = build_hamiltonian()

    st = State(state='random')
    h.evolve(st, t)
    # nev, ncv, _ = mfn.getDimensions()
    mfn.setOperator(h)
    

