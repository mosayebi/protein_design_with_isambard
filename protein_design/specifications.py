import numpy as np
import copy
from scipy.spatial import KDTree
from functools import lru_cache

import isambard
from ampal.geometry import spherical_to_cartesian
from isambard.modelling import pack_side_chains_scwrl
import budeff

from .utils import visualise


class Monomer(isambard.ampal.Assembly):
    '''An object inhereted from the isambard.ampal.Assembly with two additional methods:
        - coordinat_sytem: (which is a property) and returns a local coordinate system of the Monomer
        - visualise(): visualises the Monomer
    '''

    def __init__(self, monomer, get_coordinate_system_func):
        super().__init__()
        self._molecules = monomer._molecules
        self.id = monomer.id
        self.get_coordinate_system_func = get_coordinate_system_func

    @property
    def coordinate_system(self):
        return self.get_coordinate_system_func(self)

    def visualise(self, coordinate_system=False):
        view = visualise(self)
        if coordinate_system:
            for v, name, color in zip(self.coordinate_system,
                                      ['x', 'y', 'z'],
                                      [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
                                      ):
                view.shape.add_arrow([0, 0, 0], 5*v, color, 0.5, name)
        return view


class Dimer(isambard.ampal.Assembly):
    def __init__(self, monomer: Monomer, r=20, theta=0, phi=0, alpha=0, beta=0, gamma=0):
        ''' Build a parametric model for the dimer from the monomer of type `Monomer`.
        The relative position and orientation of the second monomer is specified relative
        to the first monomer using spherical coordinate system with parameters (`r`, `theta`, `phi`)
        and Euler angles (`alpha`, `beta`, `gamma`), respectively.

        Conventional ranges for the parameters are
        - r : [0, inf]
        - theta : [0, 180]
        - phi : [0, 360]
        - alpha & gamma : [-180, 180]
        - beta : [-90, 90]
        '''
        super().__init__()

        self.r = r
        self.theta = theta
        self.phi = phi

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        m1 = copy.deepcopy(monomer)
        m2 = copy.deepcopy(monomer)

        # euler rotations (see https://en.wikipedia.org/wiki/Euler_angles)
        m2.rotate(angle=self.alpha, axis=m2.coordinate_system[2])
        m2.rotate(angle=self.beta, axis=m2.coordinate_system[0])
        m2.rotate(angle=self.gamma, axis=m2.coordinate_system[2])

        T = spherical_to_cartesian(self.r, self.phi, self.theta)
        m2.translate(T)

        self._molecules = (m1 + m2)._molecules
        self.id = f"dimer({m1.id})"
        self.relabel_all()
        for m in self._molecules:
            m.parent = self

    @lru_cache(maxsize=1)
    def num_overlaps(self, cutoff=1.0, workers=1):
        '''Check (efficiently) for the number of overlaps between backbone atoms
        (i.e. number of pairs of atoms whose distance are closer than cutoff1)
        '''
        overlaps = 0
        vs = [np.array([atom._vector for atom in m.backbone.get_atoms()])
              for m in self._molecules]
        if len(vs) > 1:
            tree = KDTree(vs[0])
            for v in vs[1:]:
                mindist, minid = tree.query(
                    v, 1, distance_upper_bound=cutoff, workers=workers)
                overlaps += (mindist < cutoff).sum()
        return overlaps

    @property
    def overlap(self):
        return self.num_overlaps(cutoff=1.0) > 0

    def visualise(self):
        return visualise(self)


def build_model(spec_seq_params):
    '''check for overlap first, if there is no overlap pack side-chains'''
    specification, sequences, params = spec_seq_params
    model = specification(*params)
    if not model.overlap:
        model = pack_side_chains_scwrl(model, sequences)
    return model


def get_buff_total_energy(ampal_object, overlap_energy=0):
    # if there was overlap, the ampal_object is still of type Dimer,
    # otherwise the packed-side-chain model is of type ampal.assembly.Assembly
    if type(ampal_object) is Dimer:
        energy = overlap_energy
    else:
        energy = budeff.get_internal_energy(ampal_object).total_energy
    return energy
