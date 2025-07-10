from typing import Sequence

import torch
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType, BondType


class AtomFeaturizer:
    """
    Featurizes an RDKit atom using a multi-hot encoding directly as a torch.Tensor.

    The feature vector is composed of:
      - Atomic number (one-hot): 
          * When v2 is True: atoms 1-36 plus iodine (53).
          * Otherwise (v2 False): atoms 1-100.
      - Degree (one-hot) for degrees 0 to 5.
      - Formal charge (one-hot):
          * When v2 is True: [-2, -1, 1, 2, 0]
          * Otherwise: [-1, -2, 1, 2, 0]
      - Chiral tag (one-hot) for available chiral tags (0, 1, 2, 3).
      - Number of hydrogens (one-hot) for counts 0 to 4.
      - Hybridization (one-hot):
          * When v2 is True: [S, SP, SP2, SP2D, SP3, SP3D, SP3D2]
          * Otherwise: [SP, SP2, SP3, SP3D, SP3D2]
      - Aromaticity as a binary feature.
      - Mass (scaled, e.g. divided by 100).

    Each one-hot encoding includes an extra slot for unknown values.
    """

    def __init__(self,
                 v2: bool = True,
                 atomic_num: Sequence[int] = None,
                 degrees: Sequence[int] = None,
                 formal_charge: Sequence[int] = None,
                 chiral_tags: Sequence[int] = None,
                 num_Hs: Sequence[int] = None,
                 hybridizations: Sequence[HybridizationType] = None):
        if atomic_num is None:
            if v2:
                atomic_num = list(range(1, 37)) + [53]
            else:
                print("Using all atomic numbers from 1 to 100")
                atomic_num = list(range(1, 101))
        if degrees is None:
            degrees = list(range(6))
        if formal_charge is None:
            if v2:
                formal_charge = [-2, -1, 1, 2, 0]
            else:
                formal_charge = [-1, -2, 1, 2, 0]
        if chiral_tags is None:
            chiral_tags = list(range(4))  # 0, 1, 2, 3
        if num_Hs is None:
            num_Hs = list(range(5))
        if hybridizations is None:
            if v2:
                hybridizations = [
                    HybridizationType.S,
                    HybridizationType.SP,
                    HybridizationType.SP2,
                    HybridizationType.SP2D,
                    HybridizationType.SP3,
                    HybridizationType.SP3D,
                    HybridizationType.SP3D2,
                ]
            else:
                hybridizations = [
                    HybridizationType.SP,
                    HybridizationType.SP2,
                    HybridizationType.SP3,
                    HybridizationType.SP3D,
                    HybridizationType.SP3D2,
                ]

        self.atomic_nums = {num: i for i, num in enumerate(atomic_num)}
        self.degrees = {deg: i for i, deg in enumerate(degrees)}
        self.formal_charges = {fc: i for i, fc in enumerate(formal_charge)}
        self.chiral_tags = {ct: i for i, ct in enumerate(chiral_tags)}
        self.num_Hs = {nH: i for i, nH in enumerate(num_Hs)}
        self.hybridizations = {hyb: i for i, hyb in enumerate(hybridizations)}

        self._subfeats = [
            self.atomic_nums,
            self.degrees,
            self.formal_charges,
            self.chiral_tags,
            self.num_Hs,
            self.hybridizations,
        ]
        subfeat_sizes = [
            1 + len(self.atomic_nums),
            1 + len(self.degrees),
            1 + len(self.formal_charges),
            1 + len(self.chiral_tags),
            1 + len(self.num_Hs),
            1 + len(self.hybridizations),
        ]
        # Add 2 for Aromaticity and Mass.
        self.__size = sum(subfeat_sizes) + 2

    def __len__(self):
        return self.__size

    def __call__(self, a: Chem.rdchem.Atom) -> torch.Tensor:
        x = torch.zeros(self.__size, dtype=torch.float32)
        if a is None:
            return x

        feats = [
            a.GetAtomicNum(),
            a.GetTotalDegree(),
            a.GetFormalCharge(),
            int(a.GetChiralTag()),
            int(a.GetTotalNumHs()),
            a.GetHybridization(),
        ]
        i = 0
        for feat, choices in zip(feats, self._subfeats):
            j = choices.get(feat, len(choices))
            x[i + j] = 1.0
            i += len(choices) + 1
        x[i] = int(a.GetIsAromatic())
        x[i + 1] = a.GetMass() / 100.0
        return x


class BondFeaturizer:
    """
    Featurizes the bond information of a molecule.

    The feature vector is composed of:
      - Null flag (1 bit) when a bond is not present.
      - Bond type (one-hot) for known bond types (default: SINGLE, DOUBLE, TRIPLE, AROMATIC)
      - Conjugation (1 bit)
      - In ring (1 bit)
      - Stereochemistry (one-hot) for known stereochemistry values (default: 0-5)
    
    Output length is 14.
    """

    def __init__(self, bond_types: list = None, stereos: list = None):
        self.bond_types = bond_types or [
            BondType.SINGLE,
            BondType.DOUBLE,
            BondType.TRIPLE,
            BondType.AROMATIC,
        ]
        self.stereo = stereos or range(6)

        self.size = 1 + len(self.bond_types) + 2 + len(self.stereo) + 1

    def __len__(self):
        return self.size

    def __call__(self, b: Chem.rdchem.Bond) -> torch.Tensor:
        x = torch.zeros(self.size, dtype=torch.float32)

        if b is None:
            x[0] = 1
            return x

        i = 1
        bond_type = b.GetBondType()
        bt_bit, size = self.one_hot_index(bond_type, self.bond_types)
        if bt_bit != size:
            x[i + bt_bit] = 1

        i += size - 1
        x[i] = int(b.GetIsConjugated())
        x[i + 1] = int(b.IsInRing())
        i += 2

        stereo_bit, _ = self.one_hot_index(b.GetStereo(), self.stereo)
        x[i + stereo_bit] = 1

        return x

    @classmethod
    def one_hot_index(cls, x, xs: Sequence) -> tuple[int, int]:
        """Returns a tuple of the index of ``x`` in ``xs`` and ``len(xs) + 1`` if ``x`` is in ``xs``.
        Otherwise, returns a tuple with ``len(xs)`` and ``len(xs) + 1``."""
        n = len(xs)

        return xs.index(x) if x in xs else n, n + 1
