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
      - (v2=True only) Null flag (1 bit) when a bond is not present.
      - Bond type (one-hot) for known bond types (default: SINGLE, DOUBLE, TRIPLE, AROMATIC)
      - Conjugation (1 bit)
      - In ring (1 bit)
      - Stereochemistry (one-hot) for known stereochemistry values (default: 0-5)
    
    When v2 is True, the output length is 15.
    When v2 is False, the output length is 14.
    """

    def __init__(self, v2: bool = True, bond_types: list = None, stereos: list = None):
        self.v2 = v2
        if bond_types is None:
            bond_types = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]
        if stereos is None:
            stereos = list(range(6))  # Known stereochemistry values: 0-5

        self.bond_types = bond_types
        self.stereos = stereos

        # Precompute dictionary lookups for faster access.
        self.bond_type_to_idx = {bt: i for i, bt in enumerate(self.bond_types)}
        self.stereo_to_idx = {st: i for i, st in enumerate(self.stereos)}

        # Both v1 and v2 reserve an extra slot for unknown bond type.
        self.size_bond_types = len(self.bond_types) + 1
        self.size_stereo = len(self.stereos) + 1

        # v2 includes an extra null flag at the beginning.
        null_flag_size = 1 if self.v2 else 0
        base_size = null_flag_size + self.size_bond_types + 1 + 1 + self.size_stereo
        self.size = base_size

    def __len__(self):
        return self.size

    def __call__(self, b: Chem.rdchem.Bond) -> torch.Tensor:
        x = torch.zeros(self.size, dtype=torch.float32)
        index = 0
        if b is None:
            if self.v2:
                x[0] = 1.0  # Set null flag if bond is absent.
            return x
        if self.v2:
            x[0] = 0.0  # Bond exists.
            index = 1
        else:
            # v1: no null flag, so start at index 0.
            index = 0

        # Bond type one-hot encoding.
        bt = b.GetBondType()
        idx = self.bond_type_to_idx.get(bt, len(self.bond_types))
        # For v1-style, shift the index by 1 so that position 0 is reserved for unknown.
        if not self.v2:
            idx += 1
        x[index + idx] = 1.0
        index += self.size_bond_types

        # Conjugation (1 bit).
        x[index] = 1.0 if b.GetIsConjugated() else 0.0
        index += 1

        # In ring (1 bit).
        x[index] = 1.0 if b.IsInRing() else 0.0
        index += 1

        # Stereochemistry one-hot encoding.
        st = int(b.GetStereo())
        idx = self.stereo_to_idx.get(st, len(self.stereos))
        x[index + idx] = 1.0

        return x
