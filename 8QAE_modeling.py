#!/usr/bin/env python3
"""Fill missing residues from FASTA into a broken PDB and clean geometry.

Workflow (high level):
1) Load the input PDB as a Rosetta pose.
2) Globally align FASTA sequence to the pose sequence (Needleman–Wunsch).
3) For FASTA positions that align to gaps in the pose, insert new residues *into the
    existing pose* (preserves the experimental coordinates you already trust).
4) Optionally run Rosetta IdealizeMover.
5) Optionally run OpenMM minimization with strong positional restraints on backbone
    atoms so only subtle backbone motion is allowed (sidechains relax more freely).

Outputs are written in the working directory as:
- complete_structure_filled.pdb
- complete_structure_filled_idealized.pdb (if --idealize)
- ..._openmm_minimized.pdb (if --openmm-minimize)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pyrosetta import init, pose_from_pdb
from pyrosetta.rosetta import core, protocols
from pyrosetta.rosetta.core.chemical import aa_from_oneletter_code, name_from_aa
from pyrosetta.rosetta.core.conformation import ResidueFactory

AA_THREE_TO_ONE = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V',
    'TRP': 'W', 'TYR': 'Y',
}


def parse_fasta(fasta_path: str) -> str:
    """Read a FASTA file and return the sequence as a single string."""
    with open(fasta_path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('>')]
    return ''.join(lines)


def parse_pdb_residues(pdb_path: str):
    pdb_residues = []
    seen = set()
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            resname = line[17:20].strip()
            chain = line[21].strip()
            resseq = int(line[22:26])
            icode = line[26].strip()
            key = (chain, resseq, icode)
            if key in seen:
                continue
            seen.add(key)
            pdb_residues.append({'resname': resname, 'chain': chain, 'resseq': resseq, 'icode': icode})
    return pdb_residues


def pdb_to_sequence(pdb_residues) -> str:
    return ''.join(AA_THREE_TO_ONE.get(r['resname'], 'X') for r in pdb_residues)


def needleman_wunsch_map(
    seq_a: str,
    seq_b: str,
    match: int = 2,
    mismatch: int = -1,
    gap: int = -2,
):
    """Return mapping from positions in seq_a -> 0-based index in seq_b (or None).

    Uses a simple global alignment (Needleman–Wunsch). This avoids the "scan until mismatch"
    behavior that breaks badly for repetitive sequences.

    This mapping is the key bridge between FASTA indices and pose indices.
    """

    n = len(seq_a)
    m = len(seq_b)
    # score matrix
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    # traceback: 0 diag, 1 up (gap in B), 2 left (gap in A)
    tb = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + gap
        tb[i][0] = 1
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + gap
        tb[0][j] = 2

    for i in range(1, n + 1):
        a = seq_a[i - 1]
        for j in range(1, m + 1):
            b = seq_b[j - 1]
            s_diag = dp[i - 1][j - 1] + (match if a == b else mismatch)
            s_up = dp[i - 1][j] + gap
            s_left = dp[i][j - 1] + gap
            best = s_diag
            move = 0
            if s_up > best:
                best = s_up
                move = 1
            if s_left > best:
                best = s_left
                move = 2
            dp[i][j] = best
            tb[i][j] = move

    # traceback to build mapping
    mapping = [None] * n
    i, j = n, m
    while i > 0 or j > 0:
        move = tb[i][j]
        if i > 0 and j > 0 and move == 0:
            # aligned a[i-1] with b[j-1]
            mapping[i - 1] = j - 1
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or move == 1):
            # a[i-1] aligned to gap
            mapping[i - 1] = None
            i -= 1
        else:
            # gap in a
            j -= 1

    return mapping


def _make_residue_for_pose(pose, one_letter: str):
    """Create a new polymer residue compatible with this pose's residue type set."""
    # Important: create residues from the *same* residue type set as the pose.
    # Mixing residue objects from other poses/type-sets can crash or corrupt geometry.
    one_letter = one_letter.upper()
    if one_letter == 'X':
        raise ValueError("Cannot insert unknown residue 'X'")
    aa = aa_from_oneletter_code(one_letter)
    name3 = name_from_aa(aa)
    rts = pose.residue_type_set_for_pose()
    rtype = rts.name_map(name3)
    return ResidueFactory.create_residue(rtype)


def fill_breaks_by_insertion(pose, fasta_seq: str, pose_seq: str):
    """Insert missing FASTA positions into an existing pose loaded from PDB.

    This preserves original PDB coordinates and only inserts new residues where
    the alignment indicates gaps in the PDB.
    """

    # Align FASTA to the current pose sequence and insert residues only where the
    # pose has alignment gaps. This preserves original coordinates for all residues
    # already present in the PDB.
    mapping = needleman_wunsch_map(fasta_seq, pose_seq)
    inserted_positions = []
    inserted = 0
    last_anchor_pose_pos = None

    for fasta_i0, pdb_i0 in enumerate(mapping):
        fasta_pos = fasta_i0 + 1

        if pdb_i0 is not None:
            # Existing residue: update anchor for subsequent insertions.
            last_anchor_pose_pos = (pdb_i0 + 1) + inserted
            continue

        # Gap in PDB => insert this residue.
        aa = fasta_seq[fasta_i0]
        new_res = _make_residue_for_pose(pose, aa)

        if last_anchor_pose_pos is None:
            # Insert before the first residue.
            pose.prepend_polymer_residue_before_seqpos(new_res, 1, True)
            inserted += 1
        else:
            pose.append_polymer_residue_after_seqpos(new_res, last_anchor_pose_pos, True)
            inserted += 1
            last_anchor_pose_pos += 1

        inserted_positions.append(fasta_pos)

    return inserted_positions


def renumber_pose_inplace(pose, chain: str = 'A'):
    """Make residue numbering contiguous to avoid viewer artifacts (e.g. PyMOL gaps)."""
    # This does not change pose geometry; it just rewrites PDBInfo metadata so
    # residue numbers are 1..N in a single chain.
    pdbinfo = pose.pdb_info()
    if pdbinfo is None:
        pdbinfo = core.pose.PDBInfo(pose)
        pose.pdb_info(pdbinfo)

    if not chain or len(chain) != 1:
        raise ValueError("chain must be a single character, e.g. 'A'")

    for i in range(1, pose.total_residue() + 1):
        pdbinfo.set_resinfo(i, chain, i, ' ')


def report_peptide_bond_gaps(pose, threshold_angstrom: float = 2.1):
    """Print any suspicious C(i)-N(i+1) distances that would look like breaks in PyMOL."""
    # A real peptide bond is ~1.33 Å. Anything > ~2 Å is very suspicious.
    # This is a simple diagnostic to distinguish true chain breaks vs viewer artifacts.
    gaps = []
    for i in range(1, pose.total_residue()):
        r1 = pose.residue(i)
        r2 = pose.residue(i + 1)
        if not (r1.has('C') and r2.has('N')):
            continue
        d = (r1.xyz('C') - r2.xyz('N')).norm()
        if d > threshold_angstrom:
            gaps.append((i, i + 1, float(d)))

    if gaps:
        print('[WARN] Detected large peptide-bond gaps (C–N distance):')
        for i, j, d in gaps[:25]:
            print(f'  {i}->{j}: {d:.3f} Å')
        if len(gaps) > 25:
            print(f'  ... and {len(gaps) - 25} more')
    else:
        print('[INFO] No large peptide-bond gaps detected (by C–N distance).')

    return gaps


def openmm_minimize_restrained(
    pdb_in: str,
    pdb_out: str,
    *,
    forcefield_xml: str = 'amber14-all.xml',
    implicit: bool = False,
    k_kj_mol_nm2: float = 500.0,
    include_cb: bool = False,
    exclude_pdb_resseq: set[int] | None = None,
    tolerance_kj_mol_nm: float = 10.0,
    max_iter: int = 2000,
    strip_hydrogens: bool = False,
    platform_name: str = 'CPU',
) -> None:
    """Run OpenMM LocalEnergyMinimizer with positional restraints on backbone atoms.

    Restraints are applied to N/CA/C/O (and optionally CB) atoms in protein residues.
    """

    try:
        from openmm import unit
        from openmm import app
        import openmm as mm
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "OpenMM import failed. Install it in this environment, e.g. `pip install openmm`. "
            f"Original error: {e}"
        )

    # Read PDB into OpenMM.
    pdb = app.PDBFile(pdb_in)

    if implicit:
        ff = app.ForceField(forcefield_xml, 'implicit/gbn2.xml')
    else:
        ff = app.ForceField(forcefield_xml)

    # Add hydrogens for stable minimization.
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(ff)

    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
    )

    # Positional restraints (harmonic) around the *starting* coordinates.
    # Units: OpenMM uses nm; the k parameter is in kJ/mol/nm^2.
    backbone_names = {'N', 'CA', 'C', 'O'}
    if include_cb:
        backbone_names.add('CB')

    restraint = mm.CustomExternalForce('0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)')
    restraint.addGlobalParameter('k', float(k_kj_mol_nm2))
    restraint.addPerParticleParameter('x0')
    restraint.addPerParticleParameter('y0')
    restraint.addPerParticleParameter('z0')

    ref_positions = modeller.positions
    restrained = 0
    for atom in modeller.topology.atoms():
        res = atom.residue
        if exclude_pdb_resseq:
            # res.id is PDB residue number as a string when keepIds=True; ignore if unparsable.
            try:
                if int(res.id) in exclude_pdb_resseq:
                    continue
            except Exception:
                pass
        if atom.name not in backbone_names:
            continue
        # Heuristic: treat as protein residue if it has a CA atom.
        if not any(a.name == 'CA' for a in res.atoms()):
            continue
        p = ref_positions[atom.index]
        restraint.addParticle(atom.index, [p.x, p.y, p.z])
        restrained += 1

    system.addForce(restraint)

    # LangevinIntegrator is fine here; we're not running dynamics, just minimizing.
    integrator = mm.LangevinIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)
    platform = mm.Platform.getPlatformByName(platform_name)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)

    mm.LocalEnergyMinimizer.minimize(
        simulation.context,
        tolerance=float(tolerance_kj_mol_nm) * unit.kilojoule_per_mole / unit.nanometer,
        maxIterations=int(max_iter),
    )

    out_top = modeller.topology
    out_pos = simulation.context.getState(getPositions=True).getPositions()

    if strip_hydrogens:
        filtered = app.Modeller(out_top, out_pos)
        to_delete = [a for a in filtered.topology.atoms() if a.element == app.element.hydrogen]
        filtered.delete(to_delete)
        out_top, out_pos = filtered.topology, filtered.positions

    with open(pdb_out, 'w') as fh:
        app.PDBFile.writeFile(out_top, out_pos, fh, keepIds=True)

    print(f"[INFO] OpenMM restrained atoms: {restrained}")
    print(f"[SUCCESS] Wrote {pdb_out}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fill missing residues into a PDB (PyRosetta) and optionally OpenMM-minimize with backbone restraints."
    )
    p.add_argument('input_pdb')
    p.add_argument('input_fasta')
    p.add_argument('--idealize', action='store_true', help='Run Rosetta IdealizeMover after insertion')

    p.add_argument(
        '--openmm-minimize',
        action='store_true',
        help='Run OpenMM energy minimization with backbone positional restraints on the final output PDB',
    )
    p.add_argument('--openmm-k', type=float, default=500.0, help='Restraint strength (kJ/mol/nm^2)')
    p.add_argument('--openmm-include-cb', action='store_true', help='Also restrain CB atoms')
    p.add_argument('--openmm-implicit', action='store_true', help='Use implicit solvent (GBn2)')
    p.add_argument('--openmm-forcefield', default='amber14-all.xml', help='OpenMM forcefield XML')
    p.add_argument('--openmm-tolerance', type=float, default=10.0, help='Minimizer tolerance (kJ/mol/nm)')
    p.add_argument('--openmm-max-iter', type=int, default=2000, help='Max minimization iterations')
    p.add_argument('--openmm-strip-hydrogens', action='store_true', help='Strip hydrogens in minimized output')
    p.add_argument('--openmm-platform', default='CPU', help='OpenMM platform name (e.g. CPU, CUDA, OpenCL)')
    p.add_argument(
        '--peptide-gap-threshold',
        type=float,
        default=2.1,
        help='Report/flag peptide-bond gaps when C(i)-N(i+1) distance exceeds this (Å)',
    )
    p.add_argument(
        '--openmm-unrestrain-around-gaps',
        type=int,
        default=2,
        help='If >0, do not restrain residues within +/- this window around any flagged peptide-bond gap (default: 2)',
    )

    return p.parse_args()

def main():
    args = _parse_args()
    input_pdb = args.input_pdb
    input_fasta = args.input_fasta
    do_idealize = bool(args.idealize)

    fasta_seq = parse_fasta(input_fasta)

    # Initialize Rosetta and load the experimental PDB pose.
    init()
    pose = pose_from_pdb(input_pdb)
    pose_seq = pose.sequence()

    print(f"[INFO] FASTA length: {len(fasta_seq)} | Pose length: {pose.total_residue()} | Pose seq length: {len(pose_seq)}")

    # Sanity check the alignment before mutating the pose. If FASTA/PDB mismatch
    # is too large, inserting residues based on a bad mapping can corrupt geometry.
    mapping = needleman_wunsch_map(fasta_seq, pose_seq)
    mapped_pose_idxs = {j for j in mapping if j is not None}
    if len(mapped_pose_idxs) < int(0.8 * len(pose_seq)):
        raise RuntimeError(
            f"Alignment looks suspicious (only {len(mapped_pose_idxs)}/{len(pose_seq)} pose residues mapped). "
            "Refusing to insert residues to avoid corrupting the pose."
        )
    # Also ensure we don't have many pose residues absent from FASTA.
    if len(mapped_pose_idxs) != len(pose_seq):
        missing_in_fasta = len(pose_seq) - len(mapped_pose_idxs)
        print(f"[WARN] {missing_in_fasta} pose residues did not map to FASTA (FASTA may not match the PDB exactly).")

    # Fill missing regions by inserting residues into the existing pose.
    inserted_positions = fill_breaks_by_insertion(pose, fasta_seq, pose_seq)
    print(f"[INFO] Inserted {len(inserted_positions)} residues at FASTA positions: {inserted_positions[:20]}" + (" ..." if len(inserted_positions) > 20 else ""))
    print(f"[INFO] Final pose length: {pose.total_residue()} (FASTA length: {len(fasta_seq)})")

    # Make output contiguous for viewers and check for true geometry gaps.
    renumber_pose_inplace(pose, chain='A')
    final_gaps = report_peptide_bond_gaps(pose, threshold_angstrom=float(args.peptide_gap_threshold))

    out_pdb: str
    if do_idealize:
        # Idealize can help fix bond lengths/angles after insertions, but it can
        # also expose true breaks (e.g., if residues are far apart).
        try:
            idealize = protocols.idealize.IdealizeMover()
            idealize.apply(pose)
        except Exception as e:
            raise RuntimeError(f"Idealize failed: {e}")

        # Idealize can change geometry; re-check.
        final_gaps = report_peptide_bond_gaps(pose, threshold_angstrom=float(args.peptide_gap_threshold))

        out_pdb = "complete_structure_filled_idealized.pdb"
        pose.dump_pdb(out_pdb)
        print(f"[SUCCESS] Wrote {out_pdb}")
    else:
        out_pdb = "complete_structure_filled.pdb"
        pose.dump_pdb(out_pdb)
        print(f"[SUCCESS] Wrote {out_pdb}")

    if args.openmm_minimize:
        # OpenMM minimization step: primarily to remove local artifacts (clashes,
        # strained geometry) while keeping the overall fold fixed.
        #
        # Key trick: if we detected a peptide-bond gap (e.g., i->i+1 C–N distance is large),
        # we intentionally *do not restrain* a small window around that region so the
        # backbone can move enough to close the break.
        in_path = Path(out_pdb)
        out_min = str(in_path.with_name(in_path.stem + "_openmm_minimized.pdb"))

        exclude_resseq: set[int] | None = None
        if int(args.openmm_unrestrain_around_gaps) > 0 and final_gaps:
            w = int(args.openmm_unrestrain_around_gaps)
            exclude_resseq = set()
            nres = pose.total_residue()
            for i, j, _d in final_gaps:
                lo = max(1, int(i) - w)
                hi = min(nres, int(j) + w)
                exclude_resseq.update(range(lo, hi + 1))
            print(
                f"[INFO] OpenMM: unrestraining {len(exclude_resseq)} residues around peptide gaps "
                f"(window={w}). Example: {sorted(exclude_resseq)[:20]}" + (" ..." if len(exclude_resseq) > 20 else "")
            )

        openmm_minimize_restrained(
            out_pdb,
            out_min,
            forcefield_xml=args.openmm_forcefield,
            implicit=args.openmm_implicit,
            k_kj_mol_nm2=args.openmm_k,
            include_cb=args.openmm_include_cb,
            exclude_pdb_resseq=exclude_resseq,
            tolerance_kj_mol_nm=args.openmm_tolerance,
            max_iter=args.openmm_max_iter,
            strip_hydrogens=args.openmm_strip_hydrogens,
            platform_name=args.openmm_platform,
        )


if __name__ == "__main__":
    main()
