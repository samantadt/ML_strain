"""Command-line calculation workflow; all paths are explicit."""

import argparse
import json
from pathlib import Path

from .data import read_data


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="strain2d", description="Material-specific 2D strain surrogate workflow"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("sample", help="Maximin Latin hypercube plus zero-strain reference")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--low", type=float, default=-2.0)
    p.add_argument("--high", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    p = sub.add_parser("demo-data", help="Generate synthetic data for SOFTWARE TESTING ONLY")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    p = sub.add_parser("check", help="Strict dataset validation")
    p.add_argument("data")
    p = sub.add_parser("train", help="Fit, select on validation, calibrate, evaluate and report")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config")
    p = sub.add_parser("predict", help="Load a trusted saved run and predict a strain CSV")
    p.add_argument("--run", required=True)
    p.add_argument("--points", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser(
        "screen", help="Find supported candidates subject to gap interval and stability-score constraints"
    )
    p.add_argument("--run", required=True)
    p.add_argument("--points", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--gap-low", type=float, required=True)
    p.add_argument("--gap-high", type=float, required=True)
    p.add_argument("--stability-score", type=float, default=0.9)
    p.add_argument("--limit", type=int, default=30)
    p = sub.add_parser("suggest", help="Diverse new DFT points; ensemble disagreement if available")
    p.add_argument("--run", required=True)
    p.add_argument("--points", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=20)
    p = sub.add_parser("interpret", help="Held-out permutation importance; diagnostic only")
    p.add_argument("--run", required=True)
    p.add_argument("--repeats", type=int, default=10)
    p = sub.add_parser("sensitivity", help="Bandgap finite differences in eV per percentage point")
    p.add_argument("--run", required=True)
    p.add_argument("--points", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--step", type=float, default=0.05)
    p = sub.add_parser("example-structure", help="Unrelaxed h-BN starter; NOT a converged reference")
    p.add_argument("--out", required=True)
    p = sub.add_parser("dft-prepare", help="Fixed-cell VASP inputs and cluster templates")
    p.add_argument("--poscar", required=True)
    p.add_argument("--points", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--material", required=True, help="One material/phase/numerical-protocol identifier")
    p.add_argument("--cutoff", type=float, default=600)
    p.add_argument("--kmesh", type=int, nargs=3, default=[6, 6, 1])
    p = sub.add_parser("verify-vasp", help="Check final VASP output convergence and parse mesh-resolved gap")
    p.add_argument("xml")
    p = sub.add_parser("dft-collect", help="Collect converged VASP cases; rejected cases get an audit CSV")
    p.add_argument("--calc", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--phonon-tolerance", type=float, default=20.0)
    p.add_argument("--direct-tolerance", type=float, default=0.01)
    p = sub.add_parser("phonon-prepare", help="Generate finite-displacement VASP supercells")
    p.add_argument("--case", required=True)
    p.add_argument("--supercell", type=int, nargs=3, default=[4, 4, 1])
    p.add_argument("--force-kmesh", type=int, nargs=3, default=[2, 2, 1])
    p.add_argument("--displacement", type=float, default=0.01)
    p = sub.add_parser("phonon-collect", help="Validate forces, build force constants and sample q mesh")
    p.add_argument("--case", required=True)
    p.add_argument("--qmesh", type=int, nargs=3, default=[9, 9, 1])
    args = parser.parse_args(argv)
    if args.command in {"sample", "demo-data"}:
        from .sampling import lhs, synthetic

        df = (
            lhs(args.n, args.low, args.high, args.seed)
            if args.command == "sample"
            else synthetic(args.n, args.seed)
        )
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"Wrote {len(df)} points to {args.out}")
    elif args.command == "check":
        df = read_data(args.data)
        print(
            json.dumps(
                dict(
                    rows=len(df),
                    material=df.material_id.iloc[0],
                    source=df.data_source.iloc[0],
                    labels={
                        t: int(df[t].notna().sum())
                        for t in df
                        if t.startswith(("bandgap_", "strain_energy_", "is_"))
                    },
                ),
                indent=2,
            )
        )
    elif args.command == "train":
        from .workflow import train

        print(json.dumps(train(args.data, args.out, args.config), indent=2))
    elif args.command == "predict":
        from .workflow import load_bundle, predict_frame

        predict_frame(load_bundle(args.run), read_data(args.points, labels=False)).to_csv(
            args.out, index=False
        )
    elif args.command == "screen":
        from .workflow import screen

        print(
            f"Selected {screen(args.run, args.points, args.out, args.gap_low, args.gap_high, args.stability_score, args.limit)} candidates; validate by DFT"
        )
    elif args.command == "suggest":
        from .workflow import suggest

        print(f"Selected {suggest(args.run, args.points, args.out, args.n)} DFT candidates")
    elif args.command == "interpret":
        from .analysis import interpret

        print(json.dumps(interpret(args.run, args.repeats), indent=2))
    elif args.command == "sensitivity":
        from .workflow import sensitivity

        sensitivity(args.run, args.points, args.out, args.step)
    elif args.command == "example-structure":
        from ase.build import graphene
        from ase.io import write

        atoms = graphene(formula="BN", a=2.50, vacuum=10.0)
        write(args.out, atoms, format="vasp", direct=True)
        print("UNRELAXED h-BN starter only: converge the reference geometry first")
    elif args.command == "dft-prepare":
        from .dft import prepare

        print(
            json.dumps(
                prepare(args.poscar, args.points, args.out, args.material, args.cutoff, args.kmesh), indent=2
            )
        )
    elif args.command == "verify-vasp":
        from .dft import parse_vasp

        print(json.dumps(parse_vasp(args.xml), indent=2))
    elif args.command == "dft-collect":
        from .dft import collect

        accepted, rejected = collect(args.calc, args.out, args.phonon_tolerance, args.direct_tolerance)
        print(f"Accepted {accepted}; rejected {rejected}; see {args.out}.rejected.csv")
    elif args.command == "phonon-prepare":
        from .phonons import prepare_phonons

        print(
            f"Prepared {prepare_phonons(args.case, args.supercell, args.displacement, args.force_kmesh)} displacements"
        )
    elif args.command == "phonon-collect":
        from .phonons import collect_phonons

        print(json.dumps(collect_phonons(args.case, args.qmesh), indent=2))
