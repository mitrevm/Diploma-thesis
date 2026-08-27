#!/usr/bin/env python3
"""
Command-line script to generate config files for antibody-antigen docking
from a CSV file with two columns: first_molecule (TRGB1) and second_molecule (TRGC1).

Each row produces one config named like the molecules, e.g.,
  docking_<first_molecule>_<second_molecule>.cfg

The script performs the same template substitutions as the original script:
- Replaces the molecules list block with the two molecule paths
- Replaces the run_dir value with: run_<first>_<second>
"""

import os
import sys
import csv
import argparse


MODEL = "Model_4"
model_with_lower_case = MODEL.lower()


def build_molecules_list(first_molecule_path, second_molecule_path):
    return f"""molecules = [
    "{first_molecule_path}",
   "{second_molecule_path}"
]"""


def generate_config_for_pair(
    template_content,
    first_molecule_name,
    second_molecule_name,
    output_dir,
):
    """
    Generate a single config file for the given pair.
    """

    # Resolve molecule file paths
    first_molecule_path = f"/d/hpc/home/mm5129/Diplomska/{MODEL}/TRGB1/fold_{first_molecule_name}_{model_with_lower_case}.pdb"
    second_molecule_path = f"/d/hpc/home/mm5129/Diplomska/{MODEL}/TRGC1/fold_{second_molecule_name}_{model_with_lower_case}.pdb"

    # Create molecules list content
    molecules_list = build_molecules_list(first_molecule_path, second_molecule_path)

    # Replace the molecules section in template (keeping the same anchor text as the original script)
    new_content = template_content.replace(
        'molecules = [\n    "/d/hpc/home/mm5129/Diplomska/TRGB1/c0jmj2.pdb",\n   "/d/hpc/home/mm5129/Diplomska/TRGC1/c0jmn2_chainB.pdb"\n #  "/home/skikk/Documents/Faks/Diplomska/trgc1_COJMN2.pdb"\n]',
        molecules_list,
    )

    # Replace the run_dir with a deterministic directory per pair
    run_dir = f"run_{first_molecule_name}_{second_molecule_name}"
    new_content = new_content.replace(
        'run_dir = "run_par2_mixed_all_c0jmn2"',
        f'run_dir = "{run_dir}"',
    )

    # Filename like the molecules
    output_filename = f"docking_{first_molecule_name}_{second_molecule_name}.cfg"
    os.makedirs(f"{MODEL}_Dockings/", exist_ok=True)
    output_path = os.path.join(f"{MODEL}_Dockings/", output_filename)

    with open(output_path, "w") as f_out:
        f_out.write(new_content)

    return output_filename, run_dir, first_molecule_path, second_molecule_path


def read_csv_pairs(csv_path):
    """
    Read CSV and yield (first, second) tuples.
    Accepts files with or without a header. If a header is present,
    it will be skipped if it contains non-molecule-looking strings.
    """
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if len(row) < 2:
                continue
            first, second = row[0].strip(), row[1].strip()

            # Heuristic: skip a potential header row if it contains letters like 'first'/'second'
            headerish = (first.lower() in {"first", "first_molecule", "trgb1"}) or (
                second.lower() in {"second", "second_molecule", "trgc1"}
            )
            if headerish:
                continue

            if not first or not second:
                continue
            yield first, second


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate config files for antibody-antigen docking for each pair "
            "listed in a CSV (first_molecule, second_molecule)."
        )
    )
    parser.add_argument(
        "csv_path",
        help="Path to CSV file with two columns: first_molecule, second_molecule",
    )
    parser.add_argument(
        "-t",
        "--template",
        default="old_configs/docking_antibody_antigen_mesano_all_5.cfg",
        help=(
            "Path to template config file (default: docking_antibody_antigen_mesano_all_5.cfg)"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default="generated_configs",
        help="Output directory for generated configs (default: generated_configs)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file '{args.csv_path}' not found!")
        sys.exit(1)

    if not os.path.exists(args.template):
        print(f"Error: Template file '{args.template}' not found!")
        sys.exit(1)

    # Prepare output directory
    os.makedirs(args.output, exist_ok=True)

    # Load template once
    with open(args.template, "r") as f_tpl:
        template_content = f_tpl.read()

    print(f"Reading pairs from: {os.path.abspath(args.csv_path)}")
    print(f"Template: {os.path.abspath(args.template)}")
    print(f"Output directory: {os.path.abspath(args.output)}")
    print("-" * 50)

    total = 0
    for first_molecule, second_molecule in read_csv_pairs(args.csv_path):
        # Warn if molecule files are missing, but continue
        first_molecule_path = f"/d/hpc/home/mm5129/Diplomska/TRGB1/{first_molecule}.pdb"
        if not os.path.exists(first_molecule_path):
            print(
                f"Warning: First molecule file '{first_molecule_path}' not found! "
                "Config will be generated, but you may need to update the path."
            )

        second_molecule_path = (
            f"/d/hpc/home/mm5129/Diplomska/TRGC1/{second_molecule}_chainB.pdb"
        )
        if not os.path.exists(second_molecule_path):
            print(
                f"Warning: Second molecule file '{second_molecule_path}' not found! "
                "Config will be generated, but you may need to update the path."
            )

        output_filename, run_dir, first_path, second_path = generate_config_for_pair(
            template_content,
            first_molecule,
            second_molecule,
            args.output,
        )

        print(f"Generated: {output_filename}")
        print(f"  Run directory: {run_dir}")
        print(f"  Molecules: {first_molecule} + {second_molecule}")
        total += 1

    if total == 0:
        print("No valid pairs found in the CSV. Nothing generated.")
        sys.exit(2)

    print("\n" + "=" * 50)
    print(f"Completed successfully. Generated {total} config(s).")
    print(f"Check the directory: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()


